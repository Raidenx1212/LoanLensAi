import math
from typing import Dict, List, Tuple, Optional
from app.schemas.loan import (
    ExtractedLoanTerms,
    ExtractedFieldInfo,
    ExtractedCharge,
    CrossValidationResult,
    CrossValidationStatus,
    FieldStatus,
    ChargeCategory,
    LoanAuditReport,
    DiscrepancyItem,
    DiscrepancySeverity,
    InterestRateType,
)


class DeterministicLoanAuditor:
    """
    Pure mathematical, cross-validated, zero-invention audit engine.
    - Never converts missing values into 0.0.
    - Never invents GST rates, SAC codes, RBI circular numbers, or compliance scores.
    - Separates confirmed upfront, financed, conditional, and third-party charges.
    - Cross-validates principal, financed balance, insurance, EMI, rate, and tenure.
    - Computes APR only when required cash flows are fully verified.
    - Retains source page and evidence snippets on all findings.
    """

    @staticmethod
    def calculate_reducing_emi(principal: float, annual_rate_pct: float, tenure_months: int) -> float:
        """Standard reducing balance EMI formula."""
        if tenure_months <= 0 or principal <= 0:
            return 0.0
        if annual_rate_pct <= 0:
            return round(principal / tenure_months, 2)

        monthly_rate = (annual_rate_pct / 100.0) / 12.0
        try:
            factor = math.pow(1 + monthly_rate, tenure_months)
            if factor == 1.0:
                return round(principal / tenure_months, 2)
            emi = principal * monthly_rate * factor / (factor - 1)
            return round(emi, 2)
        except (OverflowError, ZeroDivisionError):
            return 0.0

    @staticmethod
    def calculate_flat_emi(principal: float, annual_rate_pct: float, tenure_months: int) -> float:
        """Flat rate EMI: Interest is calculated on full principal for entire tenure."""
        if tenure_months <= 0 or principal <= 0:
            return 0.0
        total_interest = principal * (annual_rate_pct / 100.0) * (tenure_months / 12.0)
        emi = (principal + total_interest) / tenure_months
        return round(emi, 2)

    @staticmethod
    def calculate_irr_newton_raphson(
        cash_flows: List[float],
        guess: float = 0.01,
        tol: float = 1e-7,
        max_iter: int = 1000,
    ) -> Optional[float]:
        """
        Calculates exact monthly Internal Rate of Return (IRR) using Newton-Raphson.
        cash_flows[0] = net disbursal (positive cash inflow to borrower)
        cash_flows[1..n] = monthly EMI payments (negative cash outflows)

        Returns None if cash flows cannot yield a valid mathematical IRR (e.g. net disbursal <= 0).
        """
        if not cash_flows or len(cash_flows) < 2:
            return None
        if cash_flows[0] <= 0:
            return None
        if all(cf >= 0 for cf in cash_flows[1:]):
            return None
        if all(cf <= 0 for cf in cash_flows):
            return None

        rate = guess
        for _ in range(max_iter):
            try:
                npv = 0.0
                derivative_npv = 0.0
                for t, cf in enumerate(cash_flows):
                    if 1.0 + rate <= 0:
                        raise ValueError("Singularity: 1 + rate <= 0")
                    denom = math.pow(1.0 + rate, t)
                    if denom == 0 or math.isinf(denom):
                        raise OverflowError("Denominator overflow")
                    npv += cf / denom
                    if t > 0:
                        derivative_npv -= (t * cf) / (denom * (1.0 + rate))

                if abs(npv) < tol:
                    return rate

                if derivative_npv == 0.0 or math.isinf(derivative_npv) or math.isnan(derivative_npv):
                    break

                step = npv / derivative_npv
                new_rate = rate - step
                if new_rate <= -0.999:
                    new_rate = -0.5
                elif new_rate > 10.0:
                    new_rate = 5.0

                if abs(new_rate - rate) < tol:
                    return new_rate
                rate = new_rate
            except (OverflowError, ZeroDivisionError, ValueError):
                break

        # Bisection fallback
        low = -0.5
        high = 5.0

        def eval_npv(r: float) -> float:
            try:
                if 1.0 + r <= 0:
                    return float("inf")
                total = 0.0
                for t, cf in enumerate(cash_flows):
                    d = math.pow(1.0 + r, t)
                    total += cf / d
                return total
            except (OverflowError, ZeroDivisionError):
                return float("inf")

        npv_low = eval_npv(low)
        npv_high = eval_npv(high)

        if math.isinf(npv_low) or math.isinf(npv_high) or (npv_low * npv_high > 0):
            high = 15.0
            npv_high = eval_npv(high)
            if math.isinf(npv_high) or (npv_low * npv_high > 0):
                return None

        for _ in range(200):
            mid = (low + high) / 2.0
            npv_mid = eval_npv(mid)
            if abs(npv_mid) < tol:
                return mid
            if npv_low * npv_mid < 0:
                high = mid
                npv_high = npv_mid
            else:
                low = mid
                npv_low = npv_mid

        return (low + high) / 2.0

    @classmethod
    def _build_field_statuses(cls, loan: ExtractedLoanTerms) -> Dict[str, ExtractedFieldInfo]:
        """Populates explicit status metadata for every tracked field without false defaults."""
        statuses: Dict[str, ExtractedFieldInfo] = dict(loan.field_metadata or {})
        tracked = {
            "principal_amount": loan.principal_amount,
            "opening_financed_balance": loan.opening_financed_balance,
            "stated_interest_rate": loan.stated_interest_rate,
            "tenure_months": loan.tenure_months,
            "stated_emi": loan.stated_emi,
            "bundled_insurance": loan.bundled_insurance,
            "gst_amount": loan.stated_gst_amount,
            "net_disbursement": loan.stated_net_disbursement,
        }
        for key, val in tracked.items():
            if key not in statuses:
                statuses[key] = ExtractedFieldInfo(
                    status=FieldStatus.USER_VERIFIED if val is not None else FieldStatus.NOT_FOUND
                )
            elif val is None and statuses[key].status == FieldStatus.CONFIRMED:
                statuses[key].status = FieldStatus.NOT_FOUND

        if "interest_type" not in statuses:
            statuses["interest_type"] = ExtractedFieldInfo(
                status=FieldStatus.CONFIRMED if loan.interest_type != InterestRateType.UNKNOWN else FieldStatus.NOT_FOUND
            )
        if "prepayment_penalty" not in statuses:
            statuses["prepayment_penalty"] = ExtractedFieldInfo(
                status=FieldStatus.CONFIRMED if loan.has_prepayment_penalty is not None else FieldStatus.NOT_FOUND
            )
        return statuses

    @classmethod
    def _normalize_charges(cls, loan: ExtractedLoanTerms) -> List[ExtractedCharge]:
        """Ensures charges list exists cleanly whether sourced from multi-page extractor or manual fields."""
        if loan.charges:
            return list(loan.charges)

        charges: List[ExtractedCharge] = []
        if loan.processing_fee is not None and loan.processing_fee > 0:
            charges.append(
                ExtractedCharge(
                    name="Processing Fee",
                    amount=round(loan.processing_fee, 2),
                    category=ChargeCategory.CONFIRMED_UPFRONT,
                    status=FieldStatus.USER_VERIFIED,
                    tax_note=loan.gst_status_note,
                )
            )
        if loan.documentation_charges is not None and loan.documentation_charges > 0:
            charges.append(
                ExtractedCharge(
                    name="Documentation Charges",
                    amount=round(loan.documentation_charges, 2),
                    category=ChargeCategory.CONFIRMED_UPFRONT,
                    status=FieldStatus.USER_VERIFIED,
                    tax_note=loan.gst_status_note,
                )
            )
        if loan.other_cash_recoveries is not None and loan.other_cash_recoveries > 0:
            charges.append(
                ExtractedCharge(
                    name="Other Cash Recoveries",
                    amount=round(loan.other_cash_recoveries, 2),
                    category=ChargeCategory.CONFIRMED_UPFRONT,
                    status=FieldStatus.USER_VERIFIED,
                )
            )
        if loan.bundled_insurance is not None and loan.bundled_insurance > 0:
            cat = ChargeCategory.FINANCED if loan.is_insurance_capitalized else ChargeCategory.CONFIRMED_UPFRONT
            charges.append(
                ExtractedCharge(
                    name="Insurance Premium",
                    amount=round(loan.bundled_insurance, 2),
                    category=cat,
                    status=FieldStatus.USER_VERIFIED,
                )
            )
        return charges

    @classmethod
    def _check_impossible_combinations(
        cls,
        loan: ExtractedLoanTerms,
        field_statuses: Dict[str, ExtractedFieldInfo],
    ) -> List[DiscrepancyItem]:
        """Rejects mathematically impossible combinations unless explicitly supported by evidence."""
        rejections: List[DiscrepancyItem] = []
        principal = loan.principal_amount
        tenure = loan.tenure_months
        rate = loan.stated_interest_rate
        emi = loan.stated_emi

        if principal is not None and principal > 0:
            if tenure is not None and tenure == 0:
                ten_ev = (field_statuses.get("tenure_months") or ExtractedFieldInfo()).evidence_snippet or ""
                if "bullet" not in ten_ev.lower() and "demand" not in ten_ev.lower():
                    field_statuses["tenure_months"] = ExtractedFieldInfo(
                        status=FieldStatus.UNCERTAIN,
                        source_page=field_statuses.get("tenure_months", ExtractedFieldInfo()).source_page,
                        evidence_snippet=ten_ev or None,
                        note="Rejected: positive principal with zero tenure",
                    )
                    rejections.append(
                        DiscrepancyItem(
                            code="IMPOSSIBLE_ZERO_TENURE",
                            severity=DiscrepancySeverity.CRITICAL,
                            title="Impossible Combination: Positive Principal with Zero Tenure",
                            description=(
                                f"Extracted principal is ₹{principal:,.2f} while tenure was recorded as 0 months "
                                "without supporting evidence for a zero-tenure facility."
                            ),
                            source_page=field_statuses.get("tenure_months", ExtractedFieldInfo()).source_page,
                            evidence_snippet=ten_ev or None,
                        )
                    )

            if emi is not None and emi == 0:
                emi_ev = (field_statuses.get("stated_emi") or ExtractedFieldInfo()).evidence_snippet or ""
                if "moratorium" not in emi_ev.lower() and "nil emi" not in emi_ev.lower():
                    field_statuses["stated_emi"] = ExtractedFieldInfo(
                        status=FieldStatus.UNCERTAIN,
                        source_page=field_statuses.get("stated_emi", ExtractedFieldInfo()).source_page,
                        evidence_snippet=emi_ev or None,
                        note="Rejected: positive principal with zero EMI",
                    )
                    rejections.append(
                        DiscrepancyItem(
                            code="IMPOSSIBLE_ZERO_EMI",
                            severity=DiscrepancySeverity.CRITICAL,
                            title="Impossible Combination: Positive Principal with Zero EMI",
                            description=(
                                f"Extracted principal is ₹{principal:,.2f} with a stated EMI of ₹0 without moratorium evidence."
                            ),
                            source_page=field_statuses.get("stated_emi", ExtractedFieldInfo()).source_page,
                            evidence_snippet=emi_ev or None,
                        )
                    )

            if rate is not None and rate == 0:
                rate_ev = (field_statuses.get("stated_interest_rate") or ExtractedFieldInfo()).evidence_snippet or ""
                if not any(tok in rate_ev.lower() for tok in ("0%", "0.0%", "zero interest", "no cost", "interest-free")):
                    field_statuses["stated_interest_rate"] = ExtractedFieldInfo(
                        status=FieldStatus.UNCERTAIN,
                        source_page=field_statuses.get("stated_interest_rate", ExtractedFieldInfo()).source_page,
                        evidence_snippet=rate_ev or None,
                        note="Rejected: zero interest rate lacks explicit 0% evidence",
                    )
                    rejections.append(
                        DiscrepancyItem(
                            code="IMPOSSIBLE_UNSUPPORTED_ZERO_RATE",
                            severity=DiscrepancySeverity.CRITICAL,
                            title="Unverified Zero Interest Rate on Positive Principal",
                            description=(
                                f"Principal is ₹{principal:,.2f} with extracted rate 0%, which is not supported by an explicit zero-interest clause."
                            ),
                            source_page=field_statuses.get("stated_interest_rate", ExtractedFieldInfo()).source_page,
                            evidence_snippet=rate_ev or None,
                        )
                    )

            if emi is not None and tenure is not None and tenure > 1 and emi > principal:
                rejections.append(
                    DiscrepancyItem(
                        code="IMPOSSIBLE_EMI_EXCEEDS_PRINCIPAL",
                        severity=DiscrepancySeverity.CRITICAL,
                        title="Impossible Combination: Single Monthly EMI Exceeds Full Principal",
                        description=(
                            f"Stated monthly EMI (₹{emi:,.2f}) exceeds entire principal (₹{principal:,.2f}) over a {tenure}-month tenure."
                        ),
                        source_page=field_statuses.get("stated_emi", ExtractedFieldInfo()).source_page,
                        evidence_snippet=field_statuses.get("stated_emi", ExtractedFieldInfo()).evidence_snippet,
                    )
                )

        return rejections

    @classmethod
    def audit_loan(cls, loan: ExtractedLoanTerms) -> LoanAuditReport:
        field_statuses = cls._build_field_statuses(loan)
        charges = cls._normalize_charges(loan)
        discrepancies: List[DiscrepancyItem] = []
        cross_validations: List[CrossValidationResult] = []

        principal = loan.principal_amount
        opening_bal = loan.opening_financed_balance
        tenure = loan.tenure_months
        stated_rate = loan.stated_interest_rate
        stated_emi = loan.stated_emi

        # 1. Separate charges strictly by category
        upfront_items = [c for c in charges if c.category == ChargeCategory.CONFIRMED_UPFRONT and c.amount is not None]
        financed_items = [c for c in charges if c.category == ChargeCategory.FINANCED and c.amount is not None]
        conditional_items = [c for c in charges if c.category == ChargeCategory.CONDITIONAL and c.amount is not None]
        third_party_items = [c for c in charges if c.category == ChargeCategory.THIRD_PARTY and c.amount is not None]

        # Separate non-insurance upfront cash recoveries vs upfront insurance
        upfront_non_ins = [c for c in upfront_items if "insurance" not in c.name.lower()]
        upfront_ins = [c for c in upfront_items if "insurance" in c.name.lower()]

        raw_pf: Optional[float] = None
        raw_doc: Optional[float] = None
        other_rec: Optional[float] = None
        for c in upfront_non_ins:
            lname = c.name.lower()
            if "processing" in lname:
                raw_pf = round((raw_pf or 0.0) + c.amount, 2)
            elif any(k in lname for k in ("doc", "stamp", "legal", "cersai", "title")):
                raw_doc = round((raw_doc or 0.0) + c.amount, 2)
            else:
                other_rec = round((other_rec or 0.0) + c.amount, 2)

        # Only backfill from scalar fields when loan.charges was not populated (e.g. manual input)
        if not loan.charges:
            if raw_pf is None and loan.processing_fee is not None:
                raw_pf = round(loan.processing_fee, 2)
            if raw_doc is None and loan.documentation_charges is not None:
                raw_doc = round(loan.documentation_charges, 2)
            if other_rec is None and loan.other_cash_recoveries is not None:
                other_rec = round(loan.other_cash_recoveries, 2)

        known_upfront_excl_tax: Optional[float] = (
            round(sum(c.amount for c in upfront_non_ins), 2)
            if upfront_non_ins
            else (
                round((raw_pf or 0.0) + (raw_doc or 0.0) + (other_rec or 0.0), 2)
                if any(v is not None for v in (raw_pf, raw_doc, other_rec))
                else None
            )
        )

        financed_total: Optional[float] = round(sum(c.amount for c in financed_items), 2) if financed_items else None
        conditional_total: Optional[float] = round(sum(c.amount for c in conditional_items), 2) if conditional_items else None
        third_party_total: Optional[float] = round(sum(c.amount for c in third_party_items), 2) if third_party_items else None

        # Bundled insurance total and capitalization check
        bundled_ins: Optional[float] = loan.bundled_insurance
        is_ins_cap = loan.is_insurance_capitalized
        if financed_total is not None and financed_total > 0:
            bundled_ins = financed_total
            is_ins_cap = True
        elif upfront_ins:
            bundled_ins = round(sum(c.amount for c in upfront_ins), 2)
            is_ins_cap = False

        # Infer insurance capitalization deterministically if Principal + Insurance == Opening Financed Balance
        if (
            not is_ins_cap
            and principal is not None
            and opening_bal is not None
            and bundled_ins is not None
            and bundled_ins > 0
            and abs((principal + bundled_ins) - opening_bal) <= 5.0
        ):
            is_ins_cap = True
            financed_total = bundled_ins

        # 2. Check Impossible Combinations BEFORE running downstream math
        impossible_rejections = cls._check_impossible_combinations(loan, field_statuses)
        if impossible_rejections:
            discrepancies.extend(impossible_rejections)
            cross_validations.append(
                CrossValidationResult(
                    check_name="Impossible Combination Guard",
                    status=CrossValidationStatus.FAIL,
                    summary="; ".join(d.title for d in impossible_rejections),
                )
            )
            return LoanAuditReport(
                lender_name=loan.lender_name,
                principal=principal,
                opening_financed_balance=opening_bal,
                tenure_months=tenure,
                stated_annual_rate=stated_rate,
                interest_type=loan.interest_type,
                stated_emi=stated_emi,
                raw_processing_fee=raw_pf,
                raw_documentation_charges=raw_doc,
                other_cash_recoveries=other_rec,
                known_upfront_recoveries_excl_tax=known_upfront_excl_tax,
                fee_gst_amount=loan.stated_gst_amount,
                stated_gst_rate_pct=loan.stated_gst_rate_pct,
                gst_status_note=loan.gst_status_note,
                bundled_insurance_amount=bundled_ins,
                is_insurance_capitalized=is_ins_cap,
                conditional_charges_total=conditional_total,
                third_party_charges_total=third_party_total,
                gross_loan_amount=opening_bal or principal,
                net_disbursed_amount=loan.stated_net_disbursement,
                is_irr_computable=False,
                audit_status="REJECTED_INVALID_COMBINATION",
                is_verified=False,
                missing_fields=loan.missing_fields,
                field_statuses=field_statuses,
                charges_breakdown=charges,
                cross_validations=cross_validations,
                negotiation_script_blocked=True,
                negotiation_block_reason=impossible_rejections[0].title,
                discrepancies=discrepancies,
                rbi_compliance_score=None,
                extraction_method=loan.extraction_method,
            )

        # 3. GST Calculation (NEVER invent 18% or SAC 9971)
        has_unquantified_tax_note = bool(
            loan.gst_status_note
            or any(c.tax_note and "plus" in c.tax_note.lower() for c in upfront_non_ins)
        )
        fee_gst_amount: Optional[float] = None
        if loan.stated_gst_amount is not None:
            fee_gst_amount = round(loan.stated_gst_amount, 2)
        elif loan.stated_gst_rate_pct is not None and known_upfront_excl_tax is not None:
            taxable_sum = sum(
                c.amount for c in upfront_non_ins
                if not (c.tax_note and "inclusive" in c.tax_note.lower())
            )
            if not upfront_non_ins:
                taxable_sum = known_upfront_excl_tax
            fee_gst_amount = round(taxable_sum * (loan.stated_gst_rate_pct / 100.0), 2)
        elif known_upfront_excl_tax is not None and known_upfront_excl_tax == 0.0 and not has_unquantified_tax_note:
            fee_gst_amount = 0.0

        # If fee_gst_amount is explicitly confirmed, update field status
        if fee_gst_amount is not None and field_statuses["gst_amount"].status == FieldStatus.NOT_FOUND:
            field_statuses["gst_amount"].status = (
                FieldStatus.CONFIRMED if loan.stated_gst_rate_pct is not None else FieldStatus.USER_VERIFIED
            )

        total_upfront_incl_gst: Optional[float] = (
            round(known_upfront_excl_tax + fee_gst_amount, 2)
            if (known_upfront_excl_tax is not None and fee_gst_amount is not None)
            else None
        )

        # 4. Financed Principal / Gross Loan Balance
        gross_loan: Optional[float] = None
        if opening_bal is not None:
            gross_loan = round(opening_bal, 2)
        elif principal is not None:
            gross_loan = round(principal + (financed_total or (bundled_ins if is_ins_cap and bundled_ins else 0.0)), 2)

        # 5. Net Disbursed Amount (Never claim 0 without evidence; UNKNOWN if fees/GST missing)
        upfront_insurance_deducted = 0.0 if is_ins_cap else (bundled_ins or 0.0)
        net_disbursed: Optional[float] = None
        if loan.stated_net_disbursement is not None:
            net_disbursed = round(loan.stated_net_disbursement, 2)
        elif principal is not None:
            if known_upfront_excl_tax is None:
                # Only compute net_disbursed when all core fields are verified (not missing) and fees were explicitly 0/user-verified
                if (
                    loan.is_verified
                    and stated_rate is not None
                    and tenure is not None
                    and not has_unquantified_tax_note
                    and field_statuses["principal_amount"].status == FieldStatus.USER_VERIFIED
                ):
                    net_disbursed = round(principal - upfront_insurance_deducted, 2)
            elif total_upfront_incl_gst is not None:
                net_disbursed = round(principal - total_upfront_incl_gst - upfront_insurance_deducted, 2)

        if net_disbursed is not None and field_statuses["net_disbursement"].status == FieldStatus.NOT_FOUND:
            field_statuses["net_disbursement"].status = (
                FieldStatus.CONFIRMED if loan.stated_net_disbursement is not None or loan.stated_gst_rate_pct is not None
                else FieldStatus.USER_VERIFIED
            )

        # 6. Cross-Validation Suite
        # Check A: Principal vs Opening Financed Balance & Insurance vs Capitalized Principal
        if principal is not None and opening_bal is not None:
            expected_cap = financed_total if financed_total is not None else (bundled_ins if is_ins_cap and bundled_ins else 0.0)
            expected_bal = round(principal + expected_cap, 2)
            bal_diff = round(opening_bal - expected_bal, 2)
            if abs(bal_diff) <= 5.0:
                cross_validations.append(
                    CrossValidationResult(
                        check_name="Principal vs Opening Financed Balance",
                        status=CrossValidationStatus.PASS,
                        summary=(
                            f"Principal (₹{principal:,.2f}) + Financed Capital (₹{expected_cap:,.2f}) "
                            f"= ₹{expected_bal:,.2f}, matching Opening Financed Balance (₹{opening_bal:,.2f})."
                        ),
                        expected_value=f"₹{expected_bal:,.2f}",
                        actual_value=f"₹{opening_bal:,.2f}",
                        source_page=field_statuses["opening_financed_balance"].source_page,
                        evidence_snippet=field_statuses["opening_financed_balance"].evidence_snippet,
                    )
                )
                if expected_cap > 0:
                    cross_validations.append(
                        CrossValidationResult(
                            check_name="Insurance vs Capitalized Principal",
                            status=CrossValidationStatus.PASS,
                            summary=f"Financed insurance of ₹{expected_cap:,.2f} reconciles the gap between principal and opening balance.",
                            expected_value=f"₹{expected_cap:,.2f}",
                            actual_value=f"₹{opening_bal - principal:,.2f}",
                        )
                    )
            else:
                cross_validations.append(
                    CrossValidationResult(
                        check_name="Principal vs Opening Financed Balance",
                        status=CrossValidationStatus.FAIL,
                        summary=(
                            f"Opening Financed Balance (₹{opening_bal:,.2f}) differs from Principal + Financed Charges "
                            f"(₹{expected_bal:,.2f}) by ₹{bal_diff:,.2f}."
                        ),
                        expected_value=f"₹{expected_bal:,.2f}",
                        actual_value=f"₹{opening_bal:,.2f}",
                        source_page=field_statuses["opening_financed_balance"].source_page,
                        evidence_snippet=field_statuses["opening_financed_balance"].evidence_snippet,
                    )
                )
                discrepancies.append(
                    DiscrepancyItem(
                        code="FINANCED_BALANCE_MISMATCH",
                        severity=DiscrepancySeverity.CRITICAL,
                        title="Unexplained Opening Financed Balance Discrepancy",
                        description=(
                            f"Opening financed balance (₹{opening_bal:,.2f}) does not reconcile with Principal "
                            f"(₹{principal:,.2f}) + financed insurance/charges (₹{expected_cap:,.2f}); gap is ₹{bal_diff:,.2f}."
                        ),
                        impact_amount=bal_diff,
                        source_page=field_statuses["opening_financed_balance"].source_page,
                        evidence_snippet=field_statuses["opening_financed_balance"].evidence_snippet,
                    )
                )
        elif principal is not None and financed_total is not None and financed_total > 0:
            cross_validations.append(
                CrossValidationResult(
                    check_name="Insurance vs Capitalized Principal",
                    status=CrossValidationStatus.PASS,
                    summary=f"Financed insurance ₹{financed_total:,.2f} capitalized onto principal ₹{principal:,.2f} (Gross: ₹{gross_loan:,.2f}).",
                    expected_value=f"₹{gross_loan:,.2f}",
                    actual_value=f"₹{gross_loan:,.2f}",
                )
            )
        else:
            cross_validations.append(
                CrossValidationResult(
                    check_name="Principal vs Opening Financed Balance",
                    status=CrossValidationStatus.SKIPPED_MISSING_DATA,
                    summary="Opening financed balance or principal not available for reconciliation.",
                )
            )

        # 7. Independent EMI Calculation & Cross-Validation
        reducing_emi: Optional[float] = None
        flat_emi: Optional[float] = None
        emi_diff: Optional[float] = None
        is_emi_matching: Optional[bool] = None
        emi_to_use: Optional[float] = stated_emi

        is_flat_structure = (loan.interest_type == InterestRateType.FLAT)

        if gross_loan is not None and gross_loan > 0 and stated_rate is not None and tenure is not None and tenure > 0:
            reducing_emi = cls.calculate_reducing_emi(gross_loan, stated_rate, tenure)
            flat_emi = cls.calculate_flat_emi(gross_loan, stated_rate, tenure)
            benchmark_emi = flat_emi if is_flat_structure else reducing_emi
            if emi_to_use is None or emi_to_use <= 0:
                emi_to_use = benchmark_emi

            if stated_emi is not None and stated_emi > 0:
                emi_diff = round(stated_emi - benchmark_emi, 2)
                is_emi_matching = abs(emi_diff) <= 5.0
                if is_emi_matching:
                    cross_validations.append(
                        CrossValidationResult(
                            check_name="Stated EMI vs Independently Calculated EMI",
                            status=CrossValidationStatus.PASS,
                            summary=f"Stated EMI ₹{stated_emi:,.2f} matches calculated benchmark EMI ₹{benchmark_emi:,.2f}.",
                            expected_value=f"₹{benchmark_emi:,.2f}",
                            actual_value=f"₹{stated_emi:,.2f}",
                            source_page=field_statuses["stated_emi"].source_page,
                            evidence_snippet=field_statuses["stated_emi"].evidence_snippet,
                        )
                    )
                else:
                    cross_validations.append(
                        CrossValidationResult(
                            check_name="Stated EMI vs Independently Calculated EMI",
                            status=CrossValidationStatus.WARNING,
                            summary=(
                                f"Stated EMI ₹{stated_emi:,.2f} differs from calculated benchmark EMI ₹{benchmark_emi:,.2f} "
                                f"by ₹{emi_diff:,.2f}/month."
                            ),
                            expected_value=f"₹{benchmark_emi:,.2f}",
                            actual_value=f"₹{stated_emi:,.2f}",
                            source_page=field_statuses["stated_emi"].source_page,
                            evidence_snippet=field_statuses["stated_emi"].evidence_snippet,
                        )
                    )
                    if not is_flat_structure and abs(stated_emi - flat_emi) <= 5.0 and stated_rate > 0:
                        discrepancies.append(
                            DiscrepancyItem(
                                code="DISGUISED_FLAT_RATE",
                                severity=DiscrepancySeverity.CRITICAL,
                                title="Flat-Rate EMI Structure Applied on Non-Flat Loan",
                                description=(
                                    f"Sanctioned EMI of ₹{stated_emi:,.2f} matches flat-rate calculation (₹{flat_emi:,.2f}) "
                                    f"instead of reducing balance EMI (₹{reducing_emi:,.2f})."
                                ),
                                impact_amount=round((stated_emi - reducing_emi) * tenure, 2),
                                source_page=field_statuses["stated_emi"].source_page,
                                evidence_snippet=field_statuses["stated_emi"].evidence_snippet,
                            )
                        )
                    elif emi_diff > 5.0:
                        discrepancies.append(
                            DiscrepancyItem(
                                code="STATED_EMI_EXCEEDS_AMORTIZATION",
                                severity=DiscrepancySeverity.WARNING,
                                title="Stated EMI Exceeds Standard Amortization",
                                description=(
                                    f"Sanctioned EMI of ₹{stated_emi:,.2f} exceeds calculated EMI of ₹{benchmark_emi:,.2f} "
                                    f"by ₹{emi_diff:,.2f}/month (₹{emi_diff * tenure:,.2f} over {tenure} months)."
                                ),
                                impact_amount=round(emi_diff * tenure, 2),
                                source_page=field_statuses["stated_emi"].source_page,
                                evidence_snippet=field_statuses["stated_emi"].evidence_snippet,
                            )
                        )
                    else:
                        discrepancies.append(
                            DiscrepancyItem(
                                code="STATED_EMI_BELOW_AMORTIZATION",
                                severity=DiscrepancySeverity.WARNING,
                                title="Stated EMI Below Standard Amortization",
                                description=(
                                    f"Sanctioned EMI of ₹{stated_emi:,.2f} is lower than calculated EMI of ₹{benchmark_emi:,.2f} "
                                    f"by ₹{abs(emi_diff):,.2f}/month, indicating potential teaser rate or balloon residual."
                                ),
                                impact_amount=round(abs(emi_diff) * tenure, 2),
                                source_page=field_statuses["stated_emi"].source_page,
                                evidence_snippet=field_statuses["stated_emi"].evidence_snippet,
                            )
                        )
            else:
                cross_validations.append(
                    CrossValidationResult(
                        check_name="Stated EMI vs Independently Calculated EMI",
                        status=CrossValidationStatus.SKIPPED_MISSING_DATA,
                        summary=f"Stated EMI not found in document; computed reducing benchmark is ₹{reducing_emi:,.2f}.",
                        expected_value=f"₹{reducing_emi:,.2f}",
                    )
                )
        else:
            cross_validations.append(
                CrossValidationResult(
                    check_name="Stated EMI vs Independently Calculated EMI",
                    status=CrossValidationStatus.SKIPPED_MISSING_DATA,
                    summary="Cannot compute independent EMI because principal, rate, or tenure is missing.",
                )
            )

        # Check C: Interest Rate vs Repayment Schedule
        if loan.schedule_emis and gross_loan is not None and gross_loan > 0 and stated_rate is not None:
            if tenure is not None and len(loan.schedule_emis) == tenure:
                sched_irr = cls.calculate_irr_newton_raphson([gross_loan] + [-e for e in loan.schedule_emis])
                if sched_irr is not None:
                    implied_ann = round(sched_irr * 12.0 * 100.0, 2)
                    rate_gap = round(abs(implied_ann - stated_rate), 2)
                    if rate_gap <= 0.15:
                        cross_validations.append(
                            CrossValidationResult(
                                check_name="Interest Rate vs Repayment Schedule",
                                status=CrossValidationStatus.PASS,
                                summary=f"Schedule-implied rate ({implied_ann:.2f}%) matches stated rate ({stated_rate:.2f}%).",
                                expected_value=f"{stated_rate:.2f}%",
                                actual_value=f"{implied_ann:.2f}%",
                            )
                        )
                    else:
                        cross_validations.append(
                            CrossValidationResult(
                                check_name="Interest Rate vs Repayment Schedule",
                                status=CrossValidationStatus.FAIL,
                                summary=f"Schedule-implied rate ({implied_ann:.2f}%) diverges from stated rate ({stated_rate:.2f}%).",
                                expected_value=f"{stated_rate:.2f}%",
                                actual_value=f"{implied_ann:.2f}%",
                            )
                        )
                        discrepancies.append(
                            DiscrepancyItem(
                                code="SCHEDULE_RATE_MISMATCH",
                                severity=DiscrepancySeverity.CRITICAL,
                                title="Repayment Schedule Rate Mismatch",
                                description=f"Back-solved schedule rate is {implied_ann:.2f}% vs stated {stated_rate:.2f}%.",
                            )
                        )
            else:
                avg_sched_emi = sum(loan.schedule_emis) / len(loan.schedule_emis)
                if reducing_emi is not None and abs(avg_sched_emi - reducing_emi) <= 5.0:
                    cross_validations.append(
                        CrossValidationResult(
                            check_name="Interest Rate vs Repayment Schedule",
                            status=CrossValidationStatus.PASS,
                            summary=f"Extracted schedule instalments (avg ₹{avg_sched_emi:,.2f}) align with {stated_rate:.2f}% amortization.",
                        )
                    )
                else:
                    cross_validations.append(
                        CrossValidationResult(
                            check_name="Interest Rate vs Repayment Schedule",
                            status=CrossValidationStatus.WARNING,
                            summary="Partial schedule instalments differ from calculated reducing EMI.",
                        )
                    )
        else:
            cross_validations.append(
                CrossValidationResult(
                    check_name="Interest Rate vs Repayment Schedule",
                    status=CrossValidationStatus.SKIPPED_MISSING_DATA,
                    summary="No tabular repayment schedule stream present in document.",
                )
            )

        # Check D: Tenure vs Number of Instalments
        if tenure is not None and loan.schedule_instalment_count is not None:
            if tenure == loan.schedule_instalment_count:
                cross_validations.append(
                    CrossValidationResult(
                        check_name="Tenure vs Number of Instalments",
                        status=CrossValidationStatus.PASS,
                        summary=f"Stated tenure ({tenure} months) matches schedule count ({loan.schedule_instalment_count} instalments).",
                        expected_value=str(tenure),
                        actual_value=str(loan.schedule_instalment_count),
                    )
                )
            else:
                cross_validations.append(
                    CrossValidationResult(
                        check_name="Tenure vs Number of Instalments",
                        status=CrossValidationStatus.FAIL,
                        summary=f"Tenure ({tenure} months) conflicts with schedule count ({loan.schedule_instalment_count} instalments).",
                        expected_value=str(tenure),
                        actual_value=str(loan.schedule_instalment_count),
                    )
                )
                discrepancies.append(
                    DiscrepancyItem(
                        code="TENURE_INSTALMENT_COUNT_MISMATCH",
                        severity=DiscrepancySeverity.CRITICAL,
                        title="Tenure vs Instalment Count Conflict",
                        description=f"Tenure is {tenure} months but schedule shows {loan.schedule_instalment_count} instalments.",
                    )
                )
        else:
            cross_validations.append(
                CrossValidationResult(
                    check_name="Tenure vs Number of Instalments",
                    status=CrossValidationStatus.SKIPPED_MISSING_DATA,
                    summary="No separate schedule instalment count found to cross-check against tenure.",
                )
            )

        # 8. Cash Outflows & True APR / IRR (Gated STRICTLY on valid cash flows)
        total_repayment: Optional[float] = None
        total_interest_payable: Optional[float] = None
        total_cost_of_borrowing: Optional[float] = None

        if emi_to_use is not None and emi_to_use > 0 and tenure is not None and tenure > 0:
            total_repayment = round(emi_to_use * tenure, 2)
            if gross_loan is not None:
                total_interest_payable = round(total_repayment - gross_loan, 2)
            if principal is not None and total_upfront_incl_gst is not None:
                total_cost_of_borrowing = round(
                    (total_repayment + total_upfront_incl_gst + upfront_insurance_deducted) - principal, 2
                )

        is_irr_computable = False
        monthly_irr: Optional[float] = None
        nominal_apr: Optional[float] = None
        effective_annual_apr: Optional[float] = None
        rate_inflation_spread: Optional[float] = None

        if net_disbursed is not None and net_disbursed <= 0:
            discrepancies.append(
                DiscrepancyItem(
                    code="UPFRONT_DEDUCTIONS_EXCEED_PRINCIPAL",
                    severity=DiscrepancySeverity.CRITICAL,
                    title="Upfront Deductions Exceed or Equal Principal",
                    description=(
                        f"Confirmed upfront deductions equal or exceed principal (net disbursed ₹{net_disbursed:,.2f}); "
                        "effective APR is non-computable."
                    ),
                    impact_amount=total_upfront_incl_gst,
                )
            )
        elif (
            net_disbursed is not None
            and net_disbursed > 0
            and emi_to_use is not None
            and emi_to_use > 0
            and tenure is not None
            and tenure > 0
        ):
            cash_flows = [net_disbursed] + [-emi_to_use] * tenure
            irr_raw = cls.calculate_irr_newton_raphson(cash_flows)
            if irr_raw is not None and not math.isnan(irr_raw) and not math.isinf(irr_raw):
                is_irr_computable = True
                monthly_irr = round(irr_raw, 6)
                nominal_apr = round(irr_raw * 12.0 * 100.0, 2)
                effective_annual_apr = round((math.pow(1.0 + irr_raw, 12) - 1.0) * 100.0, 2)
                if stated_rate is not None:
                    rate_inflation_spread = round(effective_annual_apr - stated_rate, 2)

        # 9. Evidence-Backed Discrepancies & Findings
        missing_core = [
            f for f, v in (
                ("principal_amount", principal),
                ("stated_interest_rate", stated_rate),
                ("tenure_months", tenure),
            )
            if v is None
        ]
        if missing_core:
            discrepancies.append(
                DiscrepancyItem(
                    code="INCOMPLETE_LOAN_DATA",
                    severity=DiscrepancySeverity.CRITICAL,
                    title="Essential Loan Parameters Missing / Unverified",
                    description=(
                        f"Core parameters not confirmed in document: {', '.join(missing_core)}. "
                        "Values remain UNKNOWN rather than assumed."
                    ),
                )
            )

        if is_flat_structure and stated_rate is not None:
            apr_note = f" (Effective Reducing APR: {effective_annual_apr:.2f}%)" if effective_annual_apr else ""
            discrepancies.append(
                DiscrepancyItem(
                    code="FLAT_RATE_STRUCTURE",
                    severity=DiscrepancySeverity.CRITICAL,
                    title="Flat Interest Rate Structure Detected",
                    description=(
                        f"Stated rate ({stated_rate}% flat){apr_note} applies interest to the full original principal "
                        "throughout the entire tenure."
                    ),
                    impact_amount=round((flat_emi - reducing_emi) * tenure, 2) if (flat_emi and reducing_emi and tenure) else None,
                    source_page=field_statuses["interest_type"].source_page,
                    evidence_snippet=field_statuses["interest_type"].evidence_snippet,
                )
            )

        # Capitalized / Financed insurance finding with provenance
        for c in financed_items:
            discrepancies.append(
                DiscrepancyItem(
                    code="CAPITALIZED_INSURANCE",
                    severity=DiscrepancySeverity.WARNING if (principal and c.amount < 0.03 * principal) else DiscrepancySeverity.CRITICAL,
                    title=f"{c.name} Capitalized into Loan Balance",
                    description=(
                        f"₹{c.amount:,.2f} is capitalized into the loan balance, increasing financed debt "
                        "and total interest payable over the tenure."
                    ),
                    impact_amount=c.amount,
                    source_page=c.source_page,
                    evidence_snippet=c.evidence_snippet,
                )
            )

        # Upfront insurance finding
        for c in upfront_ins:
            discrepancies.append(
                DiscrepancyItem(
                    code="BUNDLED_INSURANCE",
                    severity=DiscrepancySeverity.WARNING if (principal and c.amount < 0.03 * principal) else DiscrepancySeverity.CRITICAL,
                    title=f"Upfront {c.name} Deduction Detected",
                    description=f"₹{c.amount:,.2f} is deducted upfront from disbursement for insurance cover.",
                    impact_amount=c.amount,
                    source_page=c.source_page,
                    evidence_snippet=c.evidence_snippet,
                )
            )

        # Conditional charges separated clearly
        for c in conditional_items:
            discrepancies.append(
                DiscrepancyItem(
                    code="CONDITIONAL_CHARGE_DETECTED",
                    severity=DiscrepancySeverity.INFO,
                    title=f"Conditional Charge Excluded from Upfront Cash Deductions: {c.name}",
                    description=(
                        f"Conditional fee of ₹{c.amount:,.2f} detected and tracked separately from confirmed upfront recoveries."
                    ),
                    impact_amount=c.amount,
                    source_page=c.source_page,
                    evidence_snippet=c.evidence_snippet,
                )
            )

        # Unquantified GST / Unknown Net Disbursement finding
        if fee_gst_amount is None and (known_upfront_excl_tax or has_unquantified_tax_note):
            discrepancies.append(
                DiscrepancyItem(
                    code="UNKNOWN_TAX_AND_DISBURSEMENT",
                    severity=DiscrepancySeverity.WARNING,
                    title="Applicable GST & Final Net Disbursement Not Disclosed",
                    description=(
                        f"Confirmed upfront cash recoveries before tax total ₹{(known_upfront_excl_tax or 0.0):,.2f}, "
                        "but exact GST amount/rate and final net disbursement are not disclosed in the document. "
                        "Both GST and net disbursement remain UNKNOWN (APR withheld)."
                    ),
                    source_page=field_statuses["gst_amount"].source_page,
                    evidence_snippet=field_statuses["gst_amount"].evidence_snippet or loan.gst_status_note,
                )
            )

        # APR spread alert (only if IRR is computable)
        if rate_inflation_spread is not None and rate_inflation_spread >= 2.0:
            discrepancies.append(
                DiscrepancyItem(
                    code="HIGH_APR_SPREAD",
                    severity=DiscrepancySeverity.WARNING if rate_inflation_spread < 5.0 else DiscrepancySeverity.CRITICAL,
                    title="Significant True Cost (APR) Inflation",
                    description=(
                        f"Effective Annual APR is {effective_annual_apr:.2f}%, exceeding stated rate ({stated_rate:.2f}%) "
                        f"by {rate_inflation_spread:.2f}%."
                    ),
                    impact_amount=round(total_cost_of_borrowing - total_interest_payable, 2)
                    if (total_cost_of_borrowing is not None and total_interest_payable is not None)
                    else None,
                )
            )

        # Prepayment penalty finding (without inventing unstated circular citations)
        if loan.has_prepayment_penalty is True:
            pen_meta = field_statuses.get("prepayment_penalty", ExtractedFieldInfo())
            if loan.interest_type == InterestRateType.FLOATING:
                discrepancies.append(
                    DiscrepancyItem(
                        code="FLOATING_PREPAYMENT_PENALTY_FLAG",
                        severity=DiscrepancySeverity.CRITICAL,
                        title="Prepayment / Foreclosure Penalty on Floating-Rate Loan",
                        description=(
                            "Document includes a prepayment/foreclosure penalty clause on a floating-rate facility. "
                            "Verify exemption eligibility for individual non-business borrowers under applicable RBI norms."
                        ),
                        source_page=pen_meta.source_page,
                        evidence_snippet=pen_meta.evidence_snippet or loan.prepayment_penalty_terms,
                    )
                )
            elif loan.interest_type in (InterestRateType.UNKNOWN, InterestRateType.REDUCING):
                discrepancies.append(
                    DiscrepancyItem(
                        code="PREPAYMENT_PENALTY_VERIFICATION_REQUIRED",
                        severity=DiscrepancySeverity.WARNING,
                        title="Prepayment Penalty Detected — Rate Type Requires Confirmation",
                        description=(
                            "A prepayment penalty clause is present while the fixed vs floating nature is unconfirmed."
                        ),
                        source_page=pen_meta.source_page,
                        evidence_snippet=pen_meta.evidence_snippet or loan.prepayment_penalty_terms,
                    )
                )
            elif loan.interest_type == InterestRateType.FIXED:
                discrepancies.append(
                    DiscrepancyItem(
                        code="FIXED_RATE_PREPAYMENT_DISCLOSURE",
                        severity=DiscrepancySeverity.INFO,
                        title="Fixed-Rate Prepayment Charges Disclosed",
                        description="Prepayment/foreclosure charges noted on a fixed-rate loan.",
                        source_page=pen_meta.source_page,
                        evidence_snippet=pen_meta.evidence_snippet or loan.prepayment_penalty_terms,
                    )
                )

        # 10. Determine Audit Status & Negotiation Script Gating (Rule 12)
        has_uncertain_core = any(
            field_statuses[k].status in (FieldStatus.NOT_FOUND, FieldStatus.UNCERTAIN)
            for k in ("principal_amount", "stated_interest_rate", "tenure_months")
        )
        has_failed_cv = any(cv.status == CrossValidationStatus.FAIL for cv in cross_validations)

        unverified_block_items = []
        if missing_core or has_uncertain_core:
            unverified_block_items.extend(missing_core or ["core_loan_terms"])
        if net_disbursed is None:
            unverified_block_items.append("net_disbursement")
        if fee_gst_amount is None and (known_upfront_excl_tax or has_unquantified_tax_note):
            unverified_block_items.append("gst_amount")
        if has_failed_cv:
            unverified_block_items.append("cross_validation_conflict")

        negotiation_blocked = len(unverified_block_items) > 0
        negotiation_reason = (
            f"Negotiation script generation blocked due to unverified/unknown fields: {', '.join(dict.fromkeys(unverified_block_items))}."
            if negotiation_blocked
            else None
        )

        audit_status = "VERIFIED"
        if has_failed_cv:
            audit_status = "PARTIAL_CROSS_VALIDATION_CONFLICT"
        elif missing_core or has_uncertain_core or net_disbursed is None:
            audit_status = "PARTIAL_MISSING_DATA"

        return LoanAuditReport(
            lender_name=loan.lender_name,
            principal=principal,
            opening_financed_balance=opening_bal,
            tenure_months=tenure,
            stated_annual_rate=stated_rate,
            interest_type=loan.interest_type,
            stated_emi=stated_emi,
            calculated_reducing_emi=reducing_emi,
            calculated_flat_emi=flat_emi,
            emi_difference=emi_diff,
            is_emi_matching=is_emi_matching,
            raw_processing_fee=raw_pf,
            raw_documentation_charges=raw_doc,
            other_cash_recoveries=other_rec,
            known_upfront_recoveries_excl_tax=known_upfront_excl_tax,
            fee_gst_amount=fee_gst_amount,
            stated_gst_rate_pct=loan.stated_gst_rate_pct,
            gst_status_note=loan.gst_status_note,
            total_upfront_fees_incl_gst=total_upfront_incl_gst,
            bundled_insurance_amount=bundled_ins,
            is_insurance_capitalized=is_ins_cap,
            conditional_charges_total=conditional_total,
            third_party_charges_total=third_party_total,
            gross_loan_amount=gross_loan,
            net_disbursed_amount=net_disbursed,
            total_repayment_amount=total_repayment,
            total_interest_payable=total_interest_payable,
            total_cost_of_borrowing=total_cost_of_borrowing,
            is_irr_computable=is_irr_computable,
            true_monthly_irr=monthly_irr,
            nominal_apr=nominal_apr,
            effective_annual_apr=effective_annual_apr,
            rate_inflation_spread=rate_inflation_spread,
            audit_status=audit_status,
            is_verified=(not missing_core and not has_uncertain_core and loan.is_verified),
            missing_fields=list(dict.fromkeys(loan.missing_fields + missing_core)),
            field_statuses=field_statuses,
            charges_breakdown=charges,
            cross_validations=cross_validations,
            negotiation_script_blocked=negotiation_blocked,
            negotiation_block_reason=negotiation_reason,
            discrepancies=discrepancies,
            rbi_compliance_score=None,
            explanation=None,
            extraction_method=loan.extraction_method,
        )

    @classmethod
    def compare_loans(
        cls, loan_a: ExtractedLoanTerms, loan_b: ExtractedLoanTerms
    ) -> Tuple[LoanAuditReport, LoanAuditReport, int]:
        """
        Audits both loans deterministically and ranks them based on Total Cost of Borrowing
        (falling back to total_repayment_amount if cost is uncomputable due to unknown tax).
        """
        audit_a = cls.audit_loan(loan_a)
        audit_b = cls.audit_loan(loan_b)

        metric_a = audit_a.total_cost_of_borrowing if audit_a.total_cost_of_borrowing is not None else (audit_a.total_repayment_amount or float("inf"))
        metric_b = audit_b.total_cost_of_borrowing if audit_b.total_cost_of_borrowing is not None else (audit_b.total_repayment_amount or float("inf"))

        if metric_a <= metric_b:
            return audit_a, audit_b, 1
        return audit_a, audit_b, 2
