import math
from typing import List, Tuple, Optional
from app.schemas.loan import (
    ExtractedLoanTerms,
    LoanAuditReport,
    DiscrepancyItem,
    DiscrepancySeverity,
    InterestRateType,
)

# Standard Indian GST rate applicable on Banking & Financial Services (SAC 9971)
INDIAN_BANKING_GST_RATE = 0.18


class DeterministicLoanAuditor:
    """
    Pure mathematical and rule-based audit engine.
    Under no circumstances are any numbers rounded arbitrarily or calculated via LLM.
    Strictly zero hallucinations.
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
        # In a valid retail loan, borrower receives positive net disbursal upfront
        if cash_flows[0] <= 0:
            return None
        # All subsequent periodic payments must have negative cash flow (repayment)
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
                if new_rate <= -0.999:  # Avoid singularity near -100%
                    new_rate = -0.5
                elif new_rate > 10.0:   # Prevent runaway positive divergence
                    new_rate = 5.0

                if abs(new_rate - rate) < tol:
                    return new_rate
                rate = new_rate
            except (OverflowError, ZeroDivisionError, ValueError):
                break

        # Bisection fallback if Newton-Raphson oscillates or diverges
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

        # Check if root is bracketed
        if math.isinf(npv_low) or math.isinf(npv_high) or (npv_low * npv_high > 0):
            # Try expanded upper bound
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
    def audit_loan(cls, loan: ExtractedLoanTerms) -> LoanAuditReport:
        principal = loan.principal_amount
        tenure = loan.tenure_months
        stated_rate = loan.stated_interest_rate

        # Guard: Incomplete or missing core terms (Zero Hallucination guarantee)
        if principal <= 0 or tenure <= 0 or stated_rate <= 0 or not loan.is_verified:
            missing_items = list(loan.missing_fields)
            if principal <= 0 and "principal_amount" not in missing_items:
                missing_items.append("principal_amount")
            if tenure <= 0 and "tenure_months" not in missing_items:
                missing_items.append("tenure_months")
            if stated_rate <= 0 and "stated_interest_rate" not in missing_items:
                missing_items.append("stated_interest_rate")

            discrepancies = [
                DiscrepancyItem(
                    code="INCOMPLETE_LOAN_DATA",
                    severity=DiscrepancySeverity.CRITICAL,
                    title="Essential Loan Parameters Missing / Unverified",
                    description=(
                        f"The document analysis could not verify core parameters: {', '.join(missing_items)}. "
                        "To prevent mathematical hallucination, automatic amortization and APR calculation are paused. "
                        "Please verify the document or input the missing terms manually."
                    ),
                    impact_amount=None,
                )
            ]

            return LoanAuditReport(
                lender_name=loan.lender_name,
                principal=max(0.0, principal),
                tenure_months=max(0, tenure),
                stated_annual_rate=max(0.0, stated_rate),
                interest_type=loan.interest_type,
                stated_emi=loan.stated_emi,
                calculated_reducing_emi=0.0,
                calculated_flat_emi=0.0,
                emi_difference=0.0,
                is_emi_matching=False,
                raw_processing_fee=loan.processing_fee,
                raw_documentation_charges=loan.documentation_charges,
                fee_gst_amount=0.0,
                total_upfront_fees_incl_gst=0.0,
                bundled_insurance_amount=loan.bundled_insurance,
                gross_loan_amount=max(0.0, principal),
                net_disbursed_amount=0.0,
                total_repayment_amount=0.0,
                total_interest_payable=0.0,
                total_cost_of_borrowing=0.0,
                is_irr_computable=False,
                true_monthly_irr=None,
                nominal_apr=None,
                effective_annual_apr=None,
                rate_inflation_spread=None,
                is_verified=False,
                missing_fields=missing_items,
                discrepancies=discrepancies,
                rbi_compliance_score=50,
                explanation=None,
                extraction_method=loan.extraction_method,
            )

        # 1. Financed Principal & Capitalized Insurance
        # When insurance is capitalized, it is added to the principal to form the financed loan
        financed_principal = principal + (loan.bundled_insurance if loan.is_insurance_capitalized else 0.0)

        # 2. EMI Calculations based on Financed Principal
        reducing_emi = cls.calculate_reducing_emi(financed_principal, stated_rate, tenure)
        flat_emi = cls.calculate_flat_emi(financed_principal, stated_rate, tenure)

        # Determine benchmark EMI depending on interest type
        is_flat_structure = (loan.interest_type == InterestRateType.FLAT)
        benchmark_emi = flat_emi if is_flat_structure else reducing_emi

        stated_emi = loan.stated_emi
        emi_to_use = stated_emi if (stated_emi is not None and stated_emi > 0) else benchmark_emi

        # Directional signed EMI difference: positive means stated EMI exceeds benchmark
        emi_diff = round(emi_to_use - benchmark_emi, 2)
        is_emi_matching = abs(emi_diff) <= 5.0  # Tolerance margin for paise rounding differences

        # 3. Fee, Insurance, and 18% Statutory GST Calculations
        banking_fees_base = loan.processing_fee + loan.documentation_charges
        fee_gst_amount = round(banking_fees_base * INDIAN_BANKING_GST_RATE, 2)
        total_upfront_fees_incl_gst = round(banking_fees_base + fee_gst_amount, 2)

        # Net Disbursed = Principal - Upfront Fees - GST - Insurance (if deducted from disbursal)
        upfront_insurance_deducted = 0.0 if loan.is_insurance_capitalized else loan.bundled_insurance
        net_disbursed = round(
            principal - total_upfront_fees_incl_gst - upfront_insurance_deducted, 2
        )

        # 4. Total Cash Outflows and Borrowing Costs
        total_repayment = round(emi_to_use * tenure, 2)
        gross_loan = financed_principal
        total_interest_payable = round(total_repayment - gross_loan, 2)
        total_cost_of_borrowing = round(
            (total_repayment + total_upfront_fees_incl_gst + upfront_insurance_deducted) - principal,
            2,
        )

        # 5. True APR & IRR Cash Flow Schedule
        discrepancies: List[DiscrepancyItem] = []
        score = 100

        is_irr_computable = True
        monthly_irr: Optional[float] = None
        nominal_apr: Optional[float] = None
        effective_annual_apr: Optional[float] = None
        rate_inflation_spread: Optional[float] = None

        if net_disbursed <= 0:
            is_irr_computable = False
            discrepancies.append(
                DiscrepancyItem(
                    code="UPFRONT_DEDUCTIONS_EXCEED_PRINCIPAL",
                    severity=DiscrepancySeverity.CRITICAL,
                    title="Upfront Fees Exceed Net Disbursal",
                    description=(
                        f"Upfront deductions (fees ₹{total_upfront_fees_incl_gst:,.2f} + insurance ₹{upfront_insurance_deducted:,.2f}) "
                        f"equal or exceed the sanctioned principal of ₹{principal:,.2f}. Net disbursal is ₹{net_disbursed:,.2f}. "
                        "The borrower receives zero or negative net cash while being liable for full loan repayment."
                    ),
                    impact_amount=round(total_upfront_fees_incl_gst + upfront_insurance_deducted, 2),
                )
            )
            score -= 60
        else:
            cash_flows = [net_disbursed] + [-emi_to_use] * tenure
            monthly_irr_raw = cls.calculate_irr_newton_raphson(cash_flows)
            if monthly_irr_raw is not None and not math.isnan(monthly_irr_raw) and not math.isinf(monthly_irr_raw):
                monthly_irr = round(monthly_irr_raw, 6)
                nominal_apr = round(monthly_irr_raw * 12.0 * 100.0, 2)
                effective_annual_apr = round((math.pow(1.0 + monthly_irr_raw, 12) - 1.0) * 100.0, 2)
                rate_inflation_spread = round(effective_annual_apr - stated_rate, 2)
            else:
                is_irr_computable = False

        # 6. Discrepancy & Anomaly Engine

        # Check A: Stated EMI Mismatch vs Standard Amortization
        if not is_emi_matching and stated_emi is not None:
            if not is_flat_structure and abs(stated_emi - flat_emi) <= 5.0 and stated_rate > 0:
                discrepancies.append(
                    DiscrepancyItem(
                        code="DISGUISED_FLAT_RATE",
                        severity=DiscrepancySeverity.CRITICAL,
                        title="Deceptive Flat Rate Disguised as Reducing",
                        description=(
                            f"The sanctioned EMI of ₹{stated_emi:,.2f} exactly matches a FLAT interest calculation. "
                            f"On a reducing balance, an interest rate of {stated_rate}% requires an EMI of "
                            f"₹{reducing_emi:,.2f}. The lender is disguising a flat-rate product as reducing balance."
                        ),
                        impact_amount=round((stated_emi - reducing_emi) * tenure, 2),
                    )
                )
                score -= 35
            elif emi_diff > 5.0:
                discrepancies.append(
                    DiscrepancyItem(
                        code="STATED_EMI_EXCEEDS_AMORTIZATION",
                        severity=DiscrepancySeverity.WARNING,
                        title="Stated EMI Exceeds Standard Amortization",
                        description=(
                            f"Sanctioned EMI of ₹{stated_emi:,.2f} exceeds calculated standard EMI of "
                            f"₹{benchmark_emi:,.2f} by ₹{emi_diff:,.2f}/month. Over {tenure} months, this "
                            f"totals an extra ₹{emi_diff * tenure:,.2f} in unamortized interest or charges."
                        ),
                        impact_amount=round(emi_diff * tenure, 2),
                    )
                )
                score -= 15
            elif emi_diff < -5.0:
                discrepancies.append(
                    DiscrepancyItem(
                        code="STATED_EMI_BELOW_AMORTIZATION",
                        severity=DiscrepancySeverity.WARNING,
                        title="Stated EMI Below Standard Amortization",
                        description=(
                            f"Sanctioned EMI of ₹{stated_emi:,.2f} is lower than the standard calculated EMI of "
                            f"₹{benchmark_emi:,.2f} by ₹{abs(emi_diff):,.2f}/month. This indicates a potential teaser rate period, "
                            "balloon payment structure, or negative amortization where the principal is not amortizing at the stated rate."
                        ),
                        impact_amount=round(abs(emi_diff) * tenure, 2),
                    )
                )
                score -= 15

        # Check B: Flat Rate Loan Notice
        if is_flat_structure:
            equiv_apr_note = f" (Effective Reducing APR: {effective_annual_apr:.2f}%)" if effective_annual_apr else ""
            discrepancies.append(
                DiscrepancyItem(
                    code="FLAT_RATE_STRUCTURE",
                    severity=DiscrepancySeverity.CRITICAL,
                    title="Flat Interest Rate Structure Detected",
                    description=(
                        f"This facility is structured as a FLAT interest rate ({stated_rate}% flat){equiv_apr_note}. "
                        "Interest is charged on the original principal for the entire loan duration without reducing "
                        "as principal is repaid. A flat rate of 10% is typically equivalent to ~18-19% reducing balance APR."
                    ),
                    impact_amount=round((flat_emi - reducing_emi) * tenure, 2) if flat_emi > reducing_emi else None,
                )
            )
            score -= 25

        # Check C: Capitalized vs Bundled Insurance
        if loan.bundled_insurance > 0:
            sev = (
                DiscrepancySeverity.WARNING
                if loan.bundled_insurance < (0.03 * principal)
                else DiscrepancySeverity.CRITICAL
            )
            if loan.is_insurance_capitalized:
                discrepancies.append(
                    DiscrepancyItem(
                        code="CAPITALIZED_INSURANCE",
                        severity=sev,
                        title="Insurance Premium Capitalized into Loan Principal",
                        description=(
                            f"An insurance premium of ₹{loan.bundled_insurance:,.2f} has been added directly to your debt principal, "
                            f"increasing your gross borrowing to ₹{financed_principal:,.2f}. You will pay interest on this insurance "
                            f"premium over the entire {tenure}-month tenure. Under RBI Fair Practices Code, loan-linked insurance must be "
                            "optional and borrowers must be offered the choice to pay upfront rather than financing."
                        ),
                        impact_amount=loan.bundled_insurance,
                    )
                )
            else:
                discrepancies.append(
                    DiscrepancyItem(
                        code="BUNDLED_INSURANCE",
                        severity=sev,
                        title="Bundled Insurance Deducted Upfront",
                        description=(
                            f"An insurance premium of ₹{loan.bundled_insurance:,.2f} has been deducted upfront from your disbursal. "
                            "Under RBI Fair Practices Code and consumer lending guidelines, insurance cannot be made a mandatory "
                            "tie-in condition for loan sanction."
                        ),
                        impact_amount=loan.bundled_insurance,
                    )
                )
            score -= 15

        # Check D: Processing Fees and 18% Statutory GST Notice
        if banking_fees_base > 0:
            discrepancies.append(
                DiscrepancyItem(
                    code="STATUTORY_GST_FEE",
                    severity=DiscrepancySeverity.INFO,
                    title="18% Statutory GST on Banking Charges (SAC 9971)",
                    description=(
                        f"Lender fees of ₹{banking_fees_base:,.2f} attract 18% GST (₹{fee_gst_amount:,.2f}). "
                        f"Total upfront fee deducted is ₹{total_upfront_fees_incl_gst:,.2f}."
                    ),
                    impact_amount=fee_gst_amount,
                )
            )

        # Check E: APR Spread Alert (Real Cost vs Stated Rate)
        if rate_inflation_spread is not None and rate_inflation_spread >= 2.0:
            discrepancies.append(
                DiscrepancyItem(
                    code="HIGH_APR_SPREAD",
                    severity=(
                        DiscrepancySeverity.WARNING
                        if rate_inflation_spread < 5.0
                        else DiscrepancySeverity.CRITICAL
                    ),
                    title="Significant True Cost (APR) Inflation",
                    description=(
                        f"Due to upfront processing fees and deductions, your True Effective APR is "
                        f"{effective_annual_apr:.2f}%, which is {rate_inflation_spread:.2f}% higher "
                        f"than the headline stated rate of {stated_rate}%."
                    ),
                    impact_amount=round(total_cost_of_borrowing - total_interest_payable, 2),
                )
            )
            score -= 20

        # Check F: RBI Prepayment Penalty Rules for Retail Loans
        if loan.has_prepayment_penalty:
            if loan.interest_type == InterestRateType.FLOATING:
                discrepancies.append(
                    DiscrepancyItem(
                        code="POTENTIAL_FLOATING_PREPAYMENT_RESTRICTION",
                        severity=DiscrepancySeverity.CRITICAL,
                        title="Potential RBI Violation: Foreclosure Charges on Floating Rate",
                        description=(
                            "Potential regulatory compliance concern: Under RBI Circulars DBOD.No.Dir.BC.107/13.03.00/2013-14 "
                            "and DNBR (PD) CC.No.101/03.10.001/2019-20, banks and NBFCs are prohibited from charging foreclosure "
                            "charges or pre-payment penalties on floating rate term loans sanctioned to individual borrowers "
                            "for purposes other than business. Additionally, the RBI Master Direction on Key Fact Statement (KFS) "
                            "(RBI/2023-24/147) mandates unbundling and full disclosure of charges."
                        ),
                        impact_amount=None,
                    )
                )
                score -= 25
            elif loan.interest_type == InterestRateType.UNKNOWN or loan.interest_type == InterestRateType.REDUCING:
                discrepancies.append(
                    DiscrepancyItem(
                        code="PREPAYMENT_PENALTY_VERIFICATION_REQUIRED",
                        severity=DiscrepancySeverity.WARNING,
                        title="Prepayment Penalty: Verify Rate Structure (Fixed vs Floating)",
                        description=(
                            "A prepayment penalty clause is present, but the loan rate structure (fixed vs floating) requires verification. "
                            "Under RBI Master Directions (DBOD.No.Dir.BC.107/13.03.00/2013-14), foreclosure charges are strictly prohibited "
                            "on floating-rate loans for individuals. On fixed-rate loans, prepayment charges are permissible provided "
                            "they are clearly disclosed upfront in the Key Fact Statement (KFS)."
                        ),
                        impact_amount=None,
                    )
                )
                score -= 10
            elif loan.interest_type == InterestRateType.FIXED:
                discrepancies.append(
                    DiscrepancyItem(
                        code="FIXED_RATE_PREPAYMENT_DISCLOSURE",
                        severity=DiscrepancySeverity.INFO,
                        title="Fixed-Rate Prepayment Charges Disclosed",
                        description=(
                            "Prepayment penalty noted on a fixed-rate loan. Under RBI guidelines (including Master Direction "
                            "RBI/2023-24/147 on Key Fact Statements), prepayment charges on fixed-rate loans are permissible "
                            "provided they were transparently disclosed in the KFS."
                        ),
                        impact_amount=None,
                    )
                )

        score = max(0, min(100, score))

        return LoanAuditReport(
            lender_name=loan.lender_name,
            principal=principal,
            tenure_months=tenure,
            stated_annual_rate=stated_rate,
            interest_type=loan.interest_type,
            stated_emi=stated_emi,
            calculated_reducing_emi=reducing_emi,
            calculated_flat_emi=flat_emi,
            emi_difference=emi_diff,
            is_emi_matching=is_emi_matching,
            raw_processing_fee=loan.processing_fee,
            raw_documentation_charges=loan.documentation_charges,
            fee_gst_amount=fee_gst_amount,
            total_upfront_fees_incl_gst=total_upfront_fees_incl_gst,
            bundled_insurance_amount=loan.bundled_insurance,
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
            is_verified=loan.is_verified,
            missing_fields=loan.missing_fields,
            discrepancies=discrepancies,
            rbi_compliance_score=score,
            explanation=None,
            extraction_method=loan.extraction_method,
        )

    @classmethod
    def compare_loans(
        cls, loan_a: ExtractedLoanTerms, loan_b: ExtractedLoanTerms
    ) -> Tuple[LoanAuditReport, LoanAuditReport, int]:
        """
        Audits both loans deterministically and ranks them based on Total Cost of Borrowing.
        Returns: (audit_a, audit_b, winner_index: 1 for A, 2 for B)
        """
        audit_a = cls.audit_loan(loan_a)
        audit_b = cls.audit_loan(loan_b)

        if audit_a.total_cost_of_borrowing <= audit_b.total_cost_of_borrowing:
            return audit_a, audit_b, 1
        return audit_a, audit_b, 2
