import math
from typing import List, Tuple
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
    """

    @staticmethod
    def calculate_reducing_emi(principal: float, annual_rate_pct: float, tenure_months: int) -> float:
        """Standard reducing balance EMI formula."""
        if tenure_months <= 0:
            return 0.0
        if annual_rate_pct <= 0:
            return round(principal / tenure_months, 2)

        monthly_rate = (annual_rate_pct / 100.0) / 12.0
        factor = math.pow(1 + monthly_rate, tenure_months)
        emi = principal * monthly_rate * factor / (factor - 1)
        return round(emi, 2)

    @staticmethod
    def calculate_flat_emi(principal: float, annual_rate_pct: float, tenure_months: int) -> float:
        """Flat rate EMI: Interest is calculated on full principal for entire tenure."""
        if tenure_months <= 0:
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
    ) -> float:
        """
        Calculates exact monthly Internal Rate of Return (IRR) using Newton-Raphson.
        cash_flows[0] = net disbursal (positive cash inflow to borrower)
        cash_flows[1..n] = monthly EMI payments (negative cash outflows)
        """
        rate = guess
        for _ in range(max_iter):
            npv = 0.0
            derivative_npv = 0.0
            for t, cf in enumerate(cash_flows):
                denom = math.pow(1.0 + rate, t)
                if denom == 0:
                    break
                npv += cf / denom
                if t > 0:
                    derivative_npv -= (t * cf) / (denom * (1.0 + rate))

            if abs(npv) < tol:
                return rate

            if derivative_npv == 0.0:
                break

            new_rate = rate - (npv / derivative_npv)
            if new_rate <= -0.999:  # Avoid singularity near -100%
                new_rate = -0.5
            if abs(new_rate - rate) < tol:
                return new_rate
            rate = new_rate

        # Bisection fallback if Newton-Raphson oscillates or diverges
        low = -0.5
        high = 5.0
        for _ in range(200):
            mid = (low + high) / 2.0
            npv = sum(cf / math.pow(1.0 + mid, t) for t, cf in enumerate(cash_flows))
            if abs(npv) < tol:
                return mid
            if npv > 0:
                low = mid
            else:
                high = mid

        return rate

    @classmethod
    def audit_loan(cls, loan: ExtractedLoanTerms) -> LoanAuditReport:
        principal = loan.principal_amount
        tenure = loan.tenure_months
        stated_rate = loan.stated_interest_rate

        # 1. EMI Calculations
        reducing_emi = cls.calculate_reducing_emi(principal, stated_rate, tenure)
        flat_emi = cls.calculate_flat_emi(principal, stated_rate, tenure)

        stated_emi = loan.stated_emi
        emi_to_use = stated_emi if (stated_emi is not None and stated_emi > 0) else reducing_emi

        emi_diff = round(abs(emi_to_use - reducing_emi), 2)
        is_emi_matching = emi_diff <= 5.0  # Tolerance margin for paise rounding differences

        # 2. Fee, Insurance, and 18% GST Calculations
        banking_fees_base = loan.processing_fee + loan.documentation_charges
        fee_gst_amount = round(banking_fees_base * INDIAN_BANKING_GST_RATE, 2)
        total_upfront_fees_incl_gst = round(banking_fees_base + fee_gst_amount, 2)

        # 3. Capitalization vs Disbursal Deduction for Insurance
        gross_loan = principal
        if loan.is_insurance_capitalized and loan.bundled_insurance > 0:
            gross_loan += loan.bundled_insurance

        # Net Disbursed = Principal - Upfront Fees - GST - Insurance (if deducted from disbursal)
        upfront_insurance_deducted = 0.0 if loan.is_insurance_capitalized else loan.bundled_insurance
        net_disbursed = round(
            principal - total_upfront_fees_incl_gst - upfront_insurance_deducted, 2
        )

        # 4. Total Cash Outflows
        total_repayment = round(emi_to_use * tenure, 2)
        total_interest_payable = round(total_repayment - gross_loan, 2)
        total_cost_of_borrowing = round(
            (total_repayment + total_upfront_fees_incl_gst + upfront_insurance_deducted) - principal,
            2,
        )

        # 5. True APR / IRR Cash Flow Schedule
        cash_flows = [net_disbursed] + [-emi_to_use] * tenure
        monthly_irr = cls.calculate_irr_newton_raphson(cash_flows)

        # Nominal APR (simple annualized) and Compounded Effective APR (EAR)
        nominal_apr = round(monthly_irr * 12.0 * 100.0, 2)
        effective_annual_apr = round((math.pow(1.0 + monthly_irr, 12) - 1.0) * 100.0, 2)
        rate_inflation_spread = round(effective_annual_apr - stated_rate, 2)

        # 6. Discrepancy & Anomaly Engine
        discrepancies: List[DiscrepancyItem] = []
        score = 100

        # Check A: Stated EMI mismatch
        if not is_emi_matching and stated_emi is not None:
            # Check if stated EMI matches flat calculation instead of reducing
            if abs(stated_emi - flat_emi) <= 5.0:
                discrepancies.append(
                    DiscrepancyItem(
                        code="DISGUISED_FLAT_RATE",
                        severity=DiscrepancySeverity.CRITICAL,
                        title="Deceptive Flat Rate Disguised as Reducing",
                        description=(
                            f"The stated EMI of ₹{stated_emi:,.2f} matches a FLAT interest rate. "
                            f"On a reducing balance, an interest rate of {stated_rate}% requires an EMI of "
                            f"₹{reducing_emi:,.2f}. The lender is hiding the true interest burden."
                        ),
                        impact_amount=round((stated_emi - reducing_emi) * tenure, 2),
                    )
                )
                score -= 35
            else:
                discrepancies.append(
                    DiscrepancyItem(
                        code="EMI_MISMATCH",
                        severity=DiscrepancySeverity.WARNING,
                        title="Stated EMI Exceeds Standard Amortization",
                        description=(
                            f"Sanctioned EMI of ₹{stated_emi:,.2f} exceeds calculated reducing EMI of "
                            f"₹{reducing_emi:,.2f} by ₹{emi_diff:,.2f}/month. Over {tenure} months, this "
                            f"totals an extra ₹{emi_diff * tenure:,.2f}."
                        ),
                        impact_amount=round(emi_diff * tenure, 2),
                    )
                )
                score -= 15

        # Check B: Hidden / Bundled Insurance
        if loan.bundled_insurance > 0:
            sev = (
                DiscrepancySeverity.WARNING
                if loan.bundled_insurance < (0.03 * principal)
                else DiscrepancySeverity.CRITICAL
            )
            deduction_method = (
                "capitalized into your loan principal"
                if loan.is_insurance_capitalized
                else "deducted upfront from disbursal"
            )
            discrepancies.append(
                DiscrepancyItem(
                    code="BUNDLED_INSURANCE",
                    severity=sev,
                    title="Bundled Insurance Premium Detected",
                    description=(
                        f"An insurance premium of ₹{loan.bundled_insurance:,.2f} has been {deduction_method}. "
                        "As per RBI directives, loan protection insurance must be optional and cannot be made "
                        "a mandatory precondition for loan sanction."
                    ),
                    impact_amount=loan.bundled_insurance,
                )
            )
            score -= 15

        # Check C: Processing Fees and 18% GST Notice
        if banking_fees_base > 0:
            discrepancies.append(
                DiscrepancyItem(
                    code="STATUTORY_GST_FEE",
                    severity=DiscrepancySeverity.INFO,
                    title="18% Statutory GST on Banking Charges",
                    description=(
                        f"Lender fees of ₹{banking_fees_base:,.2f} attract 18% GST (₹{fee_gst_amount:,.2f}). "
                        f"Total upfront fee deducted is ₹{total_upfront_fees_incl_gst:,.2f}."
                    ),
                    impact_amount=fee_gst_amount,
                )
            )

        # Check D: APR Spread Alert (Real Cost vs Stated Rate)
        if rate_inflation_spread >= 2.0:
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

        # Check E: RBI Prepayment Penalty Rule for Retail Loans
        if loan.has_prepayment_penalty and (
            loan.interest_type in [InterestRateType.FLOATING, InterestRateType.REDUCING]
        ):
            discrepancies.append(
                DiscrepancyItem(
                    code="ILLEGAL_PREPAYMENT_PENALTY",
                    severity=DiscrepancySeverity.CRITICAL,
                    title="Potential RBI Violation: Foreclosure Penalty on Floating Rate",
                    description=(
                        "Under RBI circular RBI/2014-15/168, banks and NBFCs are strictly prohibited from "
                        "charging foreclosure charges or pre-payment penalties on floating rate term loans "
                        "sanctioned to individual borrowers."
                    ),
                    impact_amount=None,
                )
            )
            score -= 25

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
            true_monthly_irr=round(monthly_irr, 6),
            nominal_apr=nominal_apr,
            effective_annual_apr=effective_annual_apr,
            rate_inflation_spread=rate_inflation_spread,
            discrepancies=discrepancies,
            rbi_compliance_score=score,
            explanation=None,
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
