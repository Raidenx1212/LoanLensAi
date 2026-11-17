import pytest
from app.schemas.loan import ExtractedLoanTerms, InterestRateType
from app.services.auditor import DeterministicLoanAuditor


def test_standard_reducing_emi_calculation():
    # ₹10,00,000 for 5 years (60 months) at 10.5% per annum
    # Standard reducing balance formula yields ₹21,493.90
    emi = DeterministicLoanAuditor.calculate_reducing_emi(1000000.0, 10.5, 60)
    assert emi == 21493.90


def test_gst_and_net_disbursal_calculation():
    loan = ExtractedLoanTerms(
        lender_name="HDFC Bank",
        principal_amount=500000.0,
        stated_interest_rate=12.0,
        interest_type=InterestRateType.REDUCING,
        tenure_months=36,
        stated_emi=16607.16,
        processing_fee=5000.0,
        documentation_charges=1000.0,
        bundled_insurance=12000.0,
        is_insurance_capitalized=False,
    )
    audit = DeterministicLoanAuditor.audit_loan(loan)

    # Base fees: 5000 + 1000 = 6000
    # 18% Statutory GST on 6000 = 1080.0
    assert audit.fee_gst_amount == 1080.0
    assert audit.total_upfront_fees_incl_gst == 7080.0

    # Net Disbursal = 500000 - 7080 (fees+GST) - 12000 (insurance) = 480920.0
    assert audit.net_disbursed_amount == 480920.0

    # True APR must be strictly greater than stated 12.0% due to upfront fees & insurance
    assert audit.effective_annual_apr > 12.0


def test_detect_disguised_flat_rate():
    # ₹1,00,000 at 8% for 2 years (24 months)
    # Reducing EMI should be ₹4,522.73
    # Flat EMI: (100000 + (100000 * 0.08 * 2)) / 24 = ₹4,833.33
    loan = ExtractedLoanTerms(
        lender_name="Deceptive NBFC",
        principal_amount=100000.0,
        stated_interest_rate=8.0,
        interest_type=InterestRateType.REDUCING,
        tenure_months=24,
        stated_emi=4833.33,  # Stated flat EMI disguised as reducing
        processing_fee=2000.0,
        documentation_charges=0.0,
        bundled_insurance=0.0,
    )
    audit = DeterministicLoanAuditor.audit_loan(loan)

    discrepancy_codes = [d.code for d in audit.discrepancies]
    assert "DISGUISED_FLAT_RATE" in discrepancy_codes
    assert audit.rbi_compliance_score <= 65
    # The real APR on 8% flat is over 15% reducing!
    assert audit.effective_annual_apr >= 15.0


def test_rbi_floating_rate_prepayment_penalty_violation():
    loan = ExtractedLoanTerms(
        lender_name="Private Bank",
        principal_amount=2500000.0,
        stated_interest_rate=8.5,
        interest_type=InterestRateType.FLOATING,
        tenure_months=180,
        stated_emi=24618.63,
        has_prepayment_penalty=True,
        prepayment_penalty_terms="3% foreclosure fee if closed within 2 years",
    )
    audit = DeterministicLoanAuditor.audit_loan(loan)
    discrepancy_codes = [d.code for d in audit.discrepancies]
    assert "ILLEGAL_PREPAYMENT_PENALTY" in discrepancy_codes
    assert audit.rbi_compliance_score <= 75


def test_loan_comparison_ranking():
    # Loan A: Lower fee, standard reducing rate
    loan_a = ExtractedLoanTerms(
        lender_name="Bank Alpha",
        principal_amount=500000.0,
        stated_interest_rate=10.5,
        interest_type=InterestRateType.REDUCING,
        tenure_months=36,
        processing_fee=2000.0,
    )
    # Loan B: Same rate, but heavy processing fee and bundled insurance
    loan_b = ExtractedLoanTerms(
        lender_name="Bank Beta",
        principal_amount=500000.0,
        stated_interest_rate=10.5,
        interest_type=InterestRateType.REDUCING,
        tenure_months=36,
        processing_fee=15000.0,
        bundled_insurance=15000.0,
    )
    audit_a, audit_b, winner = DeterministicLoanAuditor.compare_loans(loan_a, loan_b)
    assert winner == 1  # Bank Alpha has lower total cost of borrowing
    assert audit_a.total_cost_of_borrowing < audit_b.total_cost_of_borrowing
