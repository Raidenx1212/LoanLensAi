import math
import os
import sys
import pytest

# Ensure backend directory is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.schemas.loan import (
    ExtractedLoanTerms,
    InterestRateType,
    LoanCategory,
    DiscrepancySeverity,
)
from app.services.auditor import DeterministicLoanAuditor


def test_reducing_emi_exact():
    """Verify standard actuarial reducing EMI matches expected formula."""
    principal = 1000000.0
    rate = 10.5
    tenure = 120
    emi = DeterministicLoanAuditor.calculate_reducing_emi(principal, rate, tenure)
    # Expected: ~13493.50
    assert 13490.0 < emi < 13500.0


def test_flat_emi_exact():
    """Verify flat rate EMI matches principal + total interest / tenure."""
    principal = 1000000.0
    rate = 10.0
    tenure = 120  # 10 years
    emi = DeterministicLoanAuditor.calculate_flat_emi(principal, rate, tenure)
    # Total interest = 1,000,000 * 0.10 * 10 = 1,000,000
    # Total repayment = 2,000,000 -> EMI = 2,000,000 / 120 = 16666.67
    assert abs(emi - 16666.67) <= 0.05


def test_capitalized_insurance_impact():
    """
    When insurance is capitalized, EMI must be computed on financed principal
    (principal + insurance), but net disbursed must NOT subtract insurance twice.
    """
    loan = ExtractedLoanTerms(
        lender_name="Test Bank",
        loan_category=LoanCategory.HOME,
        principal_amount=4500000.0,
        stated_interest_rate=10.05,
        interest_type=InterestRateType.FLOATING,
        tenure_months=134,
        stated_emi=None,
        processing_fee=47200.0,
        documentation_charges=0.0,
        bundled_insurance=25000.0,
        is_insurance_capitalized=True,
    )
    report = DeterministicLoanAuditor.audit_loan(loan)

    # Gross loan financed should be principal + insurance
    assert report.gross_loan_amount == 4525000.0

    # Calculated reducing EMI should be computed on 4,525,000, not 4,500,000
    expected_emi_on_gross = DeterministicLoanAuditor.calculate_reducing_emi(4525000.0, 10.05, 134)
    assert report.calculated_reducing_emi == expected_emi_on_gross

    # Net disbursed: 45,00,000 - (47200 * 1.18 = 55696) = 44,44,304 (insurance is capitalized, not deducted)
    expected_fees_incl_gst = round(47200.0 * 1.18, 2)
    assert report.total_upfront_fees_incl_gst == expected_fees_incl_gst
    assert report.net_disbursed_amount == round(4500000.0 - expected_fees_incl_gst, 2)

    # Discrepancy should mention capitalized insurance
    cap_discs = [d for d in report.discrepancies if d.code == "CAPITALIZED_INSURANCE"]
    assert len(cap_discs) == 1


def test_upfront_deductions_exceed_principal_no_crash():
    """
    If upfront fees exceed principal (net disbursal <= 0),
    auditor must NOT crash with OverflowError or return absurd APRs.
    """
    loan = ExtractedLoanTerms(
        lender_name="Predatory Lender",
        loan_category=LoanCategory.PERSONAL,
        principal_amount=50000.0,
        stated_interest_rate=15.0,
        interest_type=InterestRateType.REDUCING,
        tenure_months=12,
        stated_emi=5000.0,
        processing_fee=50000.0,  # 50,000 + 18% GST = 59,000 > 50,000 principal
        documentation_charges=0.0,
    )
    report = DeterministicLoanAuditor.audit_loan(loan)

    assert report.is_irr_computable is False
    assert report.effective_annual_apr is None
    assert report.nominal_apr is None
    assert report.net_disbursed_amount < 0
    disc_codes = [d.code for d in report.discrepancies]
    assert "UPFRONT_DEDUCTIONS_EXCEED_PRINCIPAL" in disc_codes


def test_irr_newton_raphson_guards():
    """Verify calculate_irr_newton_raphson safely returns None on non-viable cash flows."""
    # 0 net disbursal
    assert DeterministicLoanAuditor.calculate_irr_newton_raphson([0.0, -500, -500]) is None
    # Negative net disbursal
    assert DeterministicLoanAuditor.calculate_irr_newton_raphson([-1000.0, -500, -500]) is None
    # No repayments (all positive or zero)
    assert DeterministicLoanAuditor.calculate_irr_newton_raphson([1000.0, 0.0, 100.0]) is None
    # Valid cash flows should return a positive rate
    rate = DeterministicLoanAuditor.calculate_irr_newton_raphson([100000.0] + [-10000.0] * 12)
    assert rate is not None
    assert rate > 0


def test_signed_emi_mismatch_exceeds():
    """Stated EMI higher than benchmark must be flagged as STATED_EMI_EXCEEDS_AMORTIZATION."""
    reducing_emi = DeterministicLoanAuditor.calculate_reducing_emi(1000000.0, 10.0, 120)
    loan = ExtractedLoanTerms(
        principal_amount=1000000.0,
        stated_interest_rate=10.0,
        interest_type=InterestRateType.REDUCING,
        tenure_months=120,
        stated_emi=reducing_emi + 2000.0,  # 2000 higher
    )
    report = DeterministicLoanAuditor.audit_loan(loan)
    disc_codes = [d.code for d in report.discrepancies]
    assert "STATED_EMI_EXCEEDS_AMORTIZATION" in disc_codes
    assert report.emi_difference == 2000.0


def test_signed_emi_mismatch_below():
    """Stated EMI lower than standard amortization must be flagged as STATED_EMI_BELOW_AMORTIZATION."""
    reducing_emi = DeterministicLoanAuditor.calculate_reducing_emi(1000000.0, 10.0, 120)
    loan = ExtractedLoanTerms(
        principal_amount=1000000.0,
        stated_interest_rate=10.0,
        interest_type=InterestRateType.REDUCING,
        tenure_months=120,
        stated_emi=reducing_emi - 2000.0,  # 2000 lower
    )
    report = DeterministicLoanAuditor.audit_loan(loan)
    disc_codes = [d.code for d in report.discrepancies]
    assert "STATED_EMI_BELOW_AMORTIZATION" in disc_codes
    assert report.emi_difference == -2000.0


def test_flat_loan_structure_flagged():
    """When loan is explicitly FLAT, benchmark is flat EMI and FLAT_RATE_STRUCTURE is flagged."""
    loan = ExtractedLoanTerms(
        principal_amount=500000.0,
        stated_interest_rate=12.0,
        interest_type=InterestRateType.FLAT,
        tenure_months=36,
    )
    report = DeterministicLoanAuditor.audit_loan(loan)
    assert report.is_emi_matching is True
    disc_codes = [d.code for d in report.discrepancies]
    assert "FLAT_RATE_STRUCTURE" in disc_codes


def test_disguised_flat_rate_on_reducing():
    """If reducing balance loan charges flat EMI, flag DISGUISED_FLAT_RATE."""
    principal = 500000.0
    rate = 12.0
    tenure = 36
    flat_emi = DeterministicLoanAuditor.calculate_flat_emi(principal, rate, tenure)
    loan = ExtractedLoanTerms(
        principal_amount=principal,
        stated_interest_rate=rate,
        interest_type=InterestRateType.REDUCING,
        tenure_months=tenure,
        stated_emi=flat_emi,
    )
    report = DeterministicLoanAuditor.audit_loan(loan)
    disc_codes = [d.code for d in report.discrepancies]
    assert "DISGUISED_FLAT_RATE" in disc_codes


def test_rbi_prepayment_penalty_rules():
    """Check regulatory categorization for FLOATING, UNKNOWN, and FIXED loans."""
    # Floating loan with penalty -> Potential RBI violation
    loan_floating = ExtractedLoanTerms(
        principal_amount=1000000.0,
        stated_interest_rate=9.5,
        interest_type=InterestRateType.FLOATING,
        tenure_months=120,
        has_prepayment_penalty=True,
    )
    report_float = DeterministicLoanAuditor.audit_loan(loan_floating)
    codes_float = [d.code for d in report_float.discrepancies]
    assert "POTENTIAL_FLOATING_PREPAYMENT_RESTRICTION" in codes_float

    # Unknown rate loan with penalty -> Verification required
    loan_unknown = ExtractedLoanTerms(
        principal_amount=1000000.0,
        stated_interest_rate=9.5,
        interest_type=InterestRateType.UNKNOWN,
        tenure_months=120,
        has_prepayment_penalty=True,
    )
    report_unknown = DeterministicLoanAuditor.audit_loan(loan_unknown)
    codes_unknown = [d.code for d in report_unknown.discrepancies]
    assert "PREPAYMENT_PENALTY_VERIFICATION_REQUIRED" in codes_unknown

    # Fixed rate loan with penalty -> Permissible under KFS disclosure (Info)
    loan_fixed = ExtractedLoanTerms(
        principal_amount=1000000.0,
        stated_interest_rate=9.5,
        interest_type=InterestRateType.FIXED,
        tenure_months=120,
        has_prepayment_penalty=True,
    )
    report_fixed = DeterministicLoanAuditor.audit_loan(loan_fixed)
    codes_fixed = [d.code for d in report_fixed.discrepancies]
    assert "FIXED_RATE_PREPAYMENT_DISCLOSURE" in codes_fixed


def test_zero_hallucination_incomplete_data():
    """Unverified or zero principal/rate/tenure must pause audit cleanly without guessing numbers."""
    loan = ExtractedLoanTerms(
        principal_amount=0.0,
        stated_interest_rate=0.0,
        tenure_months=0,
        is_verified=False,
        missing_fields=["principal_amount", "stated_interest_rate", "tenure_months"],
    )
    report = DeterministicLoanAuditor.audit_loan(loan)
    assert report.is_verified is False
    assert report.is_irr_computable is False
    assert report.effective_annual_apr is None
    disc_codes = [d.code for d in report.discrepancies]
    assert "INCOMPLETE_LOAN_DATA" in disc_codes


def test_compare_loans():
    """Loan comparison must rank by total borrowing cost."""
    loan_cheaper = ExtractedLoanTerms(
        lender_name="Bank Alpha",
        principal_amount=1000000.0,
        stated_interest_rate=9.0,
        interest_type=InterestRateType.REDUCING,
        tenure_months=60,
        processing_fee=5000.0,
    )
    loan_costlier = ExtractedLoanTerms(
        lender_name="Bank Beta",
        principal_amount=1000000.0,
        stated_interest_rate=12.0,
        interest_type=InterestRateType.REDUCING,
        tenure_months=60,
        processing_fee=25000.0,
    )
    audit_a, audit_b, winner_idx = DeterministicLoanAuditor.compare_loans(loan_cheaper, loan_costlier)
    assert winner_idx == 1
    assert audit_a.total_cost_of_borrowing < audit_b.total_cost_of_borrowing
