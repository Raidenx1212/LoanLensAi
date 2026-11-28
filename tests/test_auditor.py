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
    FieldStatus,
    CrossValidationStatus,
    InterestRateType,
    LoanCategory,
)
from app.services.auditor import DeterministicLoanAuditor


def test_reducing_emi_exact():
    """Verify standard actuarial reducing EMI matches expected formula."""
    principal = 1000000.0
    rate = 10.5
    tenure = 120
    emi = DeterministicLoanAuditor.calculate_reducing_emi(principal, rate, tenure)
    assert 13490.0 < emi < 13500.0


def test_flat_emi_exact():
    """Verify flat rate EMI matches principal + total interest / tenure."""
    principal = 1000000.0
    rate = 10.0
    tenure = 120
    emi = DeterministicLoanAuditor.calculate_flat_emi(principal, rate, tenure)
    assert abs(emi - 16666.67) <= 0.05


def test_capitalized_insurance_impact_with_explicit_gst():
    """
    When insurance is capitalized and GST % is explicitly disclosed,
    EMI must be computed on financed principal (principal + insurance),
    and net disbursed must NOT subtract insurance twice.
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
        stated_gst_rate_pct=18.0,  # Explicitly disclosed GST rate
        documentation_charges=0.0,
        bundled_insurance=25000.0,
        is_insurance_capitalized=True,
    )
    report = DeterministicLoanAuditor.audit_loan(loan)

    assert report.gross_loan_amount == 4525000.0
    expected_emi_on_gross = DeterministicLoanAuditor.calculate_reducing_emi(4525000.0, 10.05, 134)
    assert report.calculated_reducing_emi == expected_emi_on_gross

    expected_fees_incl_gst = round(47200.0 * 1.18, 2)
    assert report.total_upfront_fees_incl_gst == expected_fees_incl_gst
    assert report.net_disbursed_amount == round(4500000.0 - expected_fees_incl_gst, 2)
    assert report.is_irr_computable is True
    assert report.negotiation_script_blocked is False

    cap_discs = [d for d in report.discrepancies if d.code == "CAPITALIZED_INSURANCE"]
    assert len(cap_discs) == 1


def test_missing_gst_never_invented():
    """When upfront fees exist without stated GST rate or amount, GST and Net Disbursement remain None."""
    loan = ExtractedLoanTerms(
        principal_amount=1000000.0,
        stated_interest_rate=10.0,
        interest_type=InterestRateType.FLOATING,
        tenure_months=60,
        processing_fee=10000.0,
        gst_status_note="plus applicable GST",
    )
    report = DeterministicLoanAuditor.audit_loan(loan)
    assert report.known_upfront_recoveries_excl_tax == 10000.0
    assert report.fee_gst_amount is None
    assert report.total_upfront_fees_incl_gst is None
    assert report.net_disbursed_amount is None
    assert report.is_irr_computable is False
    assert report.effective_annual_apr is None
    assert report.rbi_compliance_score is None
    assert report.negotiation_script_blocked is True


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
        processing_fee=50000.0,
        stated_gst_rate_pct=18.0,
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
    assert DeterministicLoanAuditor.calculate_irr_newton_raphson([0.0, -500, -500]) is None
    assert DeterministicLoanAuditor.calculate_irr_newton_raphson([-1000.0, -500, -500]) is None
    assert DeterministicLoanAuditor.calculate_irr_newton_raphson([1000.0, 0.0, 100.0]) is None
    rate = DeterministicLoanAuditor.calculate_irr_newton_raphson([100000.0] + [-10000.0] * 12)
    assert rate is not None
    assert rate > 0


def test_impossible_combinations_rejected():
    """Reject positive principal with zero tenure, zero EMI, or unsupported zero rate."""
    loan_zero_tenure = ExtractedLoanTerms(
        principal_amount=1200000.0,
        stated_interest_rate=9.25,
        tenure_months=0,
    )
    report_zt = DeterministicLoanAuditor.audit_loan(loan_zero_tenure)
    assert report_zt.audit_status == "REJECTED_INVALID_COMBINATION"
    assert report_zt.negotiation_script_blocked is True
    assert report_zt.effective_annual_apr is None
    assert report_zt.net_disbursed_amount is None
    assert report_zt.field_statuses["tenure_months"].status == FieldStatus.UNCERTAIN
    assert any(d.code == "IMPOSSIBLE_ZERO_TENURE" for d in report_zt.discrepancies)

    loan_zero_emi = ExtractedLoanTerms(
        principal_amount=500000.0,
        stated_interest_rate=10.0,
        tenure_months=36,
        stated_emi=0.0,
    )
    report_ze = DeterministicLoanAuditor.audit_loan(loan_zero_emi)
    assert report_ze.audit_status == "REJECTED_INVALID_COMBINATION"
    assert any(d.code == "IMPOSSIBLE_ZERO_EMI" for d in report_ze.discrepancies)


def test_cross_validation_schedule_and_balance():
    """Cross-validate opening financed balance mismatch and tenure vs instalment count."""
    emi = DeterministicLoanAuditor.calculate_reducing_emi(500000.0, 10.0, 12)
    loan = ExtractedLoanTerms(
        principal_amount=500000.0,
        opening_financed_balance=520000.0,  # Unexplained 20,000 inflation (no financed insurance)
        stated_interest_rate=10.0,
        tenure_months=12,
        schedule_instalment_count=15,  # Conflicts with 12
        schedule_emis=[emi] * 12,
    )
    report = DeterministicLoanAuditor.audit_loan(loan)
    cv_map = {cv.check_name: cv for cv in report.cross_validations}
    assert cv_map["Principal vs Opening Financed Balance"].status == CrossValidationStatus.FAIL
    assert cv_map["Tenure vs Number of Instalments"].status == CrossValidationStatus.FAIL
    assert report.negotiation_script_blocked is True


def test_signed_emi_mismatch_exceeds_and_below():
    """Stated EMI directional mismatches are classified cleanly."""
    reducing_emi = DeterministicLoanAuditor.calculate_reducing_emi(1000000.0, 10.0, 120)
    loan_high = ExtractedLoanTerms(
        principal_amount=1000000.0,
        stated_interest_rate=10.0,
        interest_type=InterestRateType.REDUCING,
        tenure_months=120,
        stated_emi=reducing_emi + 2000.0,
    )
    rep_high = DeterministicLoanAuditor.audit_loan(loan_high)
    assert any(d.code == "STATED_EMI_EXCEEDS_AMORTIZATION" for d in rep_high.discrepancies)
    assert rep_high.emi_difference == 2000.0

    loan_low = ExtractedLoanTerms(
        principal_amount=1000000.0,
        stated_interest_rate=10.0,
        interest_type=InterestRateType.REDUCING,
        tenure_months=120,
        stated_emi=reducing_emi - 2000.0,
    )
    rep_low = DeterministicLoanAuditor.audit_loan(loan_low)
    assert any(d.code == "STATED_EMI_BELOW_AMORTIZATION" for d in rep_low.discrepancies)
    assert rep_low.emi_difference == -2000.0


def test_flat_loan_and_disguised_flat_rate():
    """Verify flat-rate loan flagging and disguised flat rate on reducing balance."""
    loan_flat = ExtractedLoanTerms(
        principal_amount=500000.0,
        stated_interest_rate=12.0,
        interest_type=InterestRateType.FLAT,
        tenure_months=36,
    )
    rep_flat = DeterministicLoanAuditor.audit_loan(loan_flat)
    assert rep_flat.calculated_flat_emi == 18888.89
    assert rep_flat.is_emi_matching is None
    assert any(d.code == "FLAT_RATE_STRUCTURE" for d in rep_flat.discrepancies)

    flat_emi = DeterministicLoanAuditor.calculate_flat_emi(500000.0, 12.0, 36)
    loan_disguised = ExtractedLoanTerms(
        principal_amount=500000.0,
        stated_interest_rate=12.0,
        interest_type=InterestRateType.REDUCING,
        tenure_months=36,
        stated_emi=flat_emi,
    )
    rep_disg = DeterministicLoanAuditor.audit_loan(loan_disguised)
    assert any(d.code == "DISGUISED_FLAT_RATE" for d in rep_disg.discrepancies)


def test_missing_fields_stay_none_never_zero():
    """Missing core terms remain None (UNKNOWN), never converted to 0% or ₹0."""
    loan = ExtractedLoanTerms(
        principal_amount=1200000.0,
        stated_interest_rate=None,
        tenure_months=None,
        is_verified=False,
        missing_fields=["stated_interest_rate", "tenure_months"],
    )
    report = DeterministicLoanAuditor.audit_loan(loan)
    assert report.principal == 1200000.0
    assert report.stated_annual_rate is None
    assert report.tenure_months is None
    assert report.calculated_reducing_emi is None
    assert report.net_disbursed_amount is None
    assert report.effective_annual_apr is None
    assert report.rbi_compliance_score is None
    assert report.is_irr_computable is False
    assert report.negotiation_script_blocked is True
    assert report.field_statuses["stated_interest_rate"].status == FieldStatus.NOT_FOUND
    assert report.field_statuses["tenure_months"].status == FieldStatus.NOT_FOUND


def test_compare_loans():
    """Loan comparison ranks by total cost of borrowing."""
    loan_cheaper = ExtractedLoanTerms(
        lender_name="Bank Alpha",
        principal_amount=1000000.0,
        stated_interest_rate=9.0,
        interest_type=InterestRateType.REDUCING,
        tenure_months=60,
        processing_fee=5000.0,
        stated_gst_rate_pct=18.0,
    )
    loan_costlier = ExtractedLoanTerms(
        lender_name="Bank Beta",
        principal_amount=1000000.0,
        stated_interest_rate=12.0,
        interest_type=InterestRateType.REDUCING,
        tenure_months=60,
        processing_fee=25000.0,
        stated_gst_rate_pct=18.0,
    )
    audit_a, audit_b, winner_idx = DeterministicLoanAuditor.compare_loans(loan_cheaper, loan_costlier)
    assert winner_idx == 1
    assert audit_a.total_cost_of_borrowing < audit_b.total_cost_of_borrowing
