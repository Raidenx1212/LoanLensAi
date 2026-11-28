import os
import sys
import pymupdf as fitz
import pytest

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.schemas.loan import (
    ChargeCategory,
    CrossValidationStatus,
    FieldStatus,
    InterestRateType,
    Language,
)
from app.services.extractor import DocumentExtractor
from app.services.llm_service import OpenRouterLLMService
from app.services.auditor import DeterministicLoanAuditor


def _build_adversarial_sanction_pdf_bytes() -> bytes:
    """
    Constructs a realistic multi-page adversarial Indian bank sanction letter PDF
    containing key terms, financed insurance, opening balance, upfront recoveries,
    unquantified GST wording, and a conditional mandate charge in an annexure footnote.
    """
    doc = fitz.open()

    # Page 1: Sanction summary table & primary loan parameters
    p1 = doc.new_page()
    p1_text = (
        "ACME HOUSING FINANCE INDIA LTD.\n"
        "REVISED SANCTION LETTER & KEY FACT STATEMENT\n"
        "Application Ref: AHFL/2025/448912\n\n"
        "1. Sanctioned Principal Amount      : Rs. 12,00,000.00\n"
        "2. Interest Rate                    : 9.25% p.a. (Floating)\n"
        "3. Repayment Period                 : 120 Monthly Instalments\n"
        "4. Equated Monthly Instalment (EMI) : Rs. 15,602.07\n"
        "5. Financed Insurance Premium       : Rs. 18,600.00 (capitalised into loan balance)\n"
        "6. Opening Financed Balance         : Rs. 12,18,600.00\n"
    )
    p1.insert_text((50, 60), p1_text, fontsize=10)

    # Page 2: Annexure I - Fees, Recoveries & Conditional Footnotes
    p2 = doc.new_page()
    p2_text = (
        "ANNEXURE I: SCHEDULE OF FEES, RECOVERIES & CONDITIONAL CHARGES\n\n"
        "A. Upfront Recoveries from Borrower:\n"
        "   - Processing Fee         : Rs. 15,000 plus applicable GST\n"
        "   - Other Cash Recoveries  : Rs. 13,118.00 (valuation, legal & CERSAI composite recovery)\n\n"
        "B. Final Net Disbursement:\n"
        "   Final disbursement amount shall be communicated post tax invoice generation.\n\n"
        "Footnote 1: Conditional mandate charge of Rs. 590 applicable if physical NACH re-presentation is triggered.\n"
    )
    p2.insert_text((50, 60), p2_text, fontsize=10)

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_adversarial_sanction_letter_full_pipeline():
    """
    Mandatory regression test for the second adversarial sanction-letter scenario:
    Verifies exact extraction, financed insurance capitalization, upfront cash recovery total
    before GST, separated conditional fee, UNKNOWN GST & net disbursement, gated APR,
    evidence snippets with source pages, and blocked negotiation script.
    """
    pdf_bytes = _build_adversarial_sanction_pdf_bytes()
    pages = DocumentExtractor.extract_pages_from_pdf_bytes(pdf_bytes)
    assert len(pages) == 2

    svc = OpenRouterLLMService()
    terms = svc._deterministic_page_extract(pages)
    report = DeterministicLoanAuditor.audit_loan(terms)

    # 1. Core extracted parameters (no false zeros)
    assert report.principal == 1200000.0
    assert report.stated_annual_rate == 9.25
    assert report.interest_type == InterestRateType.FLOATING
    assert report.tenure_months == 120
    assert report.bundled_insurance_amount == 18600.0
    assert report.is_insurance_capitalized is True
    assert report.opening_financed_balance == 1218600.0
    assert report.gross_loan_amount == 1218600.0
    assert report.stated_emi == pytest.approx(15602.07, abs=0.05)
    assert report.calculated_reducing_emi == pytest.approx(15602.07, abs=0.05)
    assert report.is_emi_matching is True

    # 2. Charge separation: confirmed upfront recoveries before GST = 28,118; conditional = 590
    assert report.raw_processing_fee == pytest.approx(15000.0, abs=0.01)
    assert report.other_cash_recoveries == pytest.approx(13118.0, abs=0.01)
    assert report.known_upfront_recoveries_excl_tax == pytest.approx(28118.0, abs=0.01)
    assert report.conditional_charges_total == pytest.approx(590.0, abs=0.01)

    upfront_charges = [c for c in report.charges_breakdown if c.category == ChargeCategory.CONFIRMED_UPFRONT]
    cond_charges = [c for c in report.charges_breakdown if c.category == ChargeCategory.CONDITIONAL]
    fin_charges = [c for c in report.charges_breakdown if c.category == ChargeCategory.FINANCED]
    assert len(upfront_charges) == 2
    assert len(cond_charges) == 1
    assert len(fin_charges) == 1
    assert cond_charges[0].source_page == 2
    assert "590" in (cond_charges[0].evidence_snippet or "")

    # 3. Missing GST and Net Disbursement remain UNKNOWN (None) & APR is withheld
    assert report.fee_gst_amount is None
    assert report.total_upfront_fees_incl_gst is None
    assert report.net_disbursed_amount is None
    assert report.field_statuses["gst_amount"].status == FieldStatus.NOT_FOUND
    assert report.field_statuses["net_disbursement"].status == FieldStatus.NOT_FOUND
    assert report.is_irr_computable is False
    assert report.effective_annual_apr is None
    assert report.rbi_compliance_score is None

    # 4. Cross-validation passes for balance, insurance capitalization, and EMI
    cv_map = {cv.check_name: cv for cv in report.cross_validations}
    assert cv_map["Principal vs Opening Financed Balance"].status == CrossValidationStatus.PASS
    assert cv_map["Insurance vs Capitalized Principal"].status == CrossValidationStatus.PASS
    assert cv_map["Stated EMI vs Independently Calculated EMI"].status == CrossValidationStatus.PASS

    # 5. Source pages and evidence snippets retained on findings
    for key in ("principal_amount", "stated_interest_rate", "tenure_months", "stated_emi", "opening_financed_balance"):
        assert report.field_statuses[key].status == FieldStatus.CONFIRMED
        assert report.field_statuses[key].source_page == 1
        assert report.field_statuses[key].evidence_snippet

    # 6. Negotiation script blocked and no invented citations/SAC/compliance scores in explanation
    assert report.negotiation_script_blocked is True
    explanation = svc._local_template_explanation(report, Language.ENGLISH)
    assert "UNKNOWN" in explanation
    assert "Negotiation Script Blocked" in explanation
    assert "SAC 9971" not in explanation
    assert "RBI/2014-15/168" not in explanation
    assert "50/100" not in explanation


def test_sample_sanction_letter_real_pdf():
    """Verifies real repository sample_sanction_letter.pdf where 18% GST IS explicitly stated."""
    sample_path = os.path.join(BACKEND_DIR, "sample_sanction_letter.pdf")
    if not os.path.exists(sample_path):
        pytest.skip("sample_sanction_letter.pdf not found")

    with open(sample_path, "rb") as f:
        pdf_bytes = f.read()

    pages = DocumentExtractor.extract_pages_from_pdf_bytes(pdf_bytes)
    svc = OpenRouterLLMService()
    terms = svc._deterministic_page_extract(pages)
    report = DeterministicLoanAuditor.audit_loan(terms)

    assert report.principal == 500000.0
    assert report.stated_annual_rate == 11.5
    assert report.tenure_months == 36
    assert report.stated_emi == 16490.0
    assert report.known_upfront_recoveries_excl_tax == pytest.approx(8700.0, abs=0.01)
    assert report.stated_gst_rate_pct == 18.0
    assert report.fee_gst_amount == pytest.approx(1566.0, abs=0.01)
    assert report.bundled_insurance_amount == 14800.0
    assert report.net_disbursed_amount == pytest.approx(474934.0, abs=1.0)
    assert report.is_irr_computable is True
    assert report.effective_annual_apr is not None
    assert report.negotiation_script_blocked is False


def test_incomplete_pdf_fails_safely_without_false_zeros():
    """
    A partial sanction letter missing rate and tenure must NOT invent 0% rate,
    0 months tenure, 0 EMI, ₹0 net disbursement, or a 50/100 compliance score.
    """
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text(
        (50, 60),
        "PROVISIONAL OFFER\nSanctioned Principal Amount: Rs. 8,00,000\nProcessing Fee: Rs. 10,000 plus applicable GST\n",
        fontsize=10,
    )
    pdf_bytes = doc.tobytes()
    doc.close()

    pages = DocumentExtractor.extract_pages_from_pdf_bytes(pdf_bytes)
    svc = OpenRouterLLMService()
    terms = svc._deterministic_page_extract(pages)
    report = DeterministicLoanAuditor.audit_loan(terms)

    assert report.principal == 800000.0
    assert report.stated_annual_rate is None
    assert report.tenure_months is None
    assert report.calculated_reducing_emi is None
    assert report.net_disbursed_amount is None
    assert report.effective_annual_apr is None
    assert report.rbi_compliance_score is None
    assert report.field_statuses["stated_interest_rate"].status == FieldStatus.NOT_FOUND
    assert report.field_statuses["tenure_months"].status == FieldStatus.NOT_FOUND
    assert report.negotiation_script_blocked is True


def test_hero_housing_multipage_real_pdf():
    """Verifies real 5-page Hero Housing Finance sanction letter extraction without false zeros."""
    hero_path = (
        r"C:\Users\ANB\.gemini\antigravity\brain\d5169085-7e05-46af-96e0-90c28ab726cc"
        r"\.user_uploaded\media_1795491951829_d0ea09c0.pdf"
    )
    if not os.path.exists(hero_path):
        pytest.skip("Hero Housing PDF not present in local brain artifacts")

    with open(hero_path, "rb") as f:
        pdf_bytes = f.read()

    pages = DocumentExtractor.extract_pages_from_pdf_bytes(pdf_bytes)
    assert len(pages) == 5

    svc = OpenRouterLLMService()
    terms = svc._deterministic_page_extract(pages)
    report = DeterministicLoanAuditor.audit_loan(terms)

    assert report.principal == 4500000.0
    assert report.stated_annual_rate == 10.05
    assert report.interest_type == InterestRateType.FLOATING
    assert report.tenure_months == 134
    assert report.stated_emi == 56002.0
    assert report.is_emi_matching is True
    assert report.raw_processing_fee == 53100.0
    assert report.bundled_insurance_amount == 25000.0
    # Processing Charges states "1.00% + Applicable taxes" without explicit GST %, so GST & Net remain UNKNOWN
    assert report.fee_gst_amount is None
    assert report.net_disbursed_amount is None
    assert report.is_irr_computable is False
    assert report.rbi_compliance_score is None
    # Bounce charge on Page 4 captured as conditional
    assert any(c.category == ChargeCategory.CONDITIONAL and c.source_page == 4 for c in report.charges_breakdown)
