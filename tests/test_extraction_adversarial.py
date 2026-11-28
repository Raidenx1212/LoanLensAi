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
    GSTClassification,
    ThirdPartyRecoveryMethod,
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

    # 2. Charge separation: confirmed upfront recoveries before GST = 28,118; conditional = 590 (not summed)
    assert report.raw_processing_fee == pytest.approx(15000.0, abs=0.01)
    assert report.other_cash_recoveries == pytest.approx(13118.0, abs=0.01)
    assert report.known_upfront_recoveries_excl_tax == pytest.approx(28118.0, abs=0.01)
    assert report.conditional_charges_total is None
    assert len(report.conditional_charges) == 1
    assert report.conditional_charges[0].amount == pytest.approx(590.0, abs=0.01)

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
    # Title Search Charges ("applicable only for Mumbai Metropolitan Region") classified CONDITIONAL, not mandatory upfront
    assert report.known_upfront_recoveries_excl_tax == 53100.0
    assert report.conditional_charges_total is None
    title_cond = next((c for c in report.conditional_charges if "title search" in c.name.lower()), None)
    assert title_cond is not None
    assert title_cond.amount == 2500.0
    assert "mumbai metropolitan" in (title_cond.trigger_condition or "").lower()

    # Bounce charge on Page 4 captured accurately as PER_BOUNCE_EVENT (NOT assisted mandate setup!)
    bounce_cond = next((c for c in report.conditional_charges if "bounce" in c.name.lower()), None)
    assert bounce_cond is not None
    assert bounce_cond.source_page == 4
    assert bounce_cond.frequency == "PER_BOUNCE_EVENT"
    assert "mandate setup" not in (bounce_cond.trigger_condition or "").lower()
    assert bounce_cond.source_section == "GENERAL TERMS AND CONDITIONS"

    # Penal Charge on Page 4 captured as 2.00% per month (NOT per annum, NOT merged with Clause 26)
    penal_cond = next((c for c in report.conditional_charges if "penal" in c.name.lower()), None)
    assert penal_cond is not None
    assert penal_cond.frequency == "PER_MONTH_ON_OVERDUE"
    assert "per month" in (penal_cond.rate_basis or "").lower()
    assert "26." not in (penal_cond.evidence_snippet or "")
    assert penal_cond.source_section == "GENERAL TERMS AND CONDITIONS"

    # Customer-Funded Insurance surfaced in charge reconciliation and findings
    assert any("insurance" in c.name.lower() and c.amount == 25000.0 for c in report.charges_breakdown)
    assert any(d.code == "BUNDLED_INSURANCE" and d.impact_amount == 25000.0 for d in report.discrepancies)

    # Processing Charges states "1.00% + Applicable taxes" without explicit GST %, so GST & Net remain UNKNOWN
    assert report.fee_gst_amount is None
    assert report.net_disbursed_amount is None
    assert report.is_irr_computable is False
    assert report.rbi_compliance_score is None


def test_northstar_three_page_adversarial_sanction_letter():
    """
    Verifies the 3-page Northstar Housing Finance adversarial sanction communication
    including percentage-of-principal fee, financed credit-protection premium,
    multi-page charges/reserves totaling ₹28,118 before tax, conditional ₹590 mandate fee,
    UNKNOWN indirect tax & final disbursement, gated APR, and blocked negotiation script.
    """
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text(
        (40, 40),
        (
            "NORTHSTAR HOUSING FINANCE LTD.\n"
            "Illustrative Sanction Communication | Ref: NSHF/HL/26/11842\n"
            "1. Key facility particulars\n"
            "Facility | Home loan - completed residential property\n"
            "Sanctioned principal | Rs. 12,00,000.00\n"
            "Tenure | 120 monthly instalments\n"
            "Interest rate | 9.25% p.a. floating; monthly rests; reducing balance\n"
            "Indicative first EMI | Rs. 15,602.07\n"
            "2. Disbursement mechanics\n"
            "The sanctioned principal is Rs. 12,00,000. At the applicant's request, the optional credit-protection premium of Rs. 18,600 is "
            "added to the facility at first disbursement rather than collected separately. The opening amount on which instalments are "
            "scheduled will therefore be Rs. 12,18,600, subject to final booking.\n"
            "The final disbursement advice, rather than this summary, will evidence the net remittance.\n"
            "3. Charges and recoveries\n"
            "A-1 | File handling and credit assessment | 1.25% of sanctioned principal; minimum Rs. 4,000\n"
            "A-2 | Applicable indirect tax on A-1 | At prevailing rate, collected with A-1\n"
            "B-4 | Document custody and instrument processing | Rs. 3,250, one time\n"
            "C-2 | Property title search and technical panel visit | Rs. 6,800; payable to empanelled agency\n"
            "D-7 | Electronic registry filing and security-interest notation | Rs. 118 including applicable tax\n"
            "For operational convenience, a facility activation reserve of Rs. 2,950 may be debited when the loan account is opened. The reserve is\n"
            "non-interest-bearing and adjusted against account-opening and mandate-registration work; any unutilised balance, if applicable, is reconciled after the\n"
            "initial servicing cycle.\n"
        ),
        fontsize=9,
    )
    p2 = doc.new_page()
    p2.insert_text(
        (40, 40),
        (
            "Where an optional protection product is selected, the premium in Section 2 is financed.\n"
            "4. Repayment and servicing terms\n"
            "Instalments are payable by NACH/e-mandate. A mandate setup and verification cost of Rs. 590 inclusive of tax\n"
            "may be collected if the account is registered through the assisted channel. No such amount is payable where the\n"
            "self-service mandate route is accepted without manual intervention.\n"
            "If an instalment remains unpaid beyond its due date, an overdue servicing charge of Rs. 450 per event may be\n"
            "applied, in addition to applicable penal charges calculated on the overdue amount at 2.00% per annum for the\n"
            "period of default, subject to the governing agreement and applicable law. Penal charges are not capitalised.\n"
        ),
        fontsize=9,
    )
    p3 = doc.new_page()
    p3.insert_text(
        (40, 40),
        (
            "ANNEXURE I - SERVICE AND ACCOUNT NOTES\n"
            "Account opening note: The facility activation reserve referred to in Section 3 is recorded under internal\n"
            "service code FAR-2950. It is not a refundable security deposit.\n"
            "Tax note: Tax on lender-supplied services is charged at the rate applicable on the date of supply. For file\n"
            "handling fee A-1, tax is calculated separately and is not included in the percentage quoted.\n"
            "Disbursement note: The title search and technical panel visit fee may be paid directly by the applicant or\n"
            "recovered by the lender against the agency invoice.\n"
            "Premium note: The optional credit-protection premium is financed and forms part of the opening loan balance.\n"
            "Account servicing note: The assisted-channel mandate fee applies only where manual verification is requested\n"
            "or required. A failed debit may also trigger bank charges levied by the borrower's bank; these third-party\n"
            "charges are not quantified here.\n"
            "ANNEXURE II - ILLUSTRATIVE CASH FLOW RECONCILIATION\n"
            "Sanctioned principal | Rs. 12,00,000.00 | Approved facility\n"
            "Optional credit-protection premium | Rs. 18,600.00 | Added to opening financed balance\n"
            "Opening financed balance | Rs. 12,18,600.00 | Basis for indicative EMI\n"
            "File handling fee | 1.25% of sanctioned principal | Cash recovery; tax additional\n"
            "Document custody / instrument processing | Rs. 3,250.00 | One-time cash recovery\n"
            "Facility activation reserve | Rs. 2,950.00 | Account-opening debit\n"
        ),
        fontsize=9,
    )
    pdf_bytes = doc.tobytes()
    doc.close()

    pages = DocumentExtractor.extract_pages_from_pdf_bytes(pdf_bytes)
    assert len(pages) == 3

    svc = OpenRouterLLMService()
    terms = svc._deterministic_page_extract(pages)
    report = DeterministicLoanAuditor.audit_loan(terms)

    assert report.lender_name == "NORTHSTAR HOUSING FINANCE LTD"
    assert report.principal == 1200000.0
    assert report.stated_annual_rate == 9.25
    assert report.interest_type == InterestRateType.FLOATING
    assert report.tenure_months == 120
    assert report.bundled_insurance_amount == 18600.0
    assert report.is_insurance_capitalized is True
    assert report.opening_financed_balance == 1218600.0
    assert report.stated_emi == pytest.approx(15602.07, abs=0.05)
    assert report.calculated_reducing_emi == pytest.approx(15602.07, abs=0.05)
    assert report.is_emi_matching is True

    # Individual charge reconciliation: exact 5 upfront items totaling ₹28,118 before GST
    upfront = [c for c in report.charges_breakdown if c.category == ChargeCategory.CONFIRMED_UPFRONT]
    upfront_amounts = sorted(c.amount for c in upfront)
    assert upfront_amounts == [118.0, 2950.0, 3250.0, 6800.0, 15000.0]
    assert report.raw_processing_fee == pytest.approx(15000.0, abs=0.01)
    assert report.known_upfront_recoveries_excl_tax == pytest.approx(28118.0, abs=0.01)

    # Check repeated annexure disclosures deduplicated & tracked on repeated_on_pages
    reserve_item = next(c for c in upfront if "activation reserve" in c.name.lower())
    assert 3 in reserve_item.repeated_on_pages
    assert "FAR-2950" in (reserve_item.notes or "")

    title_item = next(c for c in upfront if "title search" in c.name.lower())
    assert 3 in title_item.repeated_on_pages
    assert "agency" in (title_item.notes or "").lower()

    # Conditional charges MUST NOT be summed into one misleading total; each tracked individually
    cond = [c for c in report.charges_breakdown if c.category == ChargeCategory.CONDITIONAL]
    assert sorted(c.amount for c in cond) == [450.0, 590.0]
    assert report.conditional_charges_total is None
    assert len(report.conditional_charges) == 3

    mandate_fee = next(c for c in report.conditional_charges if c.amount == 590.0)
    assert mandate_fee.frequency == "ONE_TIME"
    assert mandate_fee.caused_by_borrower_action is True
    assert "assisted" in (mandate_fee.trigger_condition or "").lower()

    overdue_fee = next(c for c in report.conditional_charges if c.amount == 450.0)
    assert overdue_fee.frequency == "PER_OVERDUE_EVENT"
    assert overdue_fee.caused_by_borrower_action is True
    assert overdue_fee.payment_timing == "per overdue event"

    # Contingent penal charge (2.00% p.a., NOT capitalized)
    contingent = [c for c in report.charges_breakdown if c.category == ChargeCategory.CONTINGENT]
    assert len(contingent) == 1
    assert contingent[0].rate_pct == pytest.approx(2.0, abs=0.01)
    assert contingent[0].is_capitalized is False
    assert contingent[0].frequency == "PER_ANNUM_ON_OVERDUE"
    assert contingent[0].caused_by_borrower_action is True
    assert report.penal_charge_rate_pct == pytest.approx(2.0, abs=0.01)
    assert report.penal_charges_capitalized is False
    assert "not capitalis" in (contingent[0].notes or "").lower()

    # GST Classification: Report-level & exact service linkage
    assert report.gst_classification == GSTClassification.GST_POTENTIALLY_APPLICABLE
    file_handling_fee = next(c for c in upfront if "file handling" in c.name.lower())
    assert file_handling_fee.gst_classification == GSTClassification.GST_POTENTIALLY_APPLICABLE
    assert "file handling" in (file_handling_fee.gst_related_service or "").lower()
    registry_fee = next(c for c in upfront if "registry" in c.name.lower())
    assert registry_fee.gst_classification == GSTClassification.GST_INCLUDED

    # Third-party recovery classification
    assert title_item.third_party_recovery_method == ThirdPartyRecoveryMethod.RECOVERED_AGAINST_AGENCY_INVOICE
    third_party = [c for c in report.charges_breakdown if c.category == ChargeCategory.THIRD_PARTY]
    assert len(third_party) == 1
    assert third_party[0].amount is None
    assert third_party[0].source_page == 3
    assert third_party[0].third_party_recovery_method == ThirdPartyRecoveryMethod.PAID_DIRECTLY_BY_BORROWER

    # Every charge must retain exact page, source_section, raw_text, confidence, and clean evidence_snippet
    for c in report.charges_breakdown:
        assert c.currency == "INR"
        assert c.payment_timing
        assert c.mandatory_status
        assert c.dedup_id
        assert c.source_page is not None
        assert c.source_section
        assert c.evidence_snippet
        assert c.raw_text
        assert c.extraction_confidence in ("HIGH", "MEDIUM", "LOW")

    # No duplicate borrower-facing discrepancies; every finding retains full provenance
    disc_keys = [(d.code, d.title.strip().lower()) for d in report.discrepancies]
    assert len(disc_keys) == len(set(disc_keys))
    for d in report.discrepancies:
        assert d.source_page is not None
        assert d.source_pages
        assert d.evidence_snippet
        assert d.extraction_confidence in ("HIGH", "MEDIUM", "LOW")

    # Borrower-facing local explanation preserves individual conditional breakdown without summing
    explanation = svc._local_template_explanation(report, "english")
    assert "1,040" not in explanation
    assert "GST_POTENTIALLY_APPLICABLE" in explanation
    assert "RECOVERED_AGAINST_AGENCY_INVOICE" in explanation

    # Operational, tax, financed insurance, third-party, mandate & failed-debit notes present
    note_topics = {n.topic for n in report.document_notes}
    assert {
        "Facility Activation Reserve",
        "Financed Insurance Explanation",
        "Tax Treatment of Processing Fee",
        "Third-Party Agency Recovery Arrangement",
        "Assisted-Channel Mandate Fee",
        "Bank Charges on Failed Debits",
    }.issubset(note_topics)

    # Unknown indirect tax & missing net disbursement remain None (UNKNOWN); APR withheld; script blocked
    assert report.fee_gst_amount is None
    assert report.net_disbursed_amount is None
    assert report.is_irr_computable is False
    assert report.effective_annual_apr is None
    assert report.rbi_compliance_score is None
    assert report.negotiation_script_blocked is True

    cv_map = {cv.check_name: cv for cv in report.cross_validations}
    assert cv_map["Principal vs Opening Financed Balance"].status == CrossValidationStatus.PASS
    assert cv_map["Insurance vs Capitalized Principal"].status == CrossValidationStatus.PASS
    assert cv_map["Stated EMI vs Independently Calculated EMI"].status == CrossValidationStatus.PASS
