import asyncio
from app.services.extractor import DocumentExtractor
from app.services.auditor import DeterministicLoanAuditor
from app.services.llm_service import OpenRouterLLMService
from app.schemas.loan import Language


async def test_pdf_pipeline():
    print("1. Reading sample_sanction_letter.pdf...")
    with open("sample_sanction_letter.pdf", "rb") as f:
        pdf_bytes = f.read()

    print("2. Extracting text using PyMuPDF / Tesseract fallback...")
    extracted_text = DocumentExtractor.extract_text_from_pdf_bytes(pdf_bytes)
    print("Extracted text preview:\n", extracted_text[:300], "...\n")
    assert len(extracted_text) > 100

    print("3. Extracting structured parameters via LLM service (Strict Zero Math)...")
    llm = OpenRouterLLMService()
    loan_terms = await llm.extract_loan_parameters(extracted_text)
    print("Extracted Loan Terms:")
    print("  Lender:", loan_terms.lender_name)
    print("  Principal: ₹", loan_terms.principal_amount)
    print("  Stated ROI:", loan_terms.stated_interest_rate, "%")
    print("  Tenure:", loan_terms.tenure_months, "months")
    print("  Stated EMI: ₹", loan_terms.stated_emi)
    print("  Processing Fee: ₹", loan_terms.processing_fee)
    print("  Bundled Insurance: ₹", loan_terms.bundled_insurance)
    print("  Has Foreclosure Penalty:", loan_terms.has_prepayment_penalty)

    print("\n4. Running Deterministic Python Financial Audit Engine...")
    audit = DeterministicLoanAuditor.audit_loan(loan_terms)
    print("Audit Results:")
    print("  Calculated Reducing EMI: ₹", audit.calculated_reducing_emi)
    print("  18% GST on Fees: ₹", audit.fee_gst_amount)
    print("  Total Upfront Deductions (Fees+GST): ₹", audit.total_upfront_fees_incl_gst)
    print("  Net Disbursed Amount: ₹", audit.net_disbursed_amount)
    print("  Nominal APR: ", audit.nominal_apr, "%")
    print("  Effective Annual APR (Compounded): ", audit.effective_annual_apr, "%")
    print("  True Cost of Borrowing: ₹", audit.total_cost_of_borrowing)
    print("  RBI Fair Lending Compliance Score:", audit.rbi_compliance_score, "/ 100")
    print("\nDetected Discrepancies:")
    for d in audit.discrepancies:
        print(f"  * [{d.severity.upper()}] {d.title}: {d.description}")

    print("\n5. Testing Multilingual Explanations...")
    for lang in [Language.ENGLISH, Language.HINDI, Language.KANNADA]:
        explanation = await llm.generate_multilingual_explanation(audit, lang)
        print(f"\n--- Explanation [{lang.value.upper()}] ---")
        print(explanation[:250], "...\n")

    print("\n>>> FULL PDF EXTRACTION & AUDIT PIPELINE VERIFIED SUCCESSFULLY! <<<")


if __name__ == "__main__":
    asyncio.run(test_pdf_pipeline())
