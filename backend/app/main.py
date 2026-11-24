import os
import sys
import re
import asyncio
from typing import Optional, List

# Ensure backend folder is in Python search path
current_dir = os.path.dirname(os.path.abspath(__file__))  # app
backend_dir = os.path.dirname(current_dir)                # backend
root_dir = os.path.dirname(backend_dir)                   # project root or /app
for p in [current_dir, backend_dir, root_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.config import settings
from app.schemas.loan import (
    Language,
    LoanAuditReport,
    ExtractedLoanTerms,
    LoanComparisonRequest,
    LoanComparisonResponse,
    RankedLoanItem,
)
from app.services.extractor import DocumentExtractor
from app.services.auditor import DeterministicLoanAuditor
from app.services.llm_service import OpenRouterLLMService
from app.core.database import init_db, get_db
from app.models.audit_record import LoanAuditModel

# Initialize database tables
init_db()

app = FastAPI(
    title="LoanLens API",
    description="Deterministic AI-Powered Indian Loan Auditor and Sanction Letter Verifier",
    version="1.0.0",
)

# Enable CORS for frontend applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm_service = OpenRouterLLMService()

# Robust path resolution for local dev, Render native, and Docker
possible_frontend_dirs = [
    os.path.abspath(os.path.join(current_dir, "..", "..", "frontend")),
    os.path.abspath("/app/frontend"),
    os.path.abspath(os.path.join(os.getcwd(), "frontend")),
]
FRONTEND_DIR = possible_frontend_dirs[0]
for d in possible_frontend_dirs:
    if os.path.exists(os.path.join(d, "index.html")):
        FRONTEND_DIR = d
        break

possible_backend_dirs = [
    os.path.abspath(os.path.join(current_dir, "..")),
    os.path.abspath("/app/backend"),
    os.path.abspath(os.path.join(os.getcwd(), "backend")),
]
BACKEND_DIR = possible_backend_dirs[0]
for b in possible_backend_dirs:
    if os.path.exists(b):
        BACKEND_DIR = b
        break


@app.api_route("/", methods=["GET", "HEAD"], response_class=FileResponse)
def serve_dashboard():
    """Serves the interactive LoanLens frontend dashboard directly."""
    index_file = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "LoanLens API is running. UI file index.html not found.", "checked_dir": FRONTEND_DIR}


@app.api_route("/sample_sanction_letter.pdf", methods=["GET", "HEAD"], response_class=FileResponse)
def get_sample_sanction_pdf():
    """Serves the sample Indian bank sanction letter PDF for demo/testing."""
    sample_file = os.path.join(BACKEND_DIR, "sample_sanction_letter.pdf")
    if os.path.exists(sample_file):
        return FileResponse(sample_file, media_type="application/pdf")
    raise HTTPException(status_code=404, detail="Sample PDF not found.")


@app.api_route("/health", methods=["GET", "HEAD"])
def health_check():
    return {
        "status": "healthy",
        "service": "LoanLens Deterministic Financial Engine",
        "version": "1.0.0",
    }


@app.get("/api/v1/system-status")
def system_status():
    """Returns the operational status of AI keys and extraction engines."""
    has_api_key = bool(settings.OPENROUTER_API_KEY)
    is_placeholder = "your_openrouter" in settings.OPENROUTER_API_KEY.lower() if has_api_key else False

    has_mistral_key = bool(getattr(settings, "MISTRAL_API_KEY", ""))
    is_mistral_placeholder = "your_mistral" in settings.MISTRAL_API_KEY.lower() if has_mistral_key else False

    return {
        "openrouter_configured": has_api_key and not is_placeholder,
        "mistral_configured": has_mistral_key and not is_mistral_placeholder,
        "model": getattr(settings, "DEFAULT_LLM_MODEL", "google/gemini-2.5-flash"),
        "pdf_parser": "Mistral OCR (mistral-ocr-latest) + PyMuPDF Vector Engine",
        "fallback_available": True,
    }


@app.post(
    "/api/v1/audit",
    response_model=LoanAuditReport,
    summary="Upload PDF Sanction Letter -> OCR/Extract -> LLM JSON -> Deterministic Audit -> Multilingual Explainer",
)
async def audit_loan_document(
    file: UploadFile = File(..., description="PDF Sanction Letter or Key Fact Statement (KFS)"),
    language: Language = Form(default=Language.ENGLISH, description="Target language: en, hi, or kn"),
    openrouter_api_key: Optional[str] = Form(default=None, description="Optional OpenRouter API key"),
    mistral_api_key: Optional[str] = Form(default=None, description="Optional Mistral AI API key for Mistral OCR"),
    model_name: Optional[str] = Form(default=None, description="Selected AI model (e.g. anthropic/claude-3.5-sonnet)"),
    db: Session = Depends(get_db),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF documents are supported for loan auditing.",
        )

    # 1. High-Fidelity PDF & Table extraction using Mistral OCR or PyMuPDF
    try:
        pdf_bytes = await file.read()
        raw_text = await asyncio.to_thread(
            DocumentExtractor.extract_text_from_pdf_bytes, pdf_bytes, mistral_api_key
        )
        if not raw_text.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    "Unable to extract text from the uploaded PDF. "
                    "If this is a scanned photocopy or camera photo, please provide a Mistral API key for Mistral OCR, or ensure high resolution."
                ),
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error occurred while processing PDF text: {str(e)}",
        )

    # 2. Flagship Zero-Hallucination Structured Extraction (STRICT: Zero math)
    try:
        extracted_terms = await llm_service.extract_loan_parameters(
            raw_text, custom_api_key=openrouter_api_key, model_name=model_name
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Extraction model error: {str(e)}",
        )

    # 3. Pure Deterministic Financial Audit (Python Math Engine - offloaded to thread)
    audit_result = await asyncio.to_thread(DeterministicLoanAuditor.audit_loan, extracted_terms)
    audit_result.extraction_method = extracted_terms.extraction_method

    # Multi-page extraction metadata & full multi-page preview
    pages_found = set(re.findall(r"--- Page (\d+)", raw_text))
    audit_result.total_pages_analyzed = len(pages_found) if pages_found else 1
    audit_result.raw_extracted_text_preview = raw_text[:60000]

    # 4. Multilingual Consumer Explainer
    try:
        explanation = await llm_service.generate_multilingual_explanation(
            audit=audit_result,
            language=language,
            custom_api_key=openrouter_api_key,
            model_name=model_name,
        )
        audit_result.explanation = explanation
    except Exception as e:
        audit_result.explanation = (
            f"Audit completed successfully, but narrative explanation could not be generated: {str(e)}"
        )

    # 5. Persist audit to Database
    try:
        record = LoanAuditModel(
            lender_name=audit_result.lender_name,
            principal_amount=audit_result.principal,
            tenure_months=audit_result.tenure_months,
            stated_rate=audit_result.stated_annual_rate,
            effective_annual_apr=audit_result.effective_annual_apr,
            net_disbursed_amount=audit_result.net_disbursed_amount,
            total_cost_of_borrowing=audit_result.total_cost_of_borrowing,
            rbi_compliance_score=audit_result.rbi_compliance_score,
            audit_data=audit_result.model_dump(),
            explanation=audit_result.explanation,
        )
        db.add(record)
        db.commit()
    except Exception:
        db.rollback()

    return audit_result


@app.post(
    "/api/v1/audit/manual",
    response_model=LoanAuditReport,
    summary="Direct Deterministic Audit without PDF (Direct JSON Input)",
)
async def audit_manual_loan(
    payload: ExtractedLoanTerms,
    language: Language = Language.ENGLISH,
    openrouter_api_key: Optional[str] = None,
    model_name: Optional[str] = None,
):
    """Directly audit structured loan terms without needing a PDF upload."""
    audit_result = await asyncio.to_thread(DeterministicLoanAuditor.audit_loan, payload)
    audit_result.extraction_method = payload.extraction_method
    try:
        explanation = await llm_service.generate_multilingual_explanation(
            audit=audit_result,
            language=language,
            custom_api_key=openrouter_api_key,
            model_name=model_name,
        )
        audit_result.explanation = explanation
    except Exception as e:
        audit_result.explanation = f"Explanation generation fallback: {str(e)}"
    return audit_result


@app.post(
    "/api/v1/compare",
    response_model=LoanComparisonResponse,
    summary="Compare two loan offers and rank strictly by Total Cost of Borrowing",
)
async def compare_loans(payload: LoanComparisonRequest):
    audit_a, audit_b, winner_idx = await asyncio.to_thread(
        DeterministicLoanAuditor.compare_loans, payload.loan_a, payload.loan_b
    )

    ranked_items = []
    if winner_idx == 1:
        winner, loser = audit_a, audit_b
        winner_id, loser_id = "Offer A", "Offer B"
    else:
        winner, loser = audit_b, audit_a
        winner_id, loser_id = "Offer B", "Offer A"

    cost_diff = round(loser.total_cost_of_borrowing - winner.total_cost_of_borrowing, 2)

    ranked_items.append(
        RankedLoanItem(
            rank=1,
            identifier=winner_id,
            lender_name=winner.lender_name or "Winning Lender",
            effective_annual_apr=winner.effective_annual_apr,
            total_cost_of_borrowing=winner.total_cost_of_borrowing,
            net_disbursed_amount=winner.net_disbursed_amount,
            total_repayment_amount=winner.total_repayment_amount,
            cost_difference_vs_winner=0.0,
            key_takeaway=(
                f"Saves ₹{cost_diff:,.2f} in total borrowing costs. "
                f"True APR is {winner.effective_annual_apr:.2f}% vs {loser.effective_annual_apr:.2f}%."
            ),
        )
    )

    ranked_items.append(
        RankedLoanItem(
            rank=2,
            identifier=loser_id,
            lender_name=loser.lender_name or "Second Lender",
            effective_annual_apr=loser.effective_annual_apr,
            total_cost_of_borrowing=loser.total_cost_of_borrowing,
            net_disbursed_amount=loser.net_disbursed_amount,
            total_repayment_amount=loser.total_repayment_amount,
            cost_difference_vs_winner=cost_diff,
            key_takeaway=(
                f"More expensive by ₹{cost_diff:,.2f}. Higher upfront deductions, hidden insurance, or inflated interest structure."
            ),
        )
    )

    return LoanComparisonResponse(
        recommendation=(
            f"Choose {winner_id} ({winner.lender_name or 'Offer'}). "
            f"It delivers ₹{winner.net_disbursed_amount:,.2f} in net disbursal with a total cost of ₹{winner.total_cost_of_borrowing:,.2f}, "
            f"saving you ₹{cost_diff:,.2f} over the loan duration."
        ),
        ranked_loans=ranked_items,
    )
