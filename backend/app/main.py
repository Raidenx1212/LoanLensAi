import os
import sys
import re
import logging
import asyncio
from typing import Optional, List

# Ensure backend folder is in Python search path
current_dir = os.path.dirname(os.path.abspath(__file__))  # app
backend_dir = os.path.dirname(current_dir)                # backend
root_dir = os.path.dirname(backend_dir)                   # project root or /app
for p in [current_dir, backend_dir, root_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException, Request, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

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
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    return response

def mask_pii(text: str) -> str:
    """Masks sensitive borrower PII (PAN, phone, bank account, email) before persistence."""
    if not text:
        return text
    # Mask Email
    text = re.sub(r"\b([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b", r"\1***@\2", text)
    # Mask Indian Phone (10 digits with optional prefix)
    text = re.sub(r"(?:\+?91|0)?\s*([6-9]\d{2})\d{4}(\d{3})\b", r"+91 \1****\2", text)
    # Mask PAN (5 uppercase letters, 4 digits, 1 uppercase letter)
    text = re.sub(r"\b([A-Z]{3})[A-Z]{2}\d{4}([A-Z])\b", r"\1****\2", text)
    # Mask Account Numbers (9 to 18 consecutive digits)
    text = re.sub(r"\b(\d{2,4})\d{5,12}(\d{3})\b", r"\1******\2", text)
    return text

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
    return {"message": "LoanLens API is running. Frontend UI unavailable."}


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
    model_name: Optional[str] = Form(default=None, description="Selected AI model (e.g. google/gemini-2.5-flash)"),
    x_openrouter_key: Optional[str] = Header(default=None, alias="X-OpenRouter-Key"),
    x_mistral_key: Optional[str] = Header(default=None, alias="X-Mistral-Key"),
    db: Session = Depends(get_db),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF documents are supported for loan auditing.",
        )

    # Secure API key extraction: prefer custom request headers over body
    active_openrouter_key = x_openrouter_key or openrouter_api_key
    active_mistral_key = x_mistral_key or mistral_api_key

    # 1. High-Fidelity PDF & Table extraction using Mistral OCR or PyMuPDF
    try:
        pdf_bytes = await file.read()
        MAX_PDF_SIZE = 25 * 1024 * 1024  # 25 MB limit
        if len(pdf_bytes) > MAX_PDF_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Uploaded PDF file size exceeds the 25 MB limit.",
            )
        if not pdf_bytes.lstrip().startswith(b"%PDF"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is not a valid PDF binary.",
            )

        pages = DocumentExtractor.extract_pages_from_pdf_bytes(pdf_bytes, active_mistral_key)
        raw_text = "\n\n".join(
            p["text"] if p["text"].startswith("--- Page") else f"--- Page {p['page']} ---\n{p['text']}"
            for p in pages
        ).strip()
        if not raw_text:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    "Unable to extract text from the uploaded PDF. "
                    "If this is a scanned photocopy or camera photo, please provide a Mistral API key for Mistral OCR, or ensure high resolution."
                ),
            )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Error occurred while processing uploaded PDF")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error occurred while processing the uploaded PDF document.",
        )

    # 2. Flagship Zero-Hallucination Structured Extraction (STRICT: Zero math, Zero False Zeros)
    try:
        extracted_terms = await llm_service.extract_loan_parameters(
            raw_text, custom_api_key=active_openrouter_key, model_name=model_name, pages=pages
        )
    except Exception:
        logger.exception("Extraction pipeline failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Extraction engine failed to process the document.",
        )

    # 3. Pure Deterministic Financial Audit (Python Math Engine)
    audit_result = DeterministicLoanAuditor.audit_loan(extracted_terms)
    audit_result.extraction_method = extracted_terms.extraction_method

    # Multi-page extraction metadata & full multi-page preview
    audit_result.total_pages_analyzed = len(pages) if pages else 1
    sanitized_preview = mask_pii(raw_text[:60000])
    audit_result.raw_extracted_text_preview = sanitized_preview

    # 4. Multilingual Consumer Explainer
    try:
        explanation = await llm_service.generate_multilingual_explanation(
            audit=audit_result,
            language=language,
            custom_api_key=active_openrouter_key,
            model_name=model_name,
        )
        audit_result.explanation = explanation
    except Exception:
        logger.exception("Explanation generation failed; using deterministic template fallback")
        audit_result.explanation = llm_service._local_template_explanation(audit_result, language)

    # 5. Persist sanitized audit to Database (PII masked)
    try:
        sanitized_dump = audit_result.model_dump()
        if sanitized_dump.get("raw_extracted_text_preview"):
            sanitized_dump["raw_extracted_text_preview"] = sanitized_preview
        sanitized_explanation = mask_pii(audit_result.explanation) if audit_result.explanation else None
        sanitized_dump["explanation"] = sanitized_explanation

        record = LoanAuditModel(
            lender_name=audit_result.lender_name,
            principal_amount=audit_result.principal,
            tenure_months=audit_result.tenure_months,
            stated_rate=audit_result.stated_annual_rate,
            effective_annual_apr=audit_result.effective_annual_apr,
            net_disbursed_amount=audit_result.net_disbursed_amount,
            total_cost_of_borrowing=audit_result.total_cost_of_borrowing,
            rbi_compliance_score=audit_result.rbi_compliance_score,
            audit_data=sanitized_dump,
            explanation=sanitized_explanation,
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
    x_openrouter_key: Optional[str] = Header(default=None, alias="X-OpenRouter-Key"),
    model_name: Optional[str] = None,
):
    """Directly audit structured loan terms without needing a PDF upload."""
    audit_result = DeterministicLoanAuditor.audit_loan(payload)
    audit_result.extraction_method = payload.extraction_method
    try:
        explanation = await llm_service.generate_multilingual_explanation(
            audit=audit_result,
            language=language,
            custom_api_key=x_openrouter_key,
            model_name=model_name,
        )
        audit_result.explanation = explanation
    except Exception:
        logger.exception("Manual audit explanation generation failed; using deterministic template fallback")
        audit_result.explanation = llm_service._local_template_explanation(audit_result, language)
    return audit_result


@app.post(
    "/api/v1/compare",
    response_model=LoanComparisonResponse,
    summary="Compare two loan offers and rank strictly by Total Cost of Borrowing",
)
def compare_loans(payload: LoanComparisonRequest):
    audit_a, audit_b, winner_idx = DeterministicLoanAuditor.compare_loans(payload.loan_a, payload.loan_b)

    ranked_items = []
    if winner_idx == 1:
        winner, loser = audit_a, audit_b
        winner_id, loser_id = "Offer A", "Offer B"
    else:
        winner, loser = audit_b, audit_a
        winner_id, loser_id = "Offer B", "Offer A"

    cost_diff: Optional[float] = None
    if loser.total_cost_of_borrowing is not None and winner.total_cost_of_borrowing is not None:
        cost_diff = round(loser.total_cost_of_borrowing - winner.total_cost_of_borrowing, 2)
    elif loser.total_repayment_amount is not None and winner.total_repayment_amount is not None:
        cost_diff = round(loser.total_repayment_amount - winner.total_repayment_amount, 2)

    winner_apr_str = f"{winner.effective_annual_apr:.2f}%" if winner.effective_annual_apr is not None else "UNKNOWN"
    loser_apr_str = f"{loser.effective_annual_apr:.2f}%" if loser.effective_annual_apr is not None else "UNKNOWN"
    savings_str = f"₹{cost_diff:,.2f}" if cost_diff is not None else "UNKNOWN"
    winner_net_str = f"₹{winner.net_disbursed_amount:,.2f}" if winner.net_disbursed_amount is not None else "UNKNOWN"
    winner_cost_str = f"₹{winner.total_cost_of_borrowing:,.2f}" if winner.total_cost_of_borrowing is not None else "UNKNOWN"

    ranked_items.append(
        RankedLoanItem(
            rank=1,
            identifier=winner_id,
            lender_name=winner.lender_name or "Winning Lender",
            effective_annual_apr=winner.effective_annual_apr,
            total_cost_of_borrowing=winner.total_cost_of_borrowing,
            net_disbursed_amount=winner.net_disbursed_amount,
            total_repayment_amount=winner.total_repayment_amount,
            cost_difference_vs_winner=0.0 if cost_diff is not None else None,
            key_takeaway=(
                f"Saves {savings_str} in total borrowing costs. "
                f"True APR is {winner_apr_str} vs {loser_apr_str}."
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
                f"More expensive by {savings_str}. Higher upfront deductions, hidden insurance, or inflated interest structure."
            ),
        )
    )

    return LoanComparisonResponse(
        recommendation=(
            f"Choose {winner_id} ({winner.lender_name or 'Offer'}). "
            f"Net disbursal: {winner_net_str}; total borrowing cost: {winner_cost_str}; "
            f"estimated savings: {savings_str}."
        ),
        ranked_loans=ranked_items,
    )
