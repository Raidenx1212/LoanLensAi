import os
import sys
import pytest

# Ensure backend directory is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.main import health_check, system_status, mask_pii, compare_loans
from app.schemas.loan import (
    ExtractedLoanTerms,
    LoanComparisonRequest,
    InterestRateType,
    LoanCategory,
)


def test_health_endpoint():
    res = health_check()
    assert res["status"] == "healthy"
    assert "version" in res


def test_system_status_endpoint():
    res = system_status()
    assert "pdf_parser" in res
    assert "fallback_available" in res


def test_mask_pii_function():
    raw = (
        "Customer Swati Sabban, Email: swatisabban12@gmail.com, "
        "Mobile: 8608277673, PAN: ABCDE1234F, Account: 232353253525"
    )
    masked = mask_pii(raw)
    assert "swatisabban12@gmail.com" not in masked
    assert "8608277673" not in masked
    assert "ABCDE1234F" not in masked
    assert "232353253525" not in masked
    assert "***@" in masked


def test_compare_endpoint_sync():
    req = LoanComparisonRequest(
        loan_a=ExtractedLoanTerms(
            lender_name="Bank Alpha",
            principal_amount=1000000.0,
            stated_interest_rate=9.5,
            interest_type=InterestRateType.REDUCING,
            tenure_months=60,
            processing_fee=10000.0,
        ),
        loan_b=ExtractedLoanTerms(
            lender_name="Bank Beta",
            principal_amount=1000000.0,
            stated_interest_rate=11.5,
            interest_type=InterestRateType.REDUCING,
            tenure_months=60,
            processing_fee=25000.0,
        ),
    )
    res = compare_loans(req)
    assert res.recommendation is not None
    assert len(res.ranked_loans) == 2
    assert res.ranked_loans[0].identifier == "Offer A"
    assert res.ranked_loans[0].cost_difference_vs_winner == 0.0
    assert res.ranked_loans[1].cost_difference_vs_winner > 0


def test_schema_interest_type_normalization():
    terms_reducing = ExtractedLoanTerms(
        principal_amount=100000.0,
        stated_interest_rate=10.0,
        tenure_months=12,
        interest_type="Reducing Balance Rate",
    )
    assert terms_reducing.interest_type == InterestRateType.REDUCING

    terms_floating = ExtractedLoanTerms(
        principal_amount=100000.0,
        stated_interest_rate=10.0,
        tenure_months=12,
        interest_type="Floating Rate with benchmark",
    )
    assert terms_floating.interest_type == InterestRateType.FLOATING

    terms_flat = ExtractedLoanTerms(
        principal_amount=100000.0,
        stated_interest_rate=10.0,
        tenure_months=12,
        interest_type="Flat Interest 8%",
    )
    assert terms_flat.interest_type == InterestRateType.FLAT

    terms_unknown = ExtractedLoanTerms(
        principal_amount=100000.0,
        stated_interest_rate=10.0,
        tenure_months=12,
        interest_type="unspecified",
    )
    assert terms_unknown.interest_type == InterestRateType.UNKNOWN


def _run_coro_sync(coro):
    """Drives non-blocking coroutines synchronously without creating OS event-loop sockets."""
    try:
        coro.send(None)
    except StopIteration as stop:
        return stop.value
    finally:
        coro.close()
    raise RuntimeError("Coroutine yielded unexpectedly")


def test_security_headers_and_pdf_magic_bytes():
    """Verify security response headers and rejection of disguised non-PDF uploads."""
    from fastapi import HTTPException
    from app.main import add_security_headers, audit_loan_document

    async def _fake_call_next(_request):
        from starlette.responses import Response
        return Response("ok")

    resp = _run_coro_sync(add_security_headers(None, _fake_call_next))
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "DENY"
    assert resp.headers["Referrer-Policy"] == "no-referrer"

    class _FakeUploadFile:
        def __init__(self, filename: str, data: bytes):
            self.filename = filename
            self._data = data

        async def read(self) -> bytes:
            return self._data

    fake_pdf = _FakeUploadFile(filename="malicious.pdf", data=b"<script>alert(1)</script>")
    with pytest.raises(HTTPException) as exc_info:
        _run_coro_sync(audit_loan_document(file=fake_pdf))
    assert exc_info.value.status_code == 400
    assert "valid PDF" in exc_info.value.detail


def test_frontend_no_query_api_key_or_unescaped_markdown():
    """Verify frontend sends API keys via headers (not URL query string) and sanitizes HTML output."""
    index_path = os.path.join(PROJECT_ROOT, "frontend", "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        html = f.read()
    assert "?language=${currentLanguage}${apiKeyParam}" not in html
    assert "openrouter_api_key=" not in html
    assert "function escapeHtml(" in html
    assert "function sanitizeMarkdownHtml(" in html
    assert "sanitizeMarkdownHtml(marked.parse(" in html
