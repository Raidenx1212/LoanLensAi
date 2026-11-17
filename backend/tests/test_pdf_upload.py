import os
from fastapi.testclient import TestClient
from app.main import app
from tests.generate_sample_pdf import create_sample_sanction_letter

client = TestClient(app)


def test_pdf_upload_and_audit():
    pdf_path = "tests/test_sanction_letter.pdf"
    create_sample_sanction_letter(pdf_path)

    try:
        with open(pdf_path, "rb") as f:
            response = client.post(
                "/api/v1/audit",
                files={"file": ("sanction_letter.pdf", f, "application/pdf")},
                data={"language": "en"},
            )

        assert response.status_code == 200
        data = response.json()

        # Check extracted and computed numbers
        assert data["principal"] == 500000.0
        assert data["tenure_months"] == 36
        assert data["stated_annual_rate"] == 11.5
        assert data["raw_processing_fee"] == 7500.0
        assert data["fee_gst_amount"] == round((7500.0 + 1200.0) * 0.18, 2)
        assert data["net_disbursed_amount"] < 500000.0
        # True APR must be significantly higher than stated 11.5%
        assert data["effective_annual_apr"] > 11.5
        assert len(data["discrepancies"]) > 0
        assert data["explanation"] is not None
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
