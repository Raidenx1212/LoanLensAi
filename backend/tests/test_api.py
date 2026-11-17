import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_manual_audit_endpoint():
    payload = {
        "lender_name": "Axis Bank",
        "loan_category": "personal",
        "principal_amount": 300000.0,
        "stated_interest_rate": 11.5,
        "interest_type": "reducing",
        "tenure_months": 24,
        "processing_fee": 3000.0,
        "documentation_charges": 500.0,
        "bundled_insurance": 4500.0,
        "is_insurance_capitalized": False,
        "has_prepayment_penalty": False,
    }
    response = client.post("/api/v1/audit/manual", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["principal"] == 300000.0
    assert data["fee_gst_amount"] == 630.0  # 18% on (3000 + 500)
    assert data["net_disbursed_amount"] == 300000.0 - 4130.0 - 4500.0
    assert data["effective_annual_apr"] > 11.5
    assert data["explanation"] is not None


def test_compare_endpoint():
    payload = {
        "loan_a": {
            "lender_name": "SBI",
            "principal_amount": 500000.0,
            "stated_interest_rate": 9.5,
            "interest_type": "reducing",
            "tenure_months": 36,
            "processing_fee": 1000.0,
        },
        "loan_b": {
            "lender_name": "NBFC QuickLoan",
            "principal_amount": 500000.0,
            "stated_interest_rate": 14.0,
            "interest_type": "reducing",
            "tenure_months": 36,
            "processing_fee": 12000.0,
        },
    }
    response = client.post("/api/v1/compare", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["ranked_loans"]) == 2
    assert data["ranked_loans"][0]["lender_name"] == "SBI"
    assert data["ranked_loans"][0]["rank"] == 1
