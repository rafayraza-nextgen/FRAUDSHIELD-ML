"""
Comprehensive integration and unit test suite for F.R.A.U.D.S.H.I.E.L.D. FastAPI service.
Tests all endpoints, error handling, validation, batch processing, and CORS headers.
"""

import io
import pytest
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture(scope="module")
def client():
    """Provides a TestClient context with the FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    """Verifies the root endpoint returns 200 and system metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["system"] == "F.R.A.U.D.S.H.I.E.L.D."
    assert data["status"] == "online"
    assert "/docs" in data["docs_url"]


def test_health_check_endpoint(client):
    """Verifies /health reports model loading status and threshold."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "RandomForest" in data["model_type"]
    assert 0.0 < data["optimal_threshold"] < 1.0


def test_models_info_endpoint(client):
    """Verifies /models/info returns feature schema and training metadata."""
    response = client.get("/models/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert isinstance(data["feature_names"], list)
    assert len(data["feature_names"]) > 0
    assert "errorBalanceOrig" in data["feature_names"]


def test_predict_single_legitimate_transaction(client):
    """Tests /predict with a normal low-risk transaction."""
    payload = {
        "step": 1,
        "type": "TRANSFER",
        "amount": 100.0,
        "oldbalanceOrg": 500.0,
        "newbalanceOrig": 400.0,
        "oldbalanceDest": 1000.0,
        "newbalanceDest": 1100.0,
        "nameOrig": "C10001",
        "nameDest": "C20001",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_fraud"] is False
    assert data["risk_tier"] == "LOW"
    assert data["risk_score"] < 50.0
    assert data["recommendation"] == "AUTO_APPROVE"
    assert "explanation" in data
    assert data["latency_ms"] >= 0.0


def test_predict_single_fraudulent_transaction(client):
    """Tests /predict with an account draining TRANSFER transaction."""
    payload = {
        "step": 50,
        "type": "TRANSFER",
        "amount": 950000.0,
        "oldbalanceOrg": 950000.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "nameOrig": "C99999",
        "nameDest": "C88888",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_fraud"] is True
    assert data["risk_tier"] in ["MODERATE", "HIGH"]
    assert data["risk_score"] >= 50.0
    assert "CHALLENGE" in data["recommendation"] or "DECLINE" in data["recommendation"]


def test_predict_validation_errors(client):
    """Tests that malformed payloads trigger 422 validation errors."""
    # Negative amount
    invalid_payload = {
        "step": 1,
        "type": "PAYMENT",
        "amount": -500.0,
        "oldbalanceOrg": 100.0,
        "newbalanceOrig": 50.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422

    # Invalid transaction type
    invalid_type_payload = {
        "step": 1,
        "type": "BITCOIN_WIRE",
        "amount": 500.0,
        "oldbalanceOrg": 100.0,
        "newbalanceOrig": 50.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }
    response = client.post("/predict", json=invalid_type_payload)
    assert response.status_code == 422


def test_predict_batch_json(client):
    """Tests /predict/batch with a JSON array of transactions."""
    batch_payload = {
        "transactions": [
            {
                "step": 10,
                "type": "PAYMENT",
                "amount": 25.0,
                "oldbalanceOrg": 500.0,
                "newbalanceOrig": 475.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
            },
            {
                "step": 45,
                "type": "TRANSFER",
                "amount": 500000.0,
                "oldbalanceOrg": 500000.0,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
            },
        ]
    }
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_transactions"] == 2
    assert len(data["predictions"]) == 2
    assert data["high_risk_count"] + data["moderate_risk_count"] + data["low_risk_count"] == 2


def test_predict_batch_csv_upload(client):
    """Tests /predict/batch/csv by uploading in-memory CSV file."""
    csv_content = (
        "step,type,amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest\n"
        "1,PAYMENT,150.0,2000.0,1850.0,0.0,0.0\n"
        "2,CASH_OUT,300000.0,300000.0,0.0,500.0,300500.0\n"
    )
    files = {"file": ("test_batch.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    response = client.post("/predict/batch/csv", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["total_transactions"] == 2
    assert "predictions" in data


def test_predict_batch_csv_invalid_file(client):
    """Tests /predict/batch/csv rejects non-csv files."""
    files = {"file": ("test_batch.txt", io.BytesIO(b"invalid data"), "text/plain")}
    response = client.post("/predict/batch/csv", files=files)
    assert response.status_code == 400


def test_explain_endpoint(client):
    """Tests /explain endpoint returns granular SHAP attribution."""
    payload = {
        "step": 20,
        "type": "TRANSFER",
        "amount": 800000.0,
        "oldbalanceOrg": 800000.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "nameOrig": "C12345",
        "nameDest": "C67890",
    }
    response = client.post("/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "top_contributing_features" in data
    assert len(data["top_contributing_features"]) > 0
    assert "explanation_narrative" in data
    assert "errorBalanceOrig" in data["explanation_narrative"] or len(data["top_contributing_features"]) > 0


def test_cors_headers(client):
    """Tests that CORS preflight headers are returned."""
    headers = {
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "POST",
    }
    response = client.options("/predict", headers=headers)
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") in ["*", "http://localhost:3000"]
