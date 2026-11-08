"""
Phase 11: End-to-End Quality Assurance, Edge Case & Contract Verification Suite.
Validates:
1. Frontend-Backend Schema Contract Consistency
2. Deterministic Inference & Model Idempotency
3. Boundary & Extreme Edge Cases (zero amounts, whale transactions, step limits)
4. Data Leakage Guards & Feature Vector Isolation
5. Batch Order Preservation & Throughput Stability
"""

import io
import time
import pytest
from fastapi.testclient import TestClient
from api.main import app
from src.models.pipeline import FraudPredictionPipeline, TransactionInput, RiskTier


@pytest.fixture(scope="module")
def client():
    """Provides a TestClient context with the FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# 1. Frontend-Backend Schema Contract Tests
# ---------------------------------------------------------------------------
def test_contract_prediction_result_schema(client):
    """
    Verifies that the /predict response schema exactly mirrors the TypeScript
    PredictionResult interface declared in frontend/src/lib/api.ts.
    """
    payload = {
        "step": 30,
        "type": "TRANSFER",
        "amount": 250000.0,
        "oldbalanceOrg": 250000.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 250000.0,
        "nameOrig": "C123456",
        "nameDest": "C654321",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Required TypeScript PredictionResult fields
    expected_fields = [
        "transaction_id",
        "is_fraud",
        "fraud_probability",
        "risk_score",
        "risk_tier",
        "recommendation",
        "threshold_applied",
        "explanation",
        "latency_ms",
    ]
    for field in expected_fields:
        assert field in data, f"Contract mismatch: Missing field '{field}' in PredictionResult"

    # Verify types
    assert isinstance(data["transaction_id"], str)
    assert isinstance(data["is_fraud"], bool)
    assert isinstance(data["fraud_probability"], float)
    assert isinstance(data["risk_score"], (int, float))
    assert data["risk_tier"] in ["LOW", "MODERATE", "HIGH"]
    assert isinstance(data["recommendation"], str)
    assert isinstance(data["threshold_applied"], float)
    assert isinstance(data["latency_ms"], (int, float))

    # Required TypeScript ExplanationData fields
    expl = data["explanation"]
    assert "fraud_probability" in expl
    assert "risk_tier" in expl
    assert "base_value" in expl
    assert "risk_drivers" in expl
    assert "mitigating_factors" in expl
    assert "human_readable_summary" in expl
    assert "legal_disclaimer" in expl


def test_contract_health_schema(client):
    """Verifies that /health matches TypeScript SystemHealth interface."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    for field in ["status", "model_loaded", "model_type", "optimal_threshold", "timestamp", "version"]:
        assert field in data, f"Contract mismatch: Missing field '{field}' in SystemHealth"


def test_contract_batch_schema(client):
    """Verifies that /predict/batch matches TypeScript BatchPredictionResult interface."""
    payload = {
        "transactions": [
            {
                "step": 1,
                "type": "PAYMENT",
                "amount": 50.0,
                "oldbalanceOrg": 200.0,
                "newbalanceOrig": 150.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
            }
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    for field in [
        "total_transactions",
        "fraud_detected_count",
        "fraud_rate_pct",
        "high_risk_count",
        "moderate_risk_count",
        "low_risk_count",
        "processing_time_sec",
        "predictions",
    ]:
        assert field in data, f"Contract mismatch: Missing field '{field}' in BatchPredictionResult"


# ---------------------------------------------------------------------------
# 2. Determinism & Idempotency Tests
# ---------------------------------------------------------------------------
def test_inference_deterministic_reproducibility(client):
    """
    Ensures that identical inputs produce bitwise identical outputs across repeated calls,
    preventing stochastic drift during production inference.
    """
    payload = {
        "step": 15,
        "type": "CASH_OUT",
        "amount": 180000.0,
        "oldbalanceOrg": 180000.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 50000.0,
        "newbalanceDest": 230000.0,
        "nameOrig": "C_TEST_DETERMINISTIC",
        "nameDest": "C_DEST_DETERMINISTIC",
    }

    res1 = client.post("/predict", json=payload).json()
    res2 = client.post("/predict", json=payload).json()

    assert res1["is_fraud"] == res2["is_fraud"]
    assert res1["fraud_probability"] == res2["fraud_probability"]
    assert res1["risk_score"] == res2["risk_score"]
    assert res1["risk_tier"] == res2["risk_tier"]
    assert res1["recommendation"] == res2["recommendation"]
    assert res1["threshold_applied"] == res2["threshold_applied"]


# ---------------------------------------------------------------------------
# 3. Boundary & Extreme Edge Cases
# ---------------------------------------------------------------------------
def test_edge_case_zero_dollar_transaction(client):
    """Tests system handles a legitimate zero-dollar verification ping gracefully."""
    payload = {
        "step": 1,
        "type": "PAYMENT",
        "amount": 0.0,
        "oldbalanceOrg": 500.0,
        "newbalanceOrig": 500.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["risk_score"] <= 100.0


def test_edge_case_extreme_whale_transaction(client):
    """Tests system handles multi-million-dollar transactions without numerical overflow."""
    payload = {
        "step": 100,
        "type": "TRANSFER",
        "amount": 250000000.0,  # $250 Million
        "oldbalanceOrg": 250000000.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_fraud"] is True
    assert data["risk_tier"] in ["MODERATE", "HIGH"]
    assert 0.0 <= data["risk_score"] <= 100.0


def test_edge_case_step_boundaries(client):
    """Tests simulation step limits: step 1 and step 744 (1 month)."""
    base_payload = {
        "type": "PAYMENT",
        "amount": 10.0,
        "oldbalanceOrg": 100.0,
        "newbalanceOrig": 90.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }

    # Step 1 (minimum)
    res_min = client.post("/predict", json={**base_payload, "step": 1})
    assert res_min.status_code == 200

    # Step 744 (end of 31-day simulation)
    res_max = client.post("/predict", json={**base_payload, "step": 744})
    assert res_max.status_code == 200

    # Step 0 (invalid step, below minimum of 1)
    res_invalid = client.post("/predict", json={**base_payload, "step": 0})
    assert res_invalid.status_code == 422


def test_edge_case_optional_identifiers_omitted(client):
    """Tests inference succeeds when nameOrig and nameDest are omitted or None."""
    payload = {
        "step": 5,
        "type": "CASH_IN",
        "amount": 1000.0,
        "oldbalanceOrg": 5000.0,
        "newbalanceOrig": 6000.0,
        "oldbalanceDest": 2000.0,
        "newbalanceDest": 1000.0,
        # nameOrig and nameDest omitted
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] is not None


# ---------------------------------------------------------------------------
# 4. Data Leakage & Feature Vector Guards
# ---------------------------------------------------------------------------
def test_data_leakage_guards():
    """
    Verifies that the preprocessor feature names out strictly exclude:
    - 'isFraud' (target variable)
    - 'isFlaggedFraud' (heuristic leakage flag)
    - 'nameOrig' (arbitrary customer identifier)
    - 'nameDest' (arbitrary recipient identifier)
    """
    pipeline = FraudPredictionPipeline.get_instance()
    feature_names = pipeline.preprocessor.feature_names_out_

    forbidden_cols = ["isFraud", "isFlaggedFraud", "nameOrig", "nameDest"]
    for col in forbidden_cols:
        assert col not in feature_names, f"CRITICAL LEAKAGE DETECTED: '{col}' found in model feature set!"


# ---------------------------------------------------------------------------
# 5. Batch Order Preservation & Throughput
# ---------------------------------------------------------------------------
def test_batch_order_and_id_preservation(client):
    """
    Verifies that a batch of 10 mixed transactions maintains exact 1-to-1 input-to-output
    ordering and correctly preserves individual transaction IDs.
    """
    tx_list = []
    for i in range(10):
        tx_type = "TRANSFER" if i % 2 == 0 else "PAYMENT"
        tx_list.append({
            "step": i + 1,
            "type": tx_type,
            "amount": float(100 * (i + 1)),
            "oldbalanceOrg": 1000.0,
            "newbalanceOrig": 1000.0 - float(100 * (i + 1)),
            "oldbalanceDest": 0.0,
            "newbalanceDest": float(100 * (i + 1)) if tx_type == "TRANSFER" else 0.0,
            "nameOrig": f"C_ORDER_TEST_{i:03d}",
            "nameDest": f"M_ORDER_DEST_{i:03d}",
        })

    response = client.post("/predict/batch", json={"transactions": tx_list})
    assert response.status_code == 200
    data = response.json()

    assert data["total_transactions"] == 10
    assert len(data["predictions"]) == 10

    # Verify 1-to-1 order preservation
    for i in range(10):
        expected_id = f"C_ORDER_TEST_{i:03d}"
        assert data["predictions"][i]["transaction_id"] == expected_id, (
            f"Batch order corrupted at index {i}: expected {expected_id}, got {data['predictions'][i]['transaction_id']}"
        )

    # Verify risk breakdown sum equals total
    sum_tiers = data["high_risk_count"] + data["moderate_risk_count"] + data["low_risk_count"]
    assert sum_tiers == 10


def test_consecutive_scoring_throughput(client):
    """
    Stress tests rapid consecutive single transactions to ensure latency stability
    and absence of memory or resource accumulation.
    """
    payload = {
        "step": 20,
        "type": "PAYMENT",
        "amount": 42.0,
        "oldbalanceOrg": 500.0,
        "newbalanceOrig": 458.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }

    latencies = []
    for _ in range(15):
        t0 = time.perf_counter()
        res = client.post("/predict", json=payload)
        latencies.append((time.perf_counter() - t0) * 1000)
        assert res.status_code == 200

    avg_latency = sum(latencies) / len(latencies)
    # Average TestClient latency including TreeSHAP calculation (100 trees) should remain bounded (< 300ms)
    assert avg_latency < 300.0, f"Average latency too high: {avg_latency:.2f}ms"
