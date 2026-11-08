"""
Unit tests for unified production prediction pipeline.
Verifies Phase 8 functionality: Pydantic input validation, real-time single scoring,
batch CSV processing, risk score scaling, and consistent feature ordering.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from pydantic import ValidationError

from src.models.pipeline import (
    FraudPredictionPipeline,
    TransactionInput,
    TransactionType,
    RiskTier,
    PredictionOutput,
    BatchPredictionOutput,
)


@pytest.fixture
def valid_transaction_payload():
    """Returns a valid single transaction dictionary."""
    return {
        "step": 12,
        "type": "TRANSFER",
        "amount": 150000.0,
        "oldbalanceOrg": 150000.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 150000.0,
        "nameOrig": "C123456789",
        "nameDest": "C987654321",
    }


def test_transaction_input_validation(valid_transaction_payload):
    """Verifies that Pydantic enforces schema constraints and catches invalid data."""
    # 1. Valid payload should succeed
    tx = TransactionInput(**valid_transaction_payload)
    assert tx.amount == 150000.0
    assert tx.type == TransactionType.TRANSFER

    # 2. Negative amount must fail
    invalid_amount = valid_transaction_payload.copy()
    invalid_amount["amount"] = -50.0
    with pytest.raises(ValidationError):
        TransactionInput(**invalid_amount)

    # 3. Invalid transaction type must fail
    invalid_type = valid_transaction_payload.copy()
    invalid_type["type"] = "CRYPTO_BUY"
    with pytest.raises(ValidationError):
        TransactionInput(**invalid_type)


def test_predict_single_transaction(valid_transaction_payload):
    """Verifies that single prediction returns a complete, typed PredictionOutput schema."""
    pipeline = FraudPredictionPipeline.get_instance()
    result = pipeline.predict_single(valid_transaction_payload)

    assert isinstance(result, PredictionOutput)
    assert isinstance(result.is_fraud, bool)
    assert 0.0 <= result.fraud_probability <= 1.0
    assert 0.0 <= result.risk_score <= 100.0
    assert result.risk_tier in [RiskTier.LOW, RiskTier.MODERATE, RiskTier.HIGH]
    assert result.threshold_applied > 0.0
    assert result.latency_ms >= 0.0

    # Verify explainability contents
    exp = result.explanation
    assert "human_readable_summary" in exp
    assert "risk_drivers" in exp
    assert "legal_disclaimer" in exp
    assert len(exp["human_readable_summary"]) > 0


def test_predict_batch_dataframe(valid_transaction_payload):
    """Verifies batch inference processing over a DataFrame."""
    pipeline = FraudPredictionPipeline.get_instance()

    # Create batch of 5 transactions
    batch_records = [
        valid_transaction_payload,
        {
            "step": 5,
            "type": "PAYMENT",
            "amount": 25.0,
            "oldbalanceOrg": 500.0,
            "newbalanceOrig": 475.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "nameOrig": "C001",
            "nameDest": "M001",
        },
        {
            "step": 20,
            "type": "CASH_OUT",
            "amount": 350000.0,
            "oldbalanceOrg": 350000.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 1000.0,
            "newbalanceDest": 351000.0,
            "nameOrig": "C002",
            "nameDest": "C003",
        },
    ]
    batch_df = pd.DataFrame(batch_records)

    batch_output = pipeline.predict_batch(batch_df)

    assert isinstance(batch_output, BatchPredictionOutput)
    assert batch_output.total_transactions == 3
    assert len(batch_output.predictions) == 3
    assert batch_output.processing_time_sec >= 0.0
    assert 0.0 <= batch_output.fraud_rate_pct <= 100.0


def test_predict_batch_from_csv(valid_transaction_payload, tmp_path):
    """Verifies batch inference reading directly from a temporary CSV file."""
    pipeline = FraudPredictionPipeline.get_instance()

    batch_df = pd.DataFrame([valid_transaction_payload, valid_transaction_payload])
    csv_file = tmp_path / "batch_input.csv"
    batch_df.to_csv(csv_file, index=False)

    batch_output = pipeline.predict_batch(csv_file)
    assert batch_output.total_transactions == 2
    assert len(batch_output.predictions) == 2


def test_feature_order_consistency():
    """Ensures feature schema ordering in preprocessor strictly matches model metadata."""
    pipeline = FraudPredictionPipeline.get_instance()
    expected_features = pipeline.metadata.get("feature_names", [])

    if expected_features:
        assert list(pipeline.preprocessor.feature_names_out_) == expected_features
