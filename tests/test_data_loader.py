"""
Unit tests for data loader and dataset inspection utilities.
Verifies Phase 1 functionality: schema validation, metadata inspection, and null checks.
"""

import pytest
import pandas as pd
import numpy as np

from src.data.loader import inspect_dataset_metadata, PAYSIM_DTYPES


@pytest.fixture
def mock_transaction_df():
    """Generates a small representative batch of transaction logs."""
    return pd.DataFrame({
        "step": [1, 2, 3, 4, 5],
        "type": ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "TRANSFER"],
        "amount": [100.0, 50000.0, 2000.0, 50.0, 300000.0],
        "nameOrig": ["C001", "C002", "C003", "C004", "C005"],
        "oldbalanceOrg": [1000.0, 60000.0, 2000.0, 500.0, 300000.0],
        "newbalanceOrig": [900.0, 10000.0, 0.0, 450.0, 0.0],
        "nameDest": ["M001", "C999", "C888", "M002", "C777"],
        "oldbalanceDest": [0.0, 10000.0, 500.0, 0.0, 5000.0],
        "newbalanceDest": [0.0, 60000.0, 2500.0, 0.0, 305000.0],
        "isFraud": [0, 0, 0, 0, 1],
        "isFlaggedFraud": [0, 0, 0, 0, 1]
    })


def test_inspect_dataset_metadata(mock_transaction_df):
    """Test that inspect_dataset_metadata accurately computes counts, shapes, and class ratios."""
    meta = inspect_dataset_metadata(mock_transaction_df, target_col="isFraud")
    
    assert meta["rows"] == 5
    assert meta["columns"] == 11
    assert meta["total_missing_values"] == 0
    assert meta["target_column"] == "isFraud"
    assert 0 in meta["class_distribution"]["counts"]
    assert 1 in meta["class_distribution"]["counts"]
    
    # Verify class balance counts (4 legit, 1 fraud)
    counts = meta["class_distribution"]["counts"]
    assert counts[0] == 4
    assert counts[1] == 1
    
    # Verify percentages (80% legit, 20% fraud)
    pcts = meta["class_distribution"]["percentages"]
    assert pcts[0] == 80.0
    assert pcts[1] == 20.0


def test_paysim_dtypes_presence():
    """Ensure all 11 PaySim columns have predefined memory-optimized dtypes."""
    expected_cols = [
        "step", "type", "amount", "nameOrig", "oldbalanceOrg",
        "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest",
        "isFraud", "isFlaggedFraud"
    ]
    for col in expected_cols:
        assert col in PAYSIM_DTYPES
