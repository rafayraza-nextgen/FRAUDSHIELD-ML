"""
Unit tests for data preprocessing and leakage prevention pipeline.
Verifies Phase 2 functionality: feature engineering, scaling, splitting, and serialization.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data.preprocessor import FraudDataPreprocessor, split_data


@pytest.fixture
def sample_dataset():
    """Generates a representative synthetic transaction DataFrame."""
    np.random.seed(42)
    n = 100
    types = np.random.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"], size=n)
    amounts = np.random.uniform(10, 500000, size=n)
    old_orig = np.random.uniform(0, 1000000, size=n)
    new_orig = np.maximum(0, old_orig - amounts)
    old_dest = np.random.uniform(0, 500000, size=n)
    new_dest = old_dest + amounts
    
    # 5% fraud rate
    is_fraud = np.zeros(n, dtype=int)
    is_fraud[:5] = 1
    
    return pd.DataFrame({
        "step": np.sort(np.random.randint(1, 100, size=n)),
        "type": types,
        "amount": amounts,
        "nameOrig": [f"C{i:07d}" for i in range(n)],
        "oldbalanceOrg": old_orig,
        "newbalanceOrig": new_orig,
        "nameDest": [f"M{i:07d}" if t == "PAYMENT" else f"C{i+1000:07d}" for i, t in enumerate(types)],
        "oldbalanceDest": old_dest,
        "newbalanceDest": new_dest,
        "isFraud": is_fraud,
        "isFlaggedFraud": np.zeros(n, dtype=int),
    })


def test_stratified_split(sample_dataset):
    """Test that stratified split preserves exact class distribution in train and test splits."""
    X_train, X_test, y_train, y_test = split_data(
        sample_dataset, target_col="isFraud", test_size=0.2, split_strategy="stratified", random_state=42
    )

    assert len(X_train) == 80
    assert len(X_test) == 20
    assert "isFraud" not in X_train.columns
    assert "isFraud" not in X_test.columns

    # Verify both sets contain fraud cases proportional to test size
    assert y_train.sum() == 4
    assert y_test.sum() == 1


def test_time_based_split(sample_dataset):
    """Test that time-based split prevents future transaction leakage into training."""
    X_train, X_test, y_train, y_test = split_data(
        sample_dataset, target_col="isFraud", test_size=0.2, split_strategy="time_based"
    )

    assert len(X_train) == 80
    assert len(X_test) == 20
    assert X_train["step"].max() <= X_test["step"].min()


def test_feature_engineering_and_leakage_dropping(sample_dataset):
    """Verify engineered balance error columns and removal of IDs and heuristic flags."""
    preprocessor = FraudDataPreprocessor()
    X = sample_dataset.drop(columns=["isFraud"])
    transformed = preprocessor.fit_transform(X)

    # Identifiers and leakage columns dropped
    assert "nameOrig" not in transformed.columns
    assert "nameDest" not in transformed.columns
    assert "isFlaggedFraud" not in transformed.columns

    # Engineered features present
    assert "errorBalanceOrig" in transformed.columns
    assert "errorBalanceDest" in transformed.columns
    assert "isDestMerchant" in transformed.columns


def test_single_dict_inference_consistency(sample_dataset):
    """Ensure transforming a single dictionary matches DataFrame transformation ordering."""
    preprocessor = FraudDataPreprocessor()
    X = sample_dataset.drop(columns=["isFraud"])
    X_proc = preprocessor.fit_transform(X)

    single_tx = {
        "step": 12,
        "type": "TRANSFER",
        "amount": 250000.0,
        "nameOrig": "C1234567",
        "oldbalanceOrg": 300000.0,
        "newbalanceOrig": 50000.0,
        "nameDest": "C9999999",
        "oldbalanceDest": 10000.0,
        "newbalanceDest": 260000.0,
        "isFlaggedFraud": 0,
    }

    single_proc = preprocessor.transform(single_tx)

    # Check identical column length and names
    assert list(single_proc.columns) == list(X_proc.columns)
    assert single_proc.shape[0] == 1


def test_preprocessor_serialization(sample_dataset, tmp_path):
    """Verify preprocessor can be saved and reloaded with identical transform outputs."""
    preprocessor = FraudDataPreprocessor()
    X = sample_dataset.drop(columns=["isFraud"])
    X_train = preprocessor.fit_transform(X.iloc[:80])

    model_path = tmp_path / "preprocessor.joblib"
    preprocessor.save(model_path)
    assert model_path.exists()

    loaded_preprocessor = FraudDataPreprocessor.load(model_path)
    X_test_orig = preprocessor.transform(X.iloc[80:])
    X_test_loaded = loaded_preprocessor.transform(X.iloc[80:])

    pd.testing.assert_frame_equal(X_test_orig, X_test_loaded)
