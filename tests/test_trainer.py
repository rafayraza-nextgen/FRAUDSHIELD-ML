"""
Unit tests for baseline model trainer.
Verifies Phase 4 functionality: algorithm instantiation, training timing, probability outputs, and serialization.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.models.trainer import BaselineModelTrainer, get_baseline_models


@pytest.fixture
def dummy_train_data():
    """Generates synthetic processed numerical training dataset."""
    np.random.seed(42)
    n = 150
    X = np.random.randn(n, 8)
    # Generate binary target with ~10% positive rate
    y = (X[:, 0] + X[:, 1] * 1.5 > 1.8).astype(int)
    # Ensure both classes exist
    if y.sum() == 0:
        y[0] = 1
    return X, y


def test_baseline_models_catalog():
    """Verifies all 6 required algorithms are defined in the catalog."""
    catalog = get_baseline_models()
    expected_models = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Naive Bayes",
        "Support Vector Machine",
        "K-Nearest Neighbors",
    ]
    assert len(catalog) == 6
    for name in expected_models:
        assert name in catalog


def test_train_and_predict_single_model(dummy_train_data):
    """Verifies training, prediction, probability calculation, and timing for a single baseline."""
    X, y = dummy_train_data
    trainer = BaselineModelTrainer()

    # Train Decision Tree
    model = trainer.train_model("Decision Tree", X, y)
    assert "Decision Tree" in trainer.trained_models
    assert trainer.training_times["Decision Tree"] >= 0.0

    # Inference check
    y_pred, y_prob, pred_time = trainer.predict("Decision Tree", X[:10])
    assert len(y_pred) == 10
    assert len(y_prob) == 10
    assert pred_time >= 0.0
    assert np.all((y_prob >= 0.0) & (y_prob <= 1.0))


def test_train_all_baselines(dummy_train_data):
    """Verifies all 6 baseline models can be trained without error on sample data."""
    X, y = dummy_train_data
    trainer = BaselineModelTrainer()
    times = trainer.train_all_baselines(X, y)

    assert len(trainer.trained_models) == 6
    for name in get_baseline_models().keys():
        assert name in times
        assert times[name] >= 0.0
        y_pred, y_prob, _ = trainer.predict(name, X[:5])
        assert len(y_pred) == 5
        assert np.all((y_prob >= 0.0) & (y_prob <= 1.0))


def test_save_and_load_model(dummy_train_data, tmp_path):
    """Verifies model serialization and deserialization with matching predictions."""
    X, y = dummy_train_data
    trainer = BaselineModelTrainer()
    trainer.train_model("Logistic Regression", X, y)

    save_path = tmp_path / "lr_test.joblib"
    trainer.save_model("Logistic Regression", filepath=save_path)
    assert save_path.exists()

    loaded_model = BaselineModelTrainer.load_model(save_path)
    original_pred = trainer.trained_models["Logistic Regression"].predict(X[:10])
    loaded_pred = loaded_model.predict(X[:10])

    np.testing.assert_array_equal(original_pred, loaded_pred)
