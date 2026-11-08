"""
Unit tests for model optimizer, class imbalance handling, and threshold calibration.
Verifies Phase 6 functionality: resampling logic, threshold sweeps, plotting, and champion persistence.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from src.models.optimizer import (
    apply_resampling,
    find_optimal_threshold,
    plot_threshold_tuning_curve,
    plot_imbalance_strategies_comparison,
    train_optimized_champion,
    save_champion_model,
    load_champion_model,
)


@pytest.fixture
def synthetic_imbalanced_data():
    """Generates synthetic dataset with severe 5% class imbalance."""
    np.random.seed(42)
    n = 200
    X = pd.DataFrame(np.random.randn(n, 6), columns=[f"feat_{i}" for i in range(6)])
    y = pd.Series(np.zeros(n, dtype=int))
    y.iloc[:10] = 1  # 10 positive cases (5%)
    return X, y


def test_apply_resampling_strategies(synthetic_imbalanced_data):
    """Verifies that resampling changes class balance on training data appropriately."""
    X, y = synthetic_imbalanced_data

    # 1. Strategy: None
    X_none, y_none = apply_resampling(X, y, strategy="none")
    assert len(X_none) == len(X)
    assert (y_none == 1).sum() == 10

    # 2. Strategy: Undersample
    X_under, y_under = apply_resampling(X, y, strategy="undersample", sampling_ratio=0.5)
    assert len(X_under) < len(X)
    assert (y_under == 1).sum() == 10
    # Ratio positive / total should be around 0.33 to 0.50
    assert (y_under == 1).mean() > (y == 1).mean()

    # 3. Strategy: Oversample
    X_over, y_over = apply_resampling(X, y, strategy="oversample", sampling_ratio=0.20)
    assert len(X_over) > len(X)
    assert (y_over == 1).sum() > 10


def test_find_optimal_threshold():
    """Verifies that find_optimal_threshold finds the maximum F1 cutoff point."""
    np.random.seed(42)
    n = 100
    y_true = np.zeros(n, dtype=int)
    y_true[:20] = 1

    # Probabilities centered around 0.30 for fraud and 0.05 for legit
    y_prob = np.random.uniform(0.0, 0.1, n)
    y_prob[:20] = np.random.uniform(0.2, 0.8, 20)

    result = find_optimal_threshold(y_true, y_prob)

    assert "optimal_threshold" in result
    assert "max_f1" in result
    assert 0.01 <= result["optimal_threshold"] <= 0.99
    assert result["max_f1"] > 0.0
    assert len(result["threshold_df"]) == 99


def test_plot_curves(tmp_path):
    """Verifies plotting of threshold curves and strategy comparison barplots."""
    dummy_df = pd.DataFrame({
        "threshold": [0.1, 0.2, 0.3, 0.4, 0.5],
        "precision": [0.5, 0.6, 0.7, 0.8, 0.9],
        "recall": [0.9, 0.8, 0.7, 0.5, 0.2],
        "f1_score": [0.64, 0.68, 0.70, 0.61, 0.32],
    })

    fig1 = plot_threshold_tuning_curve(dummy_df, 0.3, save_path=tmp_path / "t_curve.png")
    assert isinstance(fig1, plt.Figure)
    assert (tmp_path / "t_curve.png").exists()
    plt.close(fig1)

    strategies_dict = {
        "Baseline": {"Recall": 0.1, "Precision": 0.8, "F1-Score": 0.18},
        "Class Weighted": {"Recall": 0.85, "Precision": 0.65, "F1-Score": 0.74},
        "SMOTE": {"Recall": 0.88, "Precision": 0.70, "F1-Score": 0.78},
    }
    fig2 = plot_imbalance_strategies_comparison(strategies_dict, save_path=tmp_path / "strat.png")
    assert isinstance(fig2, plt.Figure)
    assert (tmp_path / "strat.png").exists()
    plt.close(fig2)


def test_train_save_and_load_champion(synthetic_imbalanced_data, tmp_path):
    """Verifies training, tuning, serialization, and reloading of the champion model."""
    X, y = synthetic_imbalanced_data
    model, metadata = train_optimized_champion(X, y, resample_strategy="oversample")

    assert hasattr(model, "predict")
    assert "best_params" in metadata
    assert "best_cv_f1_score" in metadata

    m_path = tmp_path / "champion.joblib"
    meta_path = tmp_path / "metadata.json"

    save_champion_model(model, metadata, model_path=m_path, metadata_path=meta_path)
    assert m_path.exists()
    assert meta_path.exists()

    loaded_m, loaded_meta = load_champion_model(model_path=m_path, metadata_path=meta_path)
    assert loaded_meta["model_type"] == metadata["model_type"]

    orig_preds = model.predict(X[:5])
    loaded_preds = loaded_m.predict(X[:5])
    np.testing.assert_array_equal(orig_preds, loaded_preds)
