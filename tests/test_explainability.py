"""
Unit tests for Explainable AI (XAI) and SHAP attribution module.
Verifies Phase 7 functionality: global feature importance, individual transaction attribution,
plain-English summaries, and visual asset exports.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from src.explainability.explainer import FraudExplainer


@pytest.fixture
def trained_explainer_fixture():
    """Generates a small trained Random Forest model and FraudExplainer instance."""
    np.random.seed(42)
    n = 100
    features = [
        "amount", "errorBalanceOrig", "errorBalanceDest",
        "type_TRANSFER", "type_CASH_OUT", "isDestMerchant"
    ]
    X = pd.DataFrame(np.random.randn(n, len(features)), columns=features)
    # Fraud correlated with high amount and errorBalanceOrig
    y = ((X["amount"] > 0.8) & (X["errorBalanceOrig"] > 0.5)).astype(int)
    if y.sum() == 0:
        y.iloc[0] = 1

    model = RandomForestClassifier(n_estimators=20, random_state=42).fit(X, y)
    explainer = FraudExplainer(model, feature_names=features)
    return explainer, X, y


def test_global_feature_importance(trained_explainer_fixture):
    """Verifies that global feature importance correctly ranks top features."""
    explainer, X, _ = trained_explainer_fixture
    importance_df = explainer.get_global_feature_importance(X)

    assert isinstance(importance_df, pd.DataFrame)
    assert len(importance_df) == len(X.columns)
    for col in ["Feature", "Mean_Absolute_SHAP", "Relative_Importance_Pct"]:
        assert col in importance_df.columns

    # Check that rows are sorted descending by importance
    assert (
        importance_df["Mean_Absolute_SHAP"].iloc[0]
        >= importance_df["Mean_Absolute_SHAP"].iloc[-1]
    )


def test_explain_transaction_local(trained_explainer_fixture):
    """Verifies individual transaction attribution, narrative synthesis, and disclaimers."""
    explainer, X, _ = trained_explainer_fixture
    single_tx = X.iloc[0].to_dict()

    explanation = explainer.explain_transaction(single_tx)

    # Core required fields
    assert "fraud_probability" in explanation
    assert "risk_tier" in explanation
    assert "risk_drivers" in explanation
    assert "mitigating_factors" in explanation
    assert "human_readable_summary" in explanation
    assert "legal_disclaimer" in explanation

    # Check probability bounds
    assert 0.0 <= explanation["fraud_probability"] <= 1.0
    assert explanation["risk_tier"] in ["Low Risk", "Moderate Risk", "High Risk"]

    # Verify plain-English summary text contains probability and risk tier
    assert explanation["risk_tier"] in explanation["human_readable_summary"]
    assert "legal proof" in explanation["legal_disclaimer"].lower()


def test_plots_generation(trained_explainer_fixture, tmp_path):
    """Verifies generation of global bar chart and local waterfall plot."""
    explainer, X, _ = trained_explainer_fixture

    # 1. Global Bar Plot
    fig_bar = explainer.plot_feature_importance_bar(X, top_n=5, save_path=tmp_path / "global_bar.png")
    assert isinstance(fig_bar, plt.Figure)
    assert (tmp_path / "global_bar.png").exists()
    plt.close(fig_bar)

    # 2. Local Waterfall Plot
    single_tx = X.iloc[0]
    fig_waterfall = explainer.plot_transaction_waterfall(single_tx, save_path=tmp_path / "waterfall.png")
    assert isinstance(fig_waterfall, plt.Figure)
    assert (tmp_path / "waterfall.png").exists()
    plt.close(fig_waterfall)
