"""
Unit tests for data visualization utilities.
Verifies Phase 3 functionality: figure generation, plot rendering, and artifact output.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from src.utils.visualization import (
    plot_class_imbalance,
    plot_transaction_types_breakdown,
    plot_amount_distributions,
    plot_balance_dynamics,
    plot_temporal_patterns,
    plot_correlation_matrix,
    generate_all_eda_figures,
)


@pytest.fixture
def sample_eda_df():
    """Generates a representative sample transaction DataFrame for plotting tests."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "step": np.random.randint(1, 100, size=n),
        "type": np.random.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"], size=n),
        "amount": np.random.exponential(50000, size=n),
        "oldbalanceOrg": np.random.exponential(100000, size=n),
        "newbalanceOrig": np.random.exponential(80000, size=n),
        "oldbalanceDest": np.random.exponential(100000, size=n),
        "newbalanceDest": np.random.exponential(150000, size=n),
        "isFraud": np.random.choice([0, 1], size=n, p=[0.90, 0.10]),
    })


def test_individual_plots_generation(sample_eda_df, tmp_path):
    """Test that all individual plotting functions return valid figures and save correctly."""
    fig1 = plot_class_imbalance(sample_eda_df, save_path=tmp_path / "test_imbalance.png")
    assert isinstance(fig1, plt.Figure)
    assert (tmp_path / "test_imbalance.png").exists()
    plt.close(fig1)

    fig2 = plot_transaction_types_breakdown(sample_eda_df, save_path=tmp_path / "test_types.png")
    assert isinstance(fig2, plt.Figure)
    assert (tmp_path / "test_types.png").exists()
    plt.close(fig2)

    fig3 = plot_amount_distributions(sample_eda_df, save_path=tmp_path / "test_amount.png")
    assert isinstance(fig3, plt.Figure)
    assert (tmp_path / "test_amount.png").exists()
    plt.close(fig3)

    fig4 = plot_balance_dynamics(sample_eda_df, save_path=tmp_path / "test_balance.png")
    assert isinstance(fig4, plt.Figure)
    assert (tmp_path / "test_balance.png").exists()
    plt.close(fig4)

    fig5 = plot_temporal_patterns(sample_eda_df, save_path=tmp_path / "test_temporal.png")
    assert isinstance(fig5, plt.Figure)
    assert (tmp_path / "test_temporal.png").exists()
    plt.close(fig5)

    fig6 = plot_correlation_matrix(sample_eda_df, save_path=tmp_path / "test_corr.png")
    assert isinstance(fig6, plt.Figure)
    assert (tmp_path / "test_corr.png").exists()
    plt.close(fig6)


def test_generate_all_eda_figures(sample_eda_df, tmp_path):
    """Test batch figure generation in custom directory."""
    saved = generate_all_eda_figures(sample_eda_df, output_dir=tmp_path)
    assert len(saved) == 6
    for key, path in saved.items():
        assert Path(path).exists()
        assert Path(path).stat().st_size > 0
