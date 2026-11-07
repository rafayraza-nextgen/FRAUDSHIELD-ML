"""
Unit tests for model evaluator and metrics computation.
Verifies Phase 5 functionality: metric calculation, comparison table generation,
figure rendering, and CSV persistence.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from src.models.evaluator import ModelEvaluator


@pytest.fixture
def evaluation_data():
    """Generates synthetic true labels and predictions for 2 sample models."""
    np.random.seed(42)
    n = 200
    # Imbalanced ground truth (10% positive)
    y_true = np.zeros(n, dtype=int)
    y_true[:20] = 1

    # Model A: High recall, moderate precision
    y_pred_a = y_true.copy()
    y_pred_a[20:30] = 1  # 10 false positives
    y_prob_a = y_pred_a * 0.8 + np.random.uniform(0, 0.2, size=n)

    # Model B: Low recall, high precision
    y_pred_b = np.zeros(n, dtype=int)
    y_pred_b[:10] = 1  # Missed 10 fraud cases (10 false negatives)
    y_prob_b = y_pred_b * 0.9 + np.random.uniform(0, 0.1, size=n)

    return y_true, (y_pred_a, y_prob_a), (y_pred_b, y_prob_b)


def test_evaluate_single_model_metrics(evaluation_data):
    """Verifies that individual model evaluation correctly computes all 12 target metrics."""
    y_true, (y_pred_a, y_prob_a), _ = evaluation_data
    evaluator = ModelEvaluator()

    metrics = evaluator.evaluate_model(
        "Model A", y_true, y_pred_a, y_prob_a, training_time=1.5, inference_time=0.05
    )

    required_keys = [
        "Model", "Accuracy", "Precision", "Recall", "F1-Score",
        "ROC-AUC", "PR-AUC", "True Positives (TP)", "False Positives (FP)",
        "False Negatives (FN)", "True Negatives (TN)", "Training Time (s)", "Inference Time (s)"
    ]
    for key in required_keys:
        assert key in metrics

    # Recall for Model A should be 1.0 (caught all 20)
    assert metrics["Recall"] == 1.0
    assert metrics["True Positives (TP)"] == 20
    assert metrics["False Positives (FP)"] == 10
    assert metrics["False Negatives (FN)"] == 0


def test_comparison_table_and_csv_export(evaluation_data, tmp_path):
    """Verifies comparison table generation and CSV persistence."""
    y_true, (y_pred_a, y_prob_a), (y_pred_b, y_prob_b) = evaluation_data
    evaluator = ModelEvaluator()

    evaluator.evaluate_model("Model A", y_true, y_pred_a, y_prob_a, 1.0, 0.1)
    evaluator.evaluate_model("Model B", y_true, y_pred_b, y_prob_b, 0.5, 0.05)

    comp_df = evaluator.generate_comparison_table()
    assert len(comp_df) == 2
    assert "PR-AUC" in comp_df.columns
    assert comp_df.iloc[0]["PR-AUC"] >= comp_df.iloc[1]["PR-AUC"]

    # Export CSV check
    csv_path = tmp_path / "model_comparison.csv"
    saved_path = evaluator.save_comparison_table(csv_path)
    assert saved_path.exists()

    loaded_df = pd.read_csv(csv_path)
    assert len(loaded_df) == 2
    assert "Model" in loaded_df.columns


def test_plots_generation(evaluation_data, tmp_path):
    """Verifies confusion matrix, ROC curve, and PR curve generation."""
    y_true, (y_pred_a, y_prob_a), (y_pred_b, y_prob_b) = evaluation_data
    evaluator = ModelEvaluator()

    evaluator.evaluate_model("Model A", y_true, y_pred_a, y_prob_a)
    evaluator.evaluate_model("Model B", y_true, y_pred_b, y_prob_b)

    # 1. Confusion Matrices
    fig_cm = evaluator.plot_confusion_matrices(save_path=tmp_path / "cm.png")
    assert isinstance(fig_cm, plt.Figure)
    assert (tmp_path / "cm.png").exists()
    plt.close(fig_cm)

    # 2. ROC Curves
    fig_roc = evaluator.plot_roc_curves(y_true, save_path=tmp_path / "roc.png")
    assert isinstance(fig_roc, plt.Figure)
    assert (tmp_path / "roc.png").exists()
    plt.close(fig_roc)

    # 3. PR Curves
    fig_pr = evaluator.plot_precision_recall_curves(y_true, save_path=tmp_path / "pr.png")
    assert isinstance(fig_pr, plt.Figure)
    assert (tmp_path / "pr.png").exists()
    plt.close(fig_pr)
