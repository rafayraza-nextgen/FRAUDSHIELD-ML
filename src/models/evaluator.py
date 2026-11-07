"""
Model evaluation, comparison, and metrics computation for F.R.A.U.D.S.H.I.E.L.D.
Computes Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, and Confusion Matrices.
Generates multi-model curves and persists reports/model_comparison.csv.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

from src.config import REPORTS_DIR, FIGURES_DIR

MODEL_COLORS = {
    "Logistic Regression": "#2563eb",   # Blue
    "Decision Tree": "#059669",         # Emerald Green
    "Random Forest": "#7c3aed",         # Purple
    "Naive Bayes": "#d97706",           # Amber
    "Support Vector Machine": "#dc2626",# Red
    "K-Nearest Neighbors": "#0891b2",   # Cyan
}


class ModelEvaluator:
    """
    Evaluates classification models specifically within the context of fraud detection.
    Focuses on imbalanced metrics (Recall, Precision, PR-AUC) and timing benchmarks.
    """

    def __init__(self):
        self.results: Dict[str, Dict[str, Any]] = {}
        self.comparison_df: Optional[pd.DataFrame] = None

    def evaluate_model(
        self,
        name: str,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        y_prob: Union[pd.Series, np.ndarray],
        training_time: float = 0.0,
        inference_time: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Computes the complete suite of performance metrics for a single model.
        """
        y_t = np.asarray(y_true)
        y_p = np.asarray(y_pred)
        y_pr = np.asarray(y_prob)

        # Basic & Imbalance Metrics
        acc = float(accuracy_score(y_t, y_p))
        prec = float(precision_score(y_t, y_p, zero_division=0))
        rec = float(recall_score(y_t, y_p, zero_division=0))
        f1 = float(f1_score(y_t, y_p, zero_division=0))

        # Probability-based area metrics (ROC-AUC & PR-AUC)
        try:
            roc_auc = float(roc_auc_score(y_t, y_pr))
        except ValueError:
            roc_auc = 0.5

        try:
            pr_auc = float(average_precision_score(y_t, y_pr))
        except ValueError:
            pr_auc = float(y_t.mean())

        # Confusion Matrix counts: tn, fp, fn, tp
        cm = confusion_matrix(y_t, y_p, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        metrics = {
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(roc_auc, 4),
            "PR-AUC": round(pr_auc, 4),
            "True Positives (TP)": int(tp),
            "False Positives (FP)": int(fp),
            "False Negatives (FN)": int(fn),
            "True Negatives (TN)": int(tn),
            "Training Time (s)": round(training_time, 4),
            "Inference Time (s)": round(inference_time, 4),
            "confusion_matrix": cm,
            "y_prob": y_pr,
        }

        self.results[name] = metrics
        return metrics

    def generate_comparison_table(self) -> pd.DataFrame:
        """
        Formats all evaluated model metrics into a sorted pandas comparison DataFrame.
        Sorted by PR-AUC descending (the primary metric for extreme class imbalance).
        """
        rows = []
        for name, m in self.results.items():
            row = {k: v for k, v in m.items() if k not in ["confusion_matrix", "y_prob"]}
            rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty and "PR-AUC" in df.columns:
            df = df.sort_values(by="PR-AUC", ascending=False).reset_index(drop=True)
        self.comparison_df = df
        return df

    def save_comparison_table(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        """
        Saves the verified actual experimental results to reports/model_comparison.csv.
        """
        if self.comparison_df is None or self.comparison_df.empty:
            self.generate_comparison_table()

        target_path = Path(filepath) if filepath else REPORTS_DIR / "model_comparison.csv"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        self.comparison_df.to_csv(target_path, index=False)
        return target_path

    def plot_confusion_matrices(self, save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Plots a grid of annotated confusion matrices for all evaluated models.
        """
        n_models = len(self.results)
        if n_models == 0:
            raise RuntimeError("No model evaluation results available to plot.")

        cols = 3
        rows = int(np.ceil(n_models / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
        axes = np.array(axes).flatten()

        for idx, (name, data) in enumerate(self.results.items()):
            ax = axes[idx]
            cm = data["confusion_matrix"]
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                ax=ax,
                annot_kws={"fontsize": 11, "fontweight": "bold"},
            )
            ax.set_title(f"{name}\nRec: {data['Recall']:.2f} | Prec: {data['Precision']:.2f} | F1: {data['F1-Score']:.2f}", fontweight="bold")
            ax.set_xlabel("Predicted Label (0: Legit, 1: Fraud)", fontsize=9)
            ax.set_ylabel("Actual Label", fontsize=9)

        # Hide any unused subplots
        for j in range(idx + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle("Confusion Matrices Across Baseline Models", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        return fig

    def plot_roc_curves(
        self,
        y_true: Union[pd.Series, np.ndarray],
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        Plots combined Receiver Operating Characteristic (ROC) curves with random chance baseline.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        y_t = np.asarray(y_true)

        for name, data in self.results.items():
            fpr, tpr, _ = roc_curve(y_t, data["y_prob"])
            color = MODEL_COLORS.get(name, "#475569")
            ax.plot(fpr, tpr, label=f"{name} (AUC = {data['ROC-AUC']:.3f})", color=color, linewidth=2)

        # 50% Random chance line
        ax.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", label="Random Chance (AUC = 0.500)")
        ax.set_title("ROC Curves Comparison Across Baseline Models", fontweight="bold", pad=12)
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
        ax.set_ylabel("True Positive Rate (Recall)", fontweight="bold")
        ax.legend(loc="lower right", frameon=True)
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        return fig

    def plot_precision_recall_curves(
        self,
        y_true: Union[pd.Series, np.ndarray],
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        Plots combined Precision-Recall curves. Essential for imbalanced fraud datasets.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        y_t = np.asarray(y_true)
        no_skill_baseline = y_t.mean()

        for name, data in self.results.items():
            precision, recall, _ = precision_recall_curve(y_t, data["y_prob"])
            color = MODEL_COLORS.get(name, "#475569")
            ax.plot(recall, precision, label=f"{name} (PR-AUC = {data['PR-AUC']:.3f})", color=color, linewidth=2)

        # No-skill baseline (proportion of positive cases)
        ax.axhline(
            y=no_skill_baseline,
            color="#94a3b8",
            linestyle="--",
            label=f"No-Skill Baseline ({no_skill_baseline:.4f})",
        )

        ax.set_title("Precision-Recall Curves (Key Imbalanced Metric)", fontweight="bold", pad=12)
        ax.set_xlabel("Recall (Fraud Coverage)", fontweight="bold")
        ax.set_ylabel("Precision (True Fraud / Total Flagged)", fontweight="bold")
        ax.legend(loc="upper right", frameon=True)
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        return fig
