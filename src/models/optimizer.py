"""
Class imbalance handling, threshold tuning, and model optimization for F.R.A.U.D.S.H.I.E.L.D.
Provides SMOTE, Random Undersampling, Cost-Sensitive Class Weighting,
Precision-Recall threshold calibration, and champion model persistence.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

from src.config import MODELS_DIR, FIGURES_DIR, RANDOM_STATE

# Optional import of imbalanced-learn with robust fallbacks
try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False


def apply_resampling(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    strategy: str = "smote",
    sampling_ratio: float = 0.20,
    random_state: int = RANDOM_STATE,
) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]:
    """
    Applies resampling strictly to training data to mitigate severe class imbalance.
    Strategies:
      - 'smote': Synthetic Minority Over-sampling Technique.
      - 'undersample': Random downsampling of the legitimate majority class.
      - 'oversample': Random duplication of fraudulent minority transactions.
      - 'none': Unaltered training distribution.
    """
    if strategy == "none":
        return X_train, y_train

    # Check for imblearn availability
    if HAS_IMBLEARN:
        if strategy == "smote":
            # SMOTE synthesizes minority cases
            sampler = SMOTE(sampling_strategy=sampling_ratio, random_state=random_state)
            X_res, y_res = sampler.fit_resample(X_train, y_train)
            return X_res, y_res
        elif strategy == "undersample":
            sampler = RandomUnderSampler(sampling_strategy=sampling_ratio, random_state=random_state)
            X_res, y_res = sampler.fit_resample(X_train, y_train)
            return X_res, y_res
        elif strategy == "oversample":
            sampler = RandomOverSampler(sampling_strategy=sampling_ratio, random_state=random_state)
            X_res, y_res = sampler.fit_resample(X_train, y_train)
            return X_res, y_res

    # Robust pure-NumPy/Pandas fallback if imblearn is not installed
    X_df = pd.DataFrame(X_train) if not isinstance(X_train, pd.DataFrame) else X_train.copy()
    y_s = pd.Series(y_train) if not isinstance(y_train, pd.Series) else y_train.copy()

    pos_idx = y_s[y_s == 1].index
    neg_idx = y_s[y_s == 0].index

    if strategy == "undersample":
        target_neg_count = int(len(pos_idx) / sampling_ratio)
        sampled_neg_idx = np.random.RandomState(random_state).choice(
            neg_idx, min(target_neg_count, len(neg_idx)), replace=False
        )
        final_idx = np.concatenate([pos_idx, sampled_neg_idx])
        return X_df.loc[final_idx], y_s.loc[final_idx]

    elif strategy in ["smote", "oversample"]:
        # Fallback to random oversampling
        target_pos_count = int(len(neg_idx) * sampling_ratio)
        sampled_pos_idx = np.random.RandomState(random_state).choice(
            pos_idx, target_pos_count, replace=True
        )
        final_idx = np.concatenate([neg_idx, sampled_pos_idx])
        return X_df.loc[final_idx], y_s.loc[final_idx]

    raise ValueError(f"Unknown resampling strategy '{strategy}'.")


def find_optimal_threshold(
    y_true: Union[pd.Series, np.ndarray],
    y_prob: Union[pd.Series, np.ndarray],
    num_thresholds: int = 99,
) -> Dict[str, Any]:
    """
    Sweeps decision thresholds t in [0.01, 0.99] to find the optimal operating threshold
    that maximizes the F1-Score (optimal harmonic balance between precision and recall).
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_prob)

    thresholds = np.linspace(0.01, 0.99, num_thresholds)
    f1_scores = []
    precisions = []
    recalls = []

    for t in thresholds:
        preds = (y_p >= t).astype(int)
        f1_scores.append(f1_score(y_t, preds, zero_division=0))
        precisions.append(precision_score(y_t, preds, zero_division=0))
        recalls.append(recall_score(y_t, preds, zero_division=0))

    threshold_df = pd.DataFrame({
        "threshold": thresholds,
        "precision": precisions,
        "recall": recalls,
        "f1_score": f1_scores,
    })

    best_idx = np.argmax(f1_scores)
    best_row = threshold_df.iloc[best_idx]

    return {
        "optimal_threshold": round(float(best_row["threshold"]), 4),
        "max_f1": round(float(best_row["f1_score"]), 4),
        "precision_at_optimal": round(float(best_row["precision"]), 4),
        "recall_at_optimal": round(float(best_row["recall"]), 4),
        "threshold_df": threshold_df,
    }


def plot_threshold_tuning_curve(
    threshold_df: pd.DataFrame,
    optimal_threshold: float,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Visualizes Precision, Recall, and F1 curves across decision thresholds.
    Highlights the optimal operating cutoff point.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5))

    ax.plot(threshold_df["threshold"], threshold_df["precision"], label="Precision", color="#2563eb", linewidth=2)
    ax.plot(threshold_df["threshold"], threshold_df["recall"], label="Recall", color="#dc2626", linewidth=2)
    ax.plot(threshold_df["threshold"], threshold_df["f1_score"], label="F1-Score", color="#10b981", linewidth=2.5)

    # Vertical line at optimal threshold
    ax.axvline(
        x=optimal_threshold,
        color="#7c3aed",
        linestyle="--",
        linewidth=2,
        label=f"Optimal Threshold (t* = {optimal_threshold:.2f})",
    )

    ax.set_title("Decision Threshold Optimization (Precision-Recall Trade-off)", fontweight="bold", pad=12)
    ax.set_xlabel("Decision Threshold (t)", fontweight="bold")
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="best", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_imbalance_strategies_comparison(
    strategies_results: Dict[str, Dict[str, float]],
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Visualizes performance comparisons across different imbalance techniques:
    Baseline vs. Class Weights vs. Undersampling vs. SMOTE.
    """
    df = pd.DataFrame(strategies_results).T.reset_index().rename(columns={"index": "Strategy"})

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Recall
    sns.barplot(data=df, x="Strategy", y="Recall", ax=axes[0], palette="Reds_r", hue="Strategy", legend=False)
    axes[0].set_title("Recall (Fraud Coverage)", fontweight="bold")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # Precision
    sns.barplot(data=df, x="Strategy", y="Precision", ax=axes[1], palette="Blues_r", hue="Strategy", legend=False)
    axes[1].set_title("Precision (Signal Fidelity)", fontweight="bold")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    # F1-Score
    sns.barplot(data=df, x="Strategy", y="F1-Score", ax=axes[2], palette="Greens_r", hue="Strategy", legend=False)
    axes[2].set_title("F1-Score (Harmonic Balance)", fontweight="bold")
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(axis="y", linestyle="--", alpha=0.5)

    for ax in axes:
        ax.tick_params(axis="x", rotation=20)

    plt.suptitle("Imbalance Handling Strategy Comparison", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def train_optimized_champion(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    resample_strategy: str = "smote",
    sampling_ratio: float = 0.20,
    random_state: int = RANDOM_STATE,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Trains and tunes our final champion Random Forest classifier combining:
    1. SMOTE / Class rebalancing
    2. Cost-sensitive class weighting (`class_weight='balanced_subsample'`)
    3. Randomized hyperparameter search with Stratified 3-Fold Cross-Validation.
    """
    # 1. Apply resampling strictly to training data
    X_res, y_res = apply_resampling(
        X_train, y_train, strategy=resample_strategy, sampling_ratio=sampling_ratio, random_state=random_state
    )

    # 2. Hyperparameter Search Space
    param_distributions = {
        "n_estimators": [100, 150, 200],
        "max_depth": [10, 14, 18, None],
        "min_samples_split": [5, 10, 20],
        "min_samples_leaf": [2, 5, 10],
        "class_weight": ["balanced", "balanced_subsample"],
    }

    base_rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_distributions,
        n_iter=6,
        scoring="f1",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
    )

    start_time = time.perf_counter()
    search.fit(X_res, y_res)
    tuning_duration = time.perf_counter() - start_time

    champion_model = search.best_estimator_

    metadata = {
        "model_type": "RandomForestClassifier",
        "best_params": search.best_params_,
        "best_cv_f1_score": round(float(search.best_score_), 4),
        "resample_strategy": resample_strategy,
        "sampling_ratio": sampling_ratio,
        "tuning_time_seconds": round(tuning_duration, 2),
        "trained_timestamp": datetime.now().isoformat(),
    }

    return champion_model, metadata


def save_champion_model(
    model: Any,
    metadata: Dict[str, Any],
    model_path: Optional[Union[str, Path]] = None,
    metadata_path: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Path]:
    """
    Persists the champion model (.joblib) and its operational metadata (.json).
    """
    m_path = Path(model_path) if model_path else MODELS_DIR / "fraud_detection_champion.joblib"
    meta_path = Path(metadata_path) if metadata_path else MODELS_DIR / "model_metadata.json"

    m_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, m_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return m_path, meta_path


def load_champion_model(
    model_path: Optional[Union[str, Path]] = None,
    metadata_path: Optional[Union[str, Path]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Loads the serialized champion model and its metadata dictionary.
    """
    m_path = Path(model_path) if model_path else MODELS_DIR / "fraud_detection_champion.joblib"
    meta_path = Path(metadata_path) if metadata_path else MODELS_DIR / "model_metadata.json"

    if not m_path.exists():
        raise FileNotFoundError(f"Champion model not found at {m_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Champion metadata not found at {meta_path}")

    model = joblib.load(m_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return model, metadata
