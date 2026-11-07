"""
Data visualization and publication-quality figure generation for F.R.A.U.D.S.H.I.E.L.D.
Provides utilities to visualize fraud distributions, balance dynamics, temporal trends, and correlations.
"""

from pathlib import Path
from typing import Optional, Union, Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.config import FIGURES_DIR

# Establish consistent visual styling across light and dark themes
PALETTE_LEGIT = "#2563eb"   # Trust Blue
PALETTE_FRAUD = "#dc2626"   # Warning / Fraud Red
PALETTE_CYBER = ["#0284c7", "#7c3aed", "#10b981", "#f59e0b", "#ef4444"]

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 14,
    "figure.dpi": 150,
})


def plot_class_imbalance(df: pd.DataFrame, save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Plots the extreme class imbalance between legitimate and fraudulent transactions.
    Uses log-scale bar chart and clear percentage annotations.
    """
    counts = df["isFraud"].value_counts().sort_index()
    percentages = df["isFraud"].value_counts(normalize=True).sort_index() * 100

    labels = ["Legitimate (0)", "Fraudulent (1)"]
    colors = [PALETTE_LEGIT, PALETTE_FRAUD]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, counts.values, color=colors, width=0.55, edgecolor="black", linewidth=0.8)

    ax.set_yscale("log")
    ax.set_ylabel("Transaction Count (Log Scale)", fontweight="bold")
    ax.set_title("Extreme Class Imbalance: Legitimate vs. Fraudulent Transactions", fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # Annotate bars with exact counts and percentages
    for bar, count, pct in zip(bars, counts.values, percentages.values):
        height = bar.get_height()
        ax.annotate(
            f"{count:,}\n({pct:.3f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_transaction_types_breakdown(df: pd.DataFrame, save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Dual-panel plot showing overall transaction volume by type versus fraud incidence.
    Demonstrates that fraud is concentrated exclusively in TRANSFER and CASH_OUT.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Overall volume by type
    type_counts = df["type"].value_counts()
    axes[0].bar(type_counts.index, type_counts.values, color="#3b82f6", edgecolor="black", linewidth=0.8)
    axes[0].set_title("Total Transaction Volume by Type", fontweight="bold")
    axes[0].set_ylabel("Count", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # Panel 2: Fraud incidence by type
    cross = pd.crosstab(df["type"], df["isFraud"])
    if 1 in cross.columns:
        fraud_counts = cross[1].reindex(type_counts.index, fill_value=0)
        axes[1].bar(fraud_counts.index, fraud_counts.values, color=PALETTE_FRAUD, edgecolor="black", linewidth=0.8)
    axes[1].set_title("Fraudulent Transactions by Type", fontweight="bold")
    axes[1].set_ylabel("Fraud Count", fontweight="bold")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_amount_distributions(df: pd.DataFrame, save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Plots the log-transformed distribution of transaction amounts partitioned by fraud status.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))

    legit_amounts = df[df["isFraud"] == 0]["amount"]
    fraud_amounts = df[df["isFraud"] == 1]["amount"]

    # Log10 transformation (+1 to handle potential zero amounts)
    log_legit = np.log10(np.maximum(1.0, legit_amounts))
    log_fraud = np.log10(np.maximum(1.0, fraud_amounts))

    sns.kdeplot(log_legit, ax=ax, label="Legitimate (0)", color=PALETTE_LEGIT, fill=True, alpha=0.3, linewidth=2)
    sns.kdeplot(log_fraud, ax=ax, label="Fraudulent (1)", color=PALETTE_FRAUD, fill=True, alpha=0.3, linewidth=2)

    ax.set_title("Transaction Amount Distribution (Log10 Transformed)", fontweight="bold", pad=12)
    ax.set_xlabel("Log10(Amount in Currency Units)", fontweight="bold")
    ax.set_ylabel("Density", fontweight="bold")
    ax.legend(title="Transaction Class", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_balance_dynamics(df: pd.DataFrame, save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Visualizes balance dynamics: Origin account balance vs. Transaction amount.
    Highlights zero-balance drains typical in fraudulent transactions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    legit_sample = df[df["isFraud"] == 0].sample(min(2000, len(df[df["isFraud"] == 0])), random_state=42)
    fraud_sample = df[df["isFraud"] == 1]

    # Panel 1: Legitimate
    axes[0].scatter(
        np.log10(np.maximum(1.0, legit_sample["oldbalanceOrg"])),
        np.log10(np.maximum(1.0, legit_sample["amount"])),
        alpha=0.4,
        color=PALETTE_LEGIT,
        s=18,
        label="Legitimate",
    )
    axes[0].set_title("Legitimate: Old Balance vs. Amount (Log10)", fontweight="bold")
    axes[0].set_xlabel("Log10(Old Balance Origin)", fontweight="bold")
    axes[0].set_ylabel("Log10(Amount)", fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Fraudulent
    axes[1].scatter(
        np.log10(np.maximum(1.0, fraud_sample["oldbalanceOrg"])),
        np.log10(np.maximum(1.0, fraud_sample["amount"])),
        alpha=0.6,
        color=PALETTE_FRAUD,
        s=24,
        label="Fraudulent",
    )
    axes[1].set_title("Fraudulent: Old Balance vs. Amount (Log10)", fontweight="bold")
    axes[1].set_xlabel("Log10(Old Balance Origin)", fontweight="bold")
    axes[1].set_ylabel("Log10(Amount)", fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_temporal_patterns(df: pd.DataFrame, save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Plots transaction volumes and fraud counts across simulated hours (step % 24).
    Exposes cyclical daily patterns and fraud concentration.
    """
    data = df.copy()
    data["hour_of_day"] = data["step"] % 24

    hourly_summary = data.groupby("hour_of_day")["isFraud"].agg(["count", "sum"]).reset_index()
    hourly_summary.columns = ["hour", "total_tx", "fraud_tx"]
    hourly_summary["fraud_rate"] = hourly_summary["fraud_tx"] / np.maximum(1, hourly_summary["total_tx"]) * 100

    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    ax2 = ax1.twinx()

    # Total transactions on primary y-axis
    ax1.plot(hourly_summary["hour"], hourly_summary["total_tx"], color="#64748b", marker="o", linewidth=2, label="Total Transactions")
    ax1.set_xlabel("Hour of Day (Simulation Step % 24)", fontweight="bold")
    ax1.set_ylabel("Total Transactions", color="#64748b", fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#64748b")
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Fraud rate on secondary y-axis
    ax2.plot(hourly_summary["hour"], hourly_summary["fraud_rate"], color=PALETTE_FRAUD, marker="s", linewidth=2.5, label="Fraud Rate (%)")
    ax2.set_ylabel("Fraud Rate (%)", color=PALETTE_FRAUD, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=PALETTE_FRAUD)

    plt.title("Temporal Fraud Dynamics Across 24-Hour Cycle", fontweight="bold", pad=12)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_correlation_matrix(df: pd.DataFrame, save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Plots an annotated Pearson correlation heatmap among numerical features and the target variable.
    """
    num_cols = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "isFraud"]
    valid_cols = [c for c in num_cols if c in df.columns]

    corr = df[valid_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6.5))
    mask = np.triu(np.ones_like(corr, dtype=bool))

    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Correlation Heatmap (Numerical Features & isFraud)", fontweight="bold", pad=12)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def generate_all_eda_figures(df: pd.DataFrame, output_dir: Optional[Union[str, Path]] = None) -> Dict[str, Path]:
    """
    Generates and saves the complete standard Phase 3 EDA visualization suite.
    """
    target_dir = Path(output_dir) if output_dir else FIGURES_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    # 1. Class Imbalance
    p1 = target_dir / "01_class_distribution.png"
    plot_class_imbalance(df, save_path=p1)
    plt.close()
    saved_files["class_distribution"] = p1

    # 2. Transaction Types Breakdown
    p2 = target_dir / "02_transaction_types_and_fraud.png"
    plot_transaction_types_breakdown(df, save_path=p2)
    plt.close()
    saved_files["transaction_types"] = p2

    # 3. Amount Distributions
    p3 = target_dir / "03_amount_distribution_log.png"
    plot_amount_distributions(df, save_path=p3)
    plt.close()
    saved_files["amount_distribution"] = p3

    # 4. Balance Dynamics
    p4 = target_dir / "04_balance_dynamics_fraud_vs_legit.png"
    plot_balance_dynamics(df, save_path=p4)
    plt.close()
    saved_files["balance_dynamics"] = p4

    # 5. Temporal Patterns
    p5 = target_dir / "05_temporal_fraud_patterns.png"
    plot_temporal_patterns(df, save_path=p5)
    plt.close()
    saved_files["temporal_patterns"] = p5

    # 6. Correlation Heatmap
    p6 = target_dir / "06_correlation_heatmap.png"
    plot_correlation_matrix(df, save_path=p6)
    plt.close()
    saved_files["correlation_heatmap"] = p6

    return saved_files
