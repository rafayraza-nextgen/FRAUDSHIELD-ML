"""
Explainable AI (XAI) and model attribution engine for F.R.A.U.D.S.H.I.E.L.D.
Provides SHAP (SHapley Additive exPlanations) and tree-attribution methods to explain
both global model patterns and individual transaction fraud-risk predictions.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FIGURES_DIR

# Check for SHAP availability
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


class FraudExplainer:
    """
    Explainability engine for F.R.A.U.D.S.H.I.E.L.D.
    Explains why transactions receive elevated fraud-risk scores using SHAP attribution.
    """

    def __init__(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
        background_data: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    ):
        self.model = model
        self.feature_names = feature_names or []
        self.background_data = background_data
        self.tree_explainer: Optional[Any] = None

        if HAS_SHAP and hasattr(model, "estimators_"):
            try:
                self.tree_explainer = shap.TreeExplainer(model)
            except Exception:
                self.tree_explainer = None

    def _get_feature_names(self, X: Union[pd.DataFrame, np.ndarray]) -> List[str]:
        if isinstance(X, pd.DataFrame):
            return list(X.columns)
        if self.feature_names:
            return self.feature_names
        return [f"Feature_{i}" for i in range(X.shape[1])]

    def compute_shap_values(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Computes SHAP values matrix for input samples.
        Falls back to tree-based feature attribution if shap is unavailable.
        """
        if self.tree_explainer is not None:
            try:
                shap_vals = self.tree_explainer.shap_values(X)
                # If multiclass/binary, pick class 1 (Fraud)
                if isinstance(shap_vals, list) and len(shap_vals) == 2:
                    return np.asarray(shap_vals[1])
                elif isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 3:
                    return shap_vals[:, :, 1]
                return np.asarray(shap_vals)
            except Exception:
                pass

        # Fallback: Tree impurity attribution
        X_mat = np.asarray(X)
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            centered_X = X_mat - np.mean(X_mat, axis=0, keepdims=True)
            return centered_X * importances
        return np.zeros_like(X_mat)

    def get_global_feature_importance(
        self,
        X_sample: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    ) -> pd.DataFrame:
        """
        Calculates global feature importance rankings across all input features.
        """
        if X_sample is not None:
            shap_values = self.compute_shap_values(X_sample)
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            feat_names = self._get_feature_names(X_sample)
        elif hasattr(self.model, "feature_importances_"):
            mean_abs_shap = self.model.feature_importances_
            feat_names = self.feature_names if self.feature_names else [f"Feature_{i}" for i in range(len(mean_abs_shap))]
        else:
            raise ValueError("Must provide X_sample or a model with feature_importances_.")

        df = pd.DataFrame({
            "Feature": feat_names,
            "Mean_Absolute_SHAP": mean_abs_shap,
        }).sort_values(by="Mean_Absolute_SHAP", ascending=False).reset_index(drop=True)

        total = df["Mean_Absolute_SHAP"].sum()
        df["Relative_Importance_Pct"] = (df["Mean_Absolute_SHAP"] / (total if total > 0 else 1.0) * 100).round(2)
        return df

    def explain_transaction(
        self,
        transaction: Union[pd.DataFrame, pd.Series, Dict[str, Any]],
        top_n: int = 4,
    ) -> Dict[str, Any]:
        """
        Generates individual-level prediction explanation for a single transaction.
        Identifies top risk drivers, mitigating factors, and a plain-English narrative.
        """
        if isinstance(transaction, dict):
            tx_df = pd.DataFrame([transaction])
        elif isinstance(transaction, pd.Series):
            tx_df = pd.DataFrame([transaction])
        else:
            tx_df = transaction.copy()

        feat_names = self._get_feature_names(tx_df)
        shap_vals = self.compute_shap_values(tx_df)[0]
        tx_vals = np.asarray(tx_df)[0]

        # Get base expected value and predicted probability
        if hasattr(self.model, "predict_proba"):
            fraud_prob = float(self.model.predict_proba(tx_df)[0, 1])
        else:
            fraud_prob = float(self.model.predict(tx_df)[0])

        base_val = 0.05  # Average population fraud baseline

        # Pair features with contributions
        attributions = []
        for name, s_val, raw_val in zip(feat_names, shap_vals, tx_vals):
            attributions.append({
                "feature": name,
                "shap_value": float(s_val),
                "feature_value": float(raw_val) if np.issubdtype(type(raw_val), np.number) else str(raw_val),
            })

        # Separate risk drivers (positive SHAP) vs. mitigating factors (negative SHAP)
        sorted_attr = sorted(attributions, key=lambda x: abs(x["shap_value"]), reverse=True)
        risk_drivers = [a for a in sorted_attr if a["shap_value"] > 0][:top_n]
        mitigating = [a for a in sorted_attr if a["shap_value"] < 0][:top_n]

        # Generate Human-Readable Narrative
        risk_tier = "High Risk" if fraud_prob >= 0.50 else ("Moderate Risk" if fraud_prob >= 0.15 else "Low Risk")

        reasons = []
        for d in risk_drivers[:3]:
            f_name = d["feature"].replace("_", " ").title()
            reasons.append(f"{f_name} (impact: +{d['shap_value']:.3f})")

        driver_text = "; ".join(reasons) if reasons else "No dominant elevated risk anomalies detected."

        narrative = (
            f"Transaction evaluated as {risk_tier} (Fraud Probability: {fraud_prob * 100:.1f}%). "
            f"Primary factors elevating risk: {driver_text}. "
            f"Baseline population risk is {base_val * 100:.1f}%."
        )

        return {
            "fraud_probability": round(fraud_prob, 4),
            "risk_tier": risk_tier,
            "base_value": base_val,
            "risk_drivers": risk_drivers,
            "mitigating_factors": mitigating,
            "human_readable_summary": narrative,
            "legal_disclaimer": (
                "Notice: Model explanations represent statistical feature attributions and do not "
                "constitute legal proof or definitive evidence of fraudulent or criminal activity."
            ),
        }

    def plot_feature_importance_bar(
        self,
        X_sample: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        top_n: int = 10,
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        Plots horizontal bar chart of top global features by mean absolute SHAP value.
        """
        df = self.get_global_feature_importance(X_sample).head(top_n)

        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(
            data=df,
            x="Mean_Absolute_SHAP",
            y="Feature",
            ax=ax,
            palette="Blues_r",
            hue="Feature",
            legend=False,
        )

        ax.set_title(f"Top {top_n} Features Influencing Fraud Predictions (Global SHAP)", fontweight="bold", pad=12)
        ax.set_xlabel("Mean Absolute SHAP Value (Average Impact on Model Output)", fontweight="bold")
        ax.set_ylabel("Engineered Feature", fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        # Annotate percentages
        for p, pct in zip(ax.patches, df["Relative_Importance_Pct"]):
            width = p.get_width()
            ax.annotate(
                f" {pct:.1f}%",
                xy=(width, p.get_y() + p.get_height() / 2),
                xytext=(3, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        return fig

    def plot_transaction_waterfall(
        self,
        transaction: Union[pd.DataFrame, pd.Series, Dict[str, Any]],
        top_n: int = 8,
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        Plots a horizontal waterfall attribution chart for a single transaction.
        Red bars indicate features increasing fraud risk; Blue bars decrease risk.
        """
        explanation = self.explain_transaction(transaction, top_n=top_n)
        prob = explanation["fraud_probability"]

        all_factors = explanation["risk_drivers"] + explanation["mitigating_factors"]
        all_factors = sorted(all_factors, key=lambda x: abs(x["shap_value"]), reverse=True)[:top_n]

        features = [f["feature"] for f in all_factors]
        shap_vals = [f["shap_value"] for f in all_factors]
        colors = ["#dc2626" if s > 0 else "#2563eb" for s in shap_vals]

        fig, ax = plt.subplots(figsize=(9, 5))
        y_pos = np.arange(len(features))

        ax.barh(y_pos, shap_vals, color=colors, edgecolor="black", linewidth=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()

        ax.axvline(0, color="#475569", linestyle="-", linewidth=1)
        ax.set_title(f"Transaction Attribution Waterfall (Risk Score: {prob*100:.1f}%)", fontweight="bold", pad=12)
        ax.set_xlabel("SHAP Contribution (Red: Increases Risk, Blue: Decreases Risk)", fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        return fig

    def plot_summary(
        self,
        X_sample: Union[pd.DataFrame, np.ndarray],
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        Plots SHAP beeswarm / distribution plot if SHAP is available;
        otherwise plots feature variance-impact distributions.
        """
        if HAS_SHAP and self.tree_explainer is not None:
            try:
                fig = plt.figure(figsize=(10, 6))
                shap_values = self.tree_explainer(X_sample)
                # Class 1 (Fraud)
                if len(shap_values.shape) == 3:
                    shap_values = shap_values[:, :, 1]
                shap.plots.beeswarm(shap_values, show=False, max_display=12)
                plt.title("SHAP Beeswarm Summary Plot (Feature Value Impact)", fontweight="bold", pad=12)
                plt.tight_layout()
                if save_path:
                    plt.savefig(save_path, dpi=200, bbox_inches="tight")
                return fig
            except Exception:
                pass

        # Fallback: Clean feature impact distribution plot
        return self.plot_feature_importance_bar(X_sample, save_path=save_path)
