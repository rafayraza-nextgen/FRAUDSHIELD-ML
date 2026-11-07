"""
Data preprocessing, feature engineering, and leakage prevention pipeline for F.R.A.U.D.S.H.I.E.L.D.
Provides a reusable Scikit-learn compatible transformer and leakage-free splitting utilities.
"""

from typing import Tuple, List, Optional, Union, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.model_selection import train_test_split
import joblib

from src.config import RANDOM_STATE


def split_data(
    df: pd.DataFrame,
    target_col: str = "isFraud",
    test_size: float = 0.2,
    split_strategy: str = "stratified",
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split transaction dataset into train and test sets preventing data leakage.

    Args:
        df: Input raw or cleaned transaction DataFrame.
        target_col: Name of target variable column (default 'isFraud').
        test_size: Proportion of dataset to include in test split (e.g. 0.20 = 20%).
        split_strategy: 'stratified' (preserves rare fraud ratio) or 'time_based' (splits by 'step').
        random_state: Seed for reproducible random sampling.

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    if split_strategy == "stratified":
        # Stratification guarantees identical fraud proportion in train and test splits
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
    elif split_strategy == "time_based":
        if "step" not in df.columns:
            raise ValueError("Time-based splitting requires 'step' column in the dataset.")
        # Sort chronologically by simulation step
        sort_idx = df["step"].argsort()
        split_point = int(len(df) * (1.0 - test_size))
        train_idx = sort_idx.iloc[:split_point]
        test_idx = sort_idx.iloc[split_point:]

        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    else:
        raise ValueError(f"Unknown split_strategy '{split_strategy}'. Choose 'stratified' or 'time_based'.")

    return X_train, X_test, y_train, y_test


class FraudDataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Production-grade preprocessing and feature engineering pipeline.

    Key Responsibilities:
    1. Feature Engineering: Calculates balance discrepancies (errorBalanceOrig, errorBalanceDest).
    2. Identifier Handling: Strips high-cardinality IDs (nameOrig, nameDest) while extracting entity flags.
    3. Leakage Guard: Removes heuristic flags like 'isFlaggedFraud'.
    4. Categorical Encoding: One-Hot Encodes transaction 'type' handling unseen categories safely.
    5. Robust Feature Scaling: Scales numerical features using RobustScaler (resilient to extreme financial outliers).
    6. Strict Leakage Prevention: Fits exclusively on training data and transforms test/inference data.
    """

    def __init__(
        self,
        engineer_features: bool = True,
        scale_features: bool = True,
        drop_identifiers: bool = True,
        drop_leakage_features: bool = True,
    ):
        self.engineer_features = engineer_features
        self.scale_features = scale_features
        self.drop_identifiers = drop_identifiers
        self.drop_leakage_features = drop_leakage_features

        # Transformers (fit during .fit())
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.scaler = RobustScaler()

        # Schema metadata
        self.categorical_cols_: List[str] = []
        self.numerical_cols_: List[str] = []
        self.encoded_feature_names_: List[str] = []
        self.feature_names_out_: List[str] = []
        self.is_fitted_: bool = False

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derives critical financial balance discrepancy features.
        In legitimate transactions: newbalanceOrig ~= oldbalanceOrg - amount
        In fraudulent transactions: discrepancies and artificial zero balances occur frequently.
        """
        data = df.copy()

        if self.engineer_features:
            # 1. Origin Account Balance Error
            # Theoretical formula: newbalanceOrig + amount - oldbalanceOrg
            if all(col in data.columns for col in ["newbalanceOrig", "amount", "oldbalanceOrg"]):
                data["errorBalanceOrig"] = data["newbalanceOrig"] + data["amount"] - data["oldbalanceOrg"]

            # 2. Destination Account Balance Error
            # Theoretical formula: oldbalanceDest + amount - newbalanceDest
            if all(col in data.columns for col in ["oldbalanceDest", "amount", "newbalanceDest"]):
                data["errorBalanceDest"] = data["oldbalanceDest"] + data["amount"] - data["newbalanceDest"]

            # 3. Destination Entity Type Flag (Merchant vs. Customer)
            if "nameDest" in data.columns:
                data["isDestMerchant"] = data["nameDest"].astype(str).str.startswith("M").astype(int)

        # 4. Remove arbitrary identifiers that cause overfitting
        if self.drop_identifiers:
            cols_to_drop = [c for c in ["nameOrig", "nameDest"] if c in data.columns]
            data.drop(columns=cols_to_drop, inplace=True)

        # 5. Prevent data leakage by dropping heuristic simulation flags
        if self.drop_leakage_features:
            if "isFlaggedFraud" in data.columns:
                data.drop(columns=["isFlaggedFraud"], inplace=True)

        return data

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FraudDataPreprocessor":
        """
        Fits encoder and scaler on training data ONLY.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        transformed = self._engineer_features(X)

        # Robust identification of categorical and numerical columns
        self.categorical_cols_ = [
            c for c in transformed.columns
            if pd.api.types.is_object_dtype(transformed[c])
            or pd.api.types.is_string_dtype(transformed[c])
            or isinstance(transformed[c].dtype, pd.CategoricalDtype)
        ]
        self.numerical_cols_ = [
            c for c in transformed.columns
            if pd.api.types.is_numeric_dtype(transformed[c])
        ]

        # Fit Categorical Encoder
        if self.categorical_cols_:
            self.encoder.fit(transformed[self.categorical_cols_])
            self.encoded_feature_names_ = list(
                self.encoder.get_feature_names_out(self.categorical_cols_)
            )
        else:
            self.encoded_feature_names_ = []

        # Fit Numerical Scaler
        if self.scale_features and self.numerical_cols_:
            self.scaler.fit(transformed[self.numerical_cols_])

        # Establish deterministic output feature order
        self.feature_names_out_ = self.encoded_feature_names_ + self.numerical_cols_
        self.is_fitted_ = True
        return self

    def transform(self, X: Union[pd.DataFrame, Dict[str, Any]]) -> pd.DataFrame:
        """
        Transforms input data using parameters fit on training data.
        Returns a clean DataFrame with deterministic column names and ordering.
        """
        if not self.is_fitted_:
            raise RuntimeError("Preprocessor has not been fitted yet. Call fit() first.")

        if isinstance(X, dict):
            X = pd.DataFrame([X])
        elif not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        transformed = self._engineer_features(X)

        # Missing value imputation (safety guard for general multi-dataset compatibility)
        if transformed[self.numerical_cols_].isnull().any().any():
            transformed[self.numerical_cols_] = transformed[self.numerical_cols_].fillna(0.0)

        # Transform Categorical Features
        if self.categorical_cols_:
            cat_encoded = self.encoder.transform(transformed[self.categorical_cols_])
            cat_df = pd.DataFrame(
                cat_encoded,
                columns=self.encoded_feature_names_,
                index=transformed.index,
            )
        else:
            cat_df = pd.DataFrame(index=transformed.index)

        # Transform Numerical Features
        if self.scale_features and self.numerical_cols_:
            num_scaled = self.scaler.transform(transformed[self.numerical_cols_])
            num_df = pd.DataFrame(
                num_scaled,
                columns=self.numerical_cols_,
                index=transformed.index,
            )
        else:
            num_df = transformed[self.numerical_cols_].copy()

        # Concatenate in strict deterministic order
        final_df = pd.concat([cat_df, num_df], axis=1)
        final_df = final_df[self.feature_names_out_]
        return final_df

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in a single step."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self) -> List[str]:
        """Returns the list of processed feature names in deterministic order."""
        if not self.is_fitted_:
            raise RuntimeError("Preprocessor has not been fitted yet.")
        return list(self.feature_names_out_)

    def save(self, filepath: Union[str, Any]) -> None:
        """Serializes the fitted preprocessor to disk using joblib."""
        if not self.is_fitted_:
            raise RuntimeError("Cannot save an unfitted preprocessor.")
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: Union[str, Any]) -> "FraudDataPreprocessor":
        """Loads a serialized preprocessor from disk."""
        return joblib.load(filepath)
