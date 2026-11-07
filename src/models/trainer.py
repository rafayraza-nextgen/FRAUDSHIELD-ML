"""
Model training and baseline management for F.R.A.U.D.S.H.I.E.L.D.
Implements the 6 baseline machine learning algorithms with timing, probability calibration,
and hardware-conscious execution.
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier

from src.config import MODELS_DIR, RANDOM_STATE


def get_baseline_models(random_state: int = RANDOM_STATE) -> Dict[str, Any]:
    """
    Returns the dictionary of the 6 required baseline classifiers.
    For SVM, we wrap LinearSVC with CalibratedClassifierCV to enable predict_proba()
    and maintain fast linear convergence suitable for tabular financial data.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            solver="lbfgs",
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=14,
            min_samples_split=15,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": CalibratedClassifierCV(
            estimator=LinearSVC(dual=False, max_iter=2000, random_state=random_state),
            method="sigmoid",
            cv=3,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5,
            weights="uniform",
            n_jobs=-1,
        ),
    }


class BaselineModelTrainer:
    """
    Manages training, execution timing, inference, and serialization
    of the 6 required baseline models.
    """

    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state
        self.models: Dict[str, Any] = get_baseline_models(random_state=random_state)
        self.trained_models: Dict[str, Any] = {}
        self.training_times: Dict[str, float] = {}

    def train_model(
        self,
        name: str,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
    ) -> Any:
        """
        Trains a single baseline model by name and records training duration in seconds.
        """
        if name not in self.models:
            raise ValueError(f"Model '{name}' not recognized. Available: {list(self.models.keys())}")

        model = self.models[name]
        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        duration = time.perf_counter() - start_time

        self.trained_models[name] = model
        self.training_times[name] = round(duration, 4)
        return model

    def train_all_baselines(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        max_samples_for_knn: int = 50000,
    ) -> Dict[str, float]:
        """
        Trains all 6 baseline models.
        Applies computational safeguards for algorithms with high complexity:
        - KNN: O(N * D) search complexity; subsamples if training records exceed max_samples_for_knn.
        """
        for name in list(self.models.keys()):
            # Computational guard for KNN on large datasets
            if name == "K-Nearest Neighbors" and len(X_train) > max_samples_for_knn:
                print(f"Subsampling {max_samples_for_knn:,} records for KNN to respect 16 GB hardware limits...")
                if isinstance(X_train, pd.DataFrame):
                    idx = np.random.RandomState(self.random_state).choice(len(X_train), max_samples_for_knn, replace=False)
                    X_knn = X_train.iloc[idx]
                    y_knn = y_train.iloc[idx] if hasattr(y_train, "iloc") else y_train[idx]
                else:
                    idx = np.random.RandomState(self.random_state).choice(len(X_train), max_samples_for_knn, replace=False)
                    X_knn = X_train[idx]
                    y_knn = y_train[idx]
                self.train_model(name, X_knn, y_knn)
            else:
                self.train_model(name, X_train, y_train)

        return self.training_times

    def predict(
        self,
        name: str,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Runs inference for a given model.
        Returns:
            (predictions, probabilities_class_1, inference_time_seconds)
        """
        if name not in self.trained_models:
            raise RuntimeError(f"Model '{name}' has not been trained yet.")

        model = self.trained_models[name]
        start_time = time.perf_counter()
        y_pred = model.predict(X)

        # Compute probability of class 1 (Fraud)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            decision = model.decision_function(X)
            # Sigmoid transformation
            y_prob = 1.0 / (1.0 + np.exp(-decision))
        else:
            y_prob = y_pred.astype(float)

        duration = time.perf_counter() - start_time
        return y_pred, y_prob, round(duration, 4)

    def save_model(self, name: str, filepath: Optional[Union[str, Path]] = None) -> Path:
        """
        Serializes a trained model artifact using joblib.
        """
        if name not in self.trained_models:
            raise RuntimeError(f"Cannot save untrained model '{name}'.")

        safe_name = name.lower().replace(" ", "_")
        target_path = Path(filepath) if filepath else MODELS_DIR / f"{safe_name}_baseline.joblib"
        target_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.trained_models[name], target_path)
        return target_path

    @staticmethod
    def load_model(filepath: Union[str, Path]) -> Any:
        """
        Deserializes a model artifact from disk.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}")
        return joblib.load(path)
