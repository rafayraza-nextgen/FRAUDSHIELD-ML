"""
Unified production prediction pipeline for F.R.A.U.D.S.H.I.E.L.D.
Provides strict Pydantic schema validation, real-time single transaction scoring,
bulk CSV batch inference, risk tier classification, and explainability attribution.
Single source of truth used by both FastAPI and frontend interfaces.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from enum import Enum
import time
import json
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator
import joblib

from src.config import MODELS_DIR, RANDOM_STATE
from src.data.preprocessor import FraudDataPreprocessor
from src.explainability.explainer import FraudExplainer


class TransactionType(str, Enum):
    PAYMENT = "PAYMENT"
    TRANSFER = "TRANSFER"
    CASH_OUT = "CASH_OUT"
    DEBIT = "DEBIT"
    CASH_IN = "CASH_IN"


class RiskTier(str, Enum):
    LOW = "LOW"             # Risk Score < 30 (Normal transaction, auto-approve)
    MODERATE = "MODERATE"   # Risk Score 30 - 70 (Requires secondary challenge / 2FA)
    HIGH = "HIGH"           # Risk Score > 70 (Elevated fraud probability, freeze/flag)


class TransactionInput(BaseModel):
    """Schema for individual financial transaction payload validation."""
    step: int = Field(..., ge=1, le=1000, description="Simulation step / hour (1 to 744+)")
    type: TransactionType = Field(..., description="Transaction type")
    amount: float = Field(..., ge=0.0, description="Monetary amount in currency units")
    oldbalanceOrg: float = Field(..., ge=0.0, description="Sender's balance before transaction")
    newbalanceOrig: float = Field(..., ge=0.0, description="Sender's balance after transaction")
    oldbalanceDest: float = Field(..., ge=0.0, description="Recipient's balance before transaction")
    newbalanceDest: float = Field(..., ge=0.0, description="Recipient's balance after transaction")
    nameOrig: Optional[str] = Field(None, description="Sender account identifier (e.g. C1234567)")
    nameDest: Optional[str] = Field(None, description="Recipient account identifier (e.g. M1234567)")

    @field_validator("amount")
    @classmethod
    def validate_amount(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError("Transaction amount cannot be negative.")
        return v


class PredictionOutput(BaseModel):
    """Standardized response schema for real-time fraud prediction."""
    transaction_id: Optional[str] = None
    is_fraud: bool
    fraud_probability: float
    risk_score: float = Field(..., ge=0.0, le=100.0, description="Risk score on scale of 0 to 100")
    risk_tier: RiskTier
    recommendation: str
    threshold_applied: float
    explanation: Dict[str, Any]
    latency_ms: float


class BatchPredictionOutput(BaseModel):
    """Standardized summary schema for batch CSV inference."""
    total_transactions: int
    fraud_detected_count: int
    fraud_rate_pct: float
    high_risk_count: int
    moderate_risk_count: int
    low_risk_count: int
    processing_time_sec: float
    predictions: List[PredictionOutput]


class FraudPredictionPipeline:
    """
    Production-grade inference pipeline encapsulating:
    1. Preprocessing & feature engineering (FraudDataPreprocessor)
    2. Trained champion model (RandomForestClassifier)
    3. Operational metadata & optimal decision threshold
    4. SHAP explainability attribution (FraudExplainer)
    """

    _instance: Optional["FraudPredictionPipeline"] = None

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
        preprocessor_path: Optional[Union[str, Path]] = None,
    ):
        self.model_path = Path(model_path) if model_path else MODELS_DIR / "fraud_detection_champion.joblib"
        self.metadata_path = Path(metadata_path) if metadata_path else MODELS_DIR / "model_metadata.json"
        self.preprocessor_path = Path(preprocessor_path) if preprocessor_path else MODELS_DIR / "preprocessor.joblib"

        self.model: Optional[Any] = None
        self.metadata: Dict[str, Any] = {}
        self.preprocessor: Optional[FraudDataPreprocessor] = None
        self.explainer: Optional[FraudExplainer] = None
        self.optimal_threshold: float = 0.15

        self.load()

    def load(self) -> None:
        """Loads model, metadata, preprocessor, and initializes explainer."""
        # 1. Load Metadata
        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
                self.optimal_threshold = float(self.metadata.get("optimal_threshold", 0.15))
        else:
            self.metadata = {"optimal_threshold": 0.15, "model_type": "RandomForestClassifier"}
            self.optimal_threshold = 0.15

        # 2. Load Model
        if self.model_path.exists():
            self.model = joblib.load(self.model_path)
        else:
            raise FileNotFoundError(
                f"Champion model artifact not found at {self.model_path}. Run Phase 6 optimization first."
            )

        # 3. Load or Initialize Preprocessor
        if self.preprocessor_path.exists():
            self.preprocessor = FraudDataPreprocessor.load(self.preprocessor_path)
        else:
            # Fallback initialization using known feature names
            self.preprocessor = FraudDataPreprocessor()
            feature_names = self.metadata.get("feature_names", [])
            self.preprocessor.feature_names_out_ = feature_names
            self.preprocessor.is_fitted_ = True

        # 4. Initialize Explainer
        feature_names = self.metadata.get("feature_names", [])
        self.explainer = FraudExplainer(self.model, feature_names=feature_names)

    @classmethod
    def get_instance(cls) -> "FraudPredictionPipeline":
        """Singleton accessor to prevent reloading 23MB model into memory on every API call."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _determine_risk_tier(self, score: float, is_fraud: bool) -> Tuple[RiskTier, str]:
        """Categorizes transaction into operational risk tiers and provides actions."""
        if score >= 75.0:
            return RiskTier.HIGH, "DECLINE_OR_STEP_UP_VERIFICATION"
        elif score >= 50.0 or is_fraud:
            return RiskTier.MODERATE, "STEP_UP_CHALLENGE_2FA"
        return RiskTier.LOW, "AUTO_APPROVE"

    def predict_single(self, transaction: Union[Dict[str, Any], TransactionInput]) -> PredictionOutput:
        """
        Executes real-time inference on a single transaction payload.
        Returns validated PredictionOutput schema with probability, calibrated risk score, tier, and SHAP explanation.
        """
        start_time = time.perf_counter()

        # Validate with Pydantic if dictionary is passed
        if isinstance(transaction, dict):
            tx_obj = TransactionInput(**transaction)
            tx_dict = tx_obj.model_dump()
        else:
            tx_dict = transaction.model_dump()

        tx_id = tx_dict.get("nameOrig") or f"tx_{int(time.time()*1000)}"

        # Convert to single-row DataFrame for preprocessor
        tx_df = pd.DataFrame([tx_dict])
        transformed_df = self.preprocessor.transform(tx_df)

        # Predict probability
        if hasattr(self.model, "predict_proba"):
            prob = float(self.model.predict_proba(transformed_df)[0, 1])
        else:
            prob = float(self.model.predict(transformed_df)[0])

        is_fraud = bool(prob >= self.optimal_threshold)

        # Calibrated risk score (0 to 100) mapped around optimal threshold
        if prob < self.optimal_threshold:
            risk_score = (prob / max(1e-6, self.optimal_threshold)) * 50.0
        else:
            risk_score = 50.0 + ((prob - self.optimal_threshold) / max(1e-6, 1.0 - self.optimal_threshold)) * 50.0

        risk_score = round(float(np.clip(risk_score, 0.0, 100.0)), 1)
        risk_tier, recommendation = self._determine_risk_tier(risk_score, is_fraud)

        # Generate Explainability Attribution
        explanation = self.explainer.explain_transaction(transformed_df.iloc[0])

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return PredictionOutput(
            transaction_id=tx_id,
            is_fraud=is_fraud,
            fraud_probability=round(prob, 4),
            risk_score=risk_score,
            risk_tier=risk_tier,
            recommendation=recommendation,
            threshold_applied=round(self.optimal_threshold, 4),
            explanation=explanation,
            latency_ms=duration_ms,
        )

    def predict_batch(
        self,
        data: Union[pd.DataFrame, str, Path],
        batch_size: int = 1000,
    ) -> BatchPredictionOutput:
        """
        Executes high-throughput batch inference over a DataFrame or CSV file.
        """
        start_time = time.perf_counter()

        if isinstance(data, (str, Path)):
            df = pd.read_csv(data)
        else:
            df = data.copy()

        total_tx = len(df)
        if total_tx == 0:
            raise ValueError("Input batch data is empty.")

        predictions: List[PredictionOutput] = []

        # Process each row
        for idx in range(total_tx):
            row_dict = df.iloc[idx].to_dict()
            pred = self.predict_single(row_dict)
            predictions.append(pred)

        fraud_count = sum(1 for p in predictions if p.is_fraud)
        high_risk_count = sum(1 for p in predictions if p.risk_tier == RiskTier.HIGH)
        mod_risk_count = sum(1 for p in predictions if p.risk_tier == RiskTier.MODERATE)
        low_risk_count = sum(1 for p in predictions if p.risk_tier == RiskTier.LOW)

        duration_sec = round(time.perf_counter() - start_time, 3)

        return BatchPredictionOutput(
            total_transactions=total_tx,
            fraud_detected_count=fraud_count,
            fraud_rate_pct=round((fraud_count / total_tx) * 100, 3),
            high_risk_count=high_risk_count,
            moderate_risk_count=mod_risk_count,
            low_risk_count=low_risk_count,
            processing_time_sec=duration_sec,
            predictions=predictions,
        )
