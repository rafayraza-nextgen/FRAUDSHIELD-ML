"""
FastAPI REST Service for F.R.A.U.D.S.H.I.E.L.D.
Provides real-time single transaction scoring, bulk CSV batch inference,
SHAP explainability attributions, model metadata, and health checks.
"""

from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
import io
import time
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.models.pipeline import (
    FraudPredictionPipeline,
    TransactionInput,
    PredictionOutput,
    BatchPredictionOutput,
    RiskTier,
)
from src.config import MODELS_DIR, REPORTS_DIR


# ---------------------------------------------------------------------------
# Lifespan Context Manager (Preload Model Singleton)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes and pre-warms the singleton inference pipeline on startup."""
    try:
        pipeline = FraudPredictionPipeline.get_instance()
        app.state.pipeline = pipeline
    except Exception as exc:
        app.state.pipeline = None
        # Still let app start up so /health can report status
        print(f"[WARNING] Pipeline initialization deferred or failed: {exc}")
    yield


# ---------------------------------------------------------------------------
# FastAPI Application Definition
# ---------------------------------------------------------------------------
app = FastAPI(
    title="F.R.A.U.D.S.H.I.E.L.D. REST API",
    description=(
        "Financial Risk Assessment & Unauthorized Data Shield — "
        "Production-grade ML engine for real-time fraud detection, "
        "batch inference, and SHAP explainability."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Cross-Origin Resource Sharing (CORS) Middleware
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permits Next.js dev server (localhost:3000) and production origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------------------------
class SystemHealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool
    model_type: str
    optimal_threshold: float
    timestamp: float
    version: str = "1.0.0"


class ModelInfoResponse(BaseModel):
    model_type: str
    optimal_threshold: float
    feature_names: List[str]
    training_metadata: Dict[str, Any]
    benchmark_metrics: Optional[Dict[str, Any]] = None


class BatchJsonRequest(BaseModel):
    transactions: List[TransactionInput] = Field(..., min_length=1, max_length=5000)


class ExplainRequest(TransactionInput):
    """Inherits TransactionInput attributes for SHAP feature attribution."""
    pass


class ExplainResponse(BaseModel):
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    risk_score: float
    base_value: float
    prediction_value: float
    risk_drivers: List[Dict[str, Any]] = []
    mitigating_factors: List[Dict[str, Any]] = []
    top_contributing_features: List[Dict[str, Any]] = []
    explanation_narrative: str
    latency_ms: float


# ---------------------------------------------------------------------------
# Helper to Retrieve Loaded Pipeline
# ---------------------------------------------------------------------------
def get_pipeline() -> FraudPredictionPipeline:
    """Retrieves active FraudPredictionPipeline singleton or raises 503."""
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is None:
        try:
            pipeline = FraudPredictionPipeline.get_instance()
            app.state.pipeline = pipeline
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Machine Learning pipeline is unavailable: {exc}",
            )
    return pipeline


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["General"])
def read_root() -> Dict[str, Any]:
    """Root endpoint welcoming clients and directing them to OpenAPI documentation."""
    return {
        "system": "F.R.A.U.D.S.H.I.E.L.D.",
        "description": "Financial Risk Assessment & Unauthorized Data Shield REST API",
        "version": "1.0.0",
        "status": "online",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
    }


@app.get("/health", response_model=SystemHealthResponse, tags=["Monitoring"])
def health_check() -> SystemHealthResponse:
    """Healthcheck endpoint for monitoring cluster health, uptime, and model status."""
    pipeline = getattr(app.state, "pipeline", None)
    is_loaded = pipeline is not None and pipeline.model is not None
    model_name = pipeline.metadata.get("model_type", "Unknown") if pipeline else "None"
    threshold = pipeline.optimal_threshold if pipeline else 0.15

    return SystemHealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        model_type=model_name,
        optimal_threshold=round(threshold, 4),
        timestamp=time.time(),
        version="1.0.0",
    )


@app.get("/models/info", response_model=ModelInfoResponse, tags=["Model Registry"])
def get_model_info() -> ModelInfoResponse:
    """Returns details on the current champion model, features, and evaluation benchmarks."""
    pipeline = get_pipeline()
    meta = pipeline.metadata

    # Check for baseline benchmark comparison table
    benchmarks = None
    comparison_csv = REPORTS_DIR / "model_comparison.csv"
    if comparison_csv.exists():
        try:
            df_comp = pd.read_csv(comparison_csv)
            benchmarks = df_comp.to_dict(orient="records")
        except Exception:
            benchmarks = None

    return ModelInfoResponse(
        model_type=meta.get("model_type", "RandomForestClassifier"),
        optimal_threshold=round(pipeline.optimal_threshold, 4),
        feature_names=meta.get("feature_names", []),
        training_metadata={
            "trained_at": meta.get("timestamp", "N/A"),
            "n_features": meta.get("n_features", len(meta.get("feature_names", []))),
            "hyperparameters": meta.get("hyperparameters", {}),
            "cross_val_strategy": "Stratified Time-Aware Split",
        },
        benchmark_metrics={"models": benchmarks} if benchmarks else None,
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Inference"])
def predict_single_transaction(transaction: TransactionInput) -> PredictionOutput:
    """
    Real-time fraud scoring for an individual financial transaction.
    Returns calibrated fraud probability, 0-100 risk score, risk tier, and action recommendation.
    """
    pipeline = get_pipeline()
    try:
        result = pipeline.predict_single(transaction)
        return result
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Inference processing error: {str(exc)}",
        )


@app.post("/predict/batch", response_model=BatchPredictionOutput, tags=["Inference"])
def predict_batch_json(payload: BatchJsonRequest) -> BatchPredictionOutput:
    """
    Batch fraud inference via JSON payload (up to 5,000 transactions per request).
    """
    pipeline = get_pipeline()
    try:
        # Convert list of Pydantic models to DataFrame
        records = [tx.model_dump() for tx in payload.transactions]
        df = pd.DataFrame(records)
        return pipeline.predict_batch(df)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Batch JSON inference error: {str(exc)}",
        )


@app.post("/predict/batch/csv", response_model=BatchPredictionOutput, tags=["Inference"])
async def predict_batch_csv(file: UploadFile = File(...)) -> BatchPredictionOutput:
    """
    Batch fraud inference via CSV file upload (e.g. PaySim or standard transaction logs).
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Uploaded file must be a .csv file.",
        )

    pipeline = get_pipeline()
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        if df.empty:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded CSV file is empty.",
            )
        return pipeline.predict_batch(df)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to process CSV file: {str(exc)}",
        )


@app.post("/explain", response_model=ExplainResponse, tags=["Explainability"])
def explain_transaction(payload: ExplainRequest) -> ExplainResponse:
    """
    Provides granular Explainable AI (XAI) feature attribution via SHAP values
    explaining why a specific transaction was flagged or approved.
    """
    pipeline = get_pipeline()
    try:
        start_time = time.perf_counter()
        pred_res = pipeline.predict_single(payload)
        expl = pred_res.explanation
        risk_drivers = expl.get("risk_drivers", [])
        mitigating = expl.get("mitigating_factors", [])
        top_contrib = risk_drivers + mitigating

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return ExplainResponse(
            transaction_id=pred_res.transaction_id or "tx_unknown",
            is_fraud=pred_res.is_fraud,
            fraud_probability=pred_res.fraud_probability,
            risk_score=pred_res.risk_score,
            base_value=float(expl.get("base_value", 0.05)),
            prediction_value=float(expl.get("fraud_probability", pred_res.fraud_probability)),
            risk_drivers=risk_drivers,
            mitigating_factors=mitigating,
            top_contributing_features=top_contrib,
            explanation_narrative=expl.get("human_readable_summary", ""),
            latency_ms=duration_ms,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Explainability generation error: {str(exc)}",
        )
