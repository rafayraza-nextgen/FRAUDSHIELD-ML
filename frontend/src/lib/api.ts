/**
 * Type-safe client for the F.R.A.U.D.S.H.I.E.L.D. FastAPI backend.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export type TransactionType = "PAYMENT" | "TRANSFER" | "CASH_OUT" | "DEBIT" | "CASH_IN";
export type RiskTier = "LOW" | "MODERATE" | "HIGH";

export interface TransactionPayload {
  step: number;
  type: TransactionType;
  amount: number;
  oldbalanceOrg: number;
  newbalanceOrig: number;
  oldbalanceDest: number;
  newbalanceDest: number;
  nameOrig?: string;
  nameDest?: string;
}

export interface ShapFeature {
  feature: string;
  shap_value: number;
  feature_value: number | string;
}

export interface ExplanationData {
  fraud_probability: number;
  risk_tier: string;
  base_value: number;
  risk_drivers: ShapFeature[];
  mitigating_factors: ShapFeature[];
  human_readable_summary: string;
  legal_disclaimer: string;
}

export interface PredictionResult {
  transaction_id?: string;
  is_fraud: boolean;
  fraud_probability: number;
  risk_score: number;
  risk_tier: RiskTier;
  recommendation: string;
  threshold_applied: number;
  explanation: ExplanationData;
  latency_ms: number;
}

export interface BatchPredictionResult {
  total_transactions: number;
  fraud_detected_count: number;
  fraud_rate_pct: number;
  high_risk_count: number;
  moderate_risk_count: number;
  low_risk_count: number;
  processing_time_sec: number;
  predictions: PredictionResult[];
}

export interface SystemHealth {
  status: string;
  model_loaded: boolean;
  model_type: string;
  optimal_threshold: number;
  timestamp: number;
  version: string;
}

export interface ModelInfo {
  model_type: string;
  optimal_threshold: number;
  feature_names: string[];
  training_metadata: Record<string, any>;
  benchmark_metrics?: {
    models: Array<{
      model: string;
      precision: number;
      recall: number;
      f1_score: number;
      roc_auc: number;
      pr_auc: number;
      fit_time_seconds: number;
    }>;
  };
}

export async function fetchHealth(): Promise<SystemHealth> {
  const res = await fetch(`${API_BASE}/health`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}

export async function fetchModelInfo(): Promise<ModelInfo> {
  const res = await fetch(`${API_BASE}/models/info`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch model info: ${res.statusText}`);
  return res.json();
}

export async function scoreSingleTransaction(payload: TransactionPayload): Promise<PredictionResult> {
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Prediction failed: ${res.statusText}`);
  }
  return res.json();
}

export async function uploadBatchCsv(file: File): Promise<BatchPredictionResult> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/predict/batch/csv`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Batch CSV scoring failed: ${res.statusText}`);
  }
  return res.json();
}
