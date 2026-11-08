# F.R.A.U.D.S.H.I.E.L.D. — System Architecture & Engineering Blueprint

**F.R.A.U.D.S.H.I.E.L.D.** (*Financial Risk Assessment & Unauthorized Data Shield*) is an enterprise-grade, end-to-end Machine Learning fraud detection platform engineered to score financial transactions in real time with calibrated risk indexing and Game-Theoretic Explainable AI (TreeSHAP).

---

## 1. High-Level System Architecture

```
   +-------------------------------------------------------------------------------+
   |                             Next.js 14 Web Client                             |
   |              (React 18 + TypeScript + Tailwind CSS + Lucide Icons)            |
   |  - Light / Dark Cyber-Fintech Themes (FOUC-free localStorage persistence)     |
   |  - Real-Time Transaction Predictor with Interactive Simulation Presets        |
   |  - High-Throughput Drag-and-Drop Batch CSV Auditing & Scored Export           |
   |  - Game-Theoretic SHAP Attribution Viewer & Multi-Model Benchmark Explorer    |
   +---------------------------------------+---------------------------------------+
                                           |
                                  HTTP REST / JSON (CORS)
                                           |
                                           v
   +-------------------------------------------------------------------------------+
   |                           FastAPI REST Service Layer                          |
   |                                 (`api/main.py`)                               |
   |  - Async Lifespan Context Manager (pre-warmed singleton model caching)        |
   |  - Pydantic v2 Schema Enforcement & Boundary Validation                       |
   |  - Endpoints:                                                                 |
   |    * GET  /           -> System greeting, version, docs discovery             |
   |    * GET  /health     -> Liveness & model readiness check                     |
   |    * GET  /models/info-> Champion metadata & benchmark metrics comparison     |
   |    * POST /predict    -> Real-time single transaction scoring                 |
   |    * POST /predict/batch     -> Bulk JSON inference (up to 5,000 records)     |
   |    * POST /predict/batch/csv -> Multipart CSV file batch streaming            |
   |    * POST /explain    -> Granular SHAP feature attribution & risk narrative   |
   +---------------------------------------+---------------------------------------+
                                           |
                             In-Process Python ML Engine
                                           |
                                           v
   +-------------------------------------------------------------------------------+
   |                       Core Machine Learning Pipeline                          |
   |                         (`src/models/pipeline.py`)                            |
   |                                                                               |
   |  +-----------------------+   +-----------------------+   +------------------+ |
   |  | Preprocessing Layer   |   | Champion Inference    |   | Explainability   | |
   |  | (`preprocessor.py`)   |-->| (`RandomForest`)      |-->| (`explainer.py`) | |
   |  | - Balance Discrepancy |   | - Calibrated Risk     |   | - TreeSHAP       | |
   |  |   Engineering         |   |   Score (0 - 100)     |   | - Risk Drivers   | |
   |  | - Leakage Drop        |   | - Threshold Applied   |   | - Mitigating     | |
   |  |   (`isFlaggedFraud`)  |   |   ($t^* = 0.09$)      |   |   Factors        | |
   |  | - RobustScaler & OHE  |   | - Risk Tier & Action  |   | - Human Narrative| |
   |  +-----------------------+   +-----------------------+   +------------------+ |
   +---------------------------------------+---------------------------------------+
                                           |
                                 Serialized Artifacts
                                           |
                                           v
   +-------------------------------------------------------------------------------+
   |                             Storage & Artifacts                               |
   |  - `models/fraud_detection_champion.joblib`: Serialized Champion RF Model     |
   |  - `models/preprocessor.joblib`: Fitted Preprocessing Pipeline & Schemas      |
   |  - `models/model_metadata.json`: Hyperparameters, Thresholds & Metrics        |
   |  - `reports/model_comparison.csv`: Benchmark Evaluation Performance Matrix    |
   |  - `reports/figures/*.png`: Visual Diagnostic Figures & SHAP Waterfalls       |
   +-------------------------------------------------------------------------------+
```

---

## 2. Component Breakdown

### A. Preprocessing & Leakage Prevention Pipeline (`src/data/preprocessor.py`)
- **Strict Isolation**: Fits scalers and encoders exclusively on the training split (`X_train`), guaranteeing zero data leakage to test or inference sets.
- **Engineered Financial Features**:
  1. `errorBalanceOrig`: Theoretical source balance discrepancy:
     $$\text{errorBalanceOrig} = \text{newbalanceOrig} + \text{amount} - \text{oldbalanceOrg}$$
  2. `errorBalanceDest`: Theoretical destination balance discrepancy:
     $$\text{errorBalanceDest} = \text{oldbalanceDest} + \text{amount} - \text{newbalanceDest}$$
  3. `isDestMerchant`: Binary flag derived from destination account prefix (`M` prefix vs. `C` customer prefix).
- **Leakage Guards**: Automatically removes `isFlaggedFraud` (simulation heuristic rule) and non-generalizable customer identifiers (`nameOrig`, `nameDest`).
- **Robust Scaling**: Implements `RobustScaler` (median & interquartile range) instead of `StandardScaler` to prevent multi-million-dollar financial outliers from skewing normalization.

### B. Machine Learning Optimization (`src/models/optimizer.py`)
- **Severe Class Imbalance Handling**: In financial fraud where fraud represents only ~0.13% of transactions, default decision thresholds ($0.50$) result in $0\%$ recall.
- **Threshold Calibration**: Evaluates the Precision-Recall curve to compute the optimal decision boundary:
  $$t^* = \arg\max_t F_1(t) \approx 0.09$$
- **Ensemble Champion**: Random Forest Classifier configured with `class_weight='balanced_subsample'` and optimized tree parameters, achieving **0.9942 PR-AUC** and **85.42% Recall** on held-out test data.

### C. Calibrated Risk Scoring (`src/models/pipeline.py`)
Rather than surfacing raw uncalibrated probabilities directly to analysts, F.R.A.U.D.S.H.I.E.L.D. normalizes risk scores onto a standardized $0 - 100$ operational scale anchored around the calibrated decision threshold $t^*$:
- **Below Threshold ($p < t^*$):**
  $$\text{Score} = \left(\frac{p}{t^*}\right) \times 50.0$$
  Categorized as **`LOW` Risk Tier** (Auto-Approve).
- **Above Threshold ($p \ge t^*$):**
  $$\text{Score} = 50.0 + \left(\frac{p - t^*}{1.0 - t^*}\right) \times 50.0$$
  - $50.0 \le \text{Score} < 75.0$: **`MODERATE` Risk Tier** (Step-Up 2FA Challenge).
  - $\text{Score} \ge 75.0$: **`HIGH` Risk Tier** (Decline / Freeze Account).

### D. Explainable AI Engine (`src/explainability/explainer.py`)
- Powered by `shap.TreeExplainer` using game-theoretic Shapley values.
- Deconstructs individual transaction predictions into:
  - **Risk Drivers**: Features pushing the prediction toward fraud (positive SHAP).
  - **Mitigating Factors**: Features suppressing fraud likelihood (negative SHAP).
  - **Plain-English Explanations**: Automated risk narrative generated for non-technical fraud investigators.

### E. Next.js 14 Frontend (`frontend/`)
- Cyber-fintech aesthetic featuring Trust Blue (`#2563EB`), Cyber Cyan (`#06B6D4`), and Alert Crimson (`#EF4444`).
- Real-time client-side calculation of balance error features as the user types.
- Drag-and-drop batch processing supporting large CSV files with immediate client-side summary metrics.
