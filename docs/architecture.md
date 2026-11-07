# F.R.A.U.D.S.H.I.E.L.D. — System Architecture

**F.R.A.U.D.S.H.I.E.L.D.** (*Financial Risk Assessment & Unauthorized Data Shield*) is an end-to-end Machine Learning platform engineered to detect potentially fraudulent financial transactions across multiple heterogeneous transaction datasets.

```
                           +--------------------------------------+
                           |          Next.js Frontend            |
                           |   (React 19 + TS + Tailwind CSS)     |
                           |   Light / Dark Cyber-Fintech UI      |
                           +------------------+-------------------+
                                              |
                                     HTTP REST API Calls
                                              |
                                              v
                           +--------------------------------------+
                           |          FastAPI REST API            |
                           |     Pydantic Schemas & Validation    |
                           +------------------+-------------------+
                                              |
                                    Service Layer / IPC
                                              |
                                              v
      +-----------------------------------------------------------------------------------+
      |                            Core Python ML Engine                                  |
      |                                                                                   |
      |   +---------------------+   +---------------------+   +-----------------------+   |
      |   |  Dataset Adapters   |   | Preprocessing & XAI |   | Model Registry        |   |
      |   |  - PaySim Adapter   |-->| - Pipeline Builder  |-->| - Scikit-learn Models |   |
      |   |  - CreditCard Adp.  |   | - Leakage Checker   |   | - Imbalance Handlers  |   |
      |   |  - Custom CSV Adp.  |   | - SHAP Explainers   |   | - Serialized Artifacts|   |
      |   +---------------------+   +---------------------+   +-----------------------+   |
      +-----------------------------------------------------------------------------------+
                                              |
                                              v
                           +--------------------------------------+
                           |         Storage & Artifacts          |
                           |  - SQLite (Local) / Supabase-ready   |
                           |  - Joblib Models & Metadata JSONs    |
                           +--------------------------------------+
```

---

## 1. Architectural Highlights

### A. Multi-Dataset Adapter Architecture
F.R.A.U.D.S.H.I.E.L.D. is not tied exclusively to one schema. It features an extensible adapter architecture:
- **PaySim Adapter**: Handles simulated mobile-money logs (`step`, `type`, `amount`, `oldbalanceOrg`, etc.).
- **Credit Card Adapter**: Handles PCA-transformed transaction records (`Time`, `V1`-`V28`, `Amount`, `Class`).
- **Generic User CSV Adapter**: Dynamically inspects column types, handles missing values, checks class balance, and enables user-mapped target classification.
- **Strict Isolation**: A model trained on Dataset A is never applied to Dataset B. Each dataset maintains its own preprocessing pipeline, feature schema, and versioned model artifact.

### B. Machine Learning & Explainable AI (XAI)
- **Zero Data Leakage**: Preprocessing transformations are fit solely on training data splits.
- **Multi-Model Benchmark**: Logistic Regression, Decision Trees, Random Forests, Naive Bayes, Support Vector Machines (SVM), and K-Nearest Neighbors (KNN).
- **Imbalance Strategies**: Evaluates class weights, undersampling, oversampling, and SMOTE.
- **Explainability**: SHAP (SHapley Additive exPlanations) generates global feature importances and transaction-level risk attribution.

### C. Decoupled Service Architecture
1. **Frontend**: Next.js 14+ / React application with dark/light themes, real-time analytics, risk dials, and CSV batch processing.
2. **API Layer**: FastAPI asynchronous REST service providing schema validation, single prediction, and batch inference endpoints.
3. **Storage**: Local SQLite for development with pluggable Supabase (PostgreSQL) support.
