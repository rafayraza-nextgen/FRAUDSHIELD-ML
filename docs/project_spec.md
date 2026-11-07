# F.R.A.U.D.S.H.I.E.L.D. — Master Specification & Phase Roadmap

## 1. Project Overview
- **System**: F.R.A.U.D.S.H.I.E.L.D. (Financial Risk Assessment & Unauthorized Data Shield)
- **Objective**: Build an enterprise-grade, end-to-end Machine Learning fraud detection platform capable of real-time individual and batch transaction scoring with explainability.
- **Repository**: `FRAUDSHIELD-ML`

## 2. 12-Phase Development Roadmap

| Phase | Title | Focus & Core Deliverables |
| :--- | :--- | :--- |
| **0** | **Project Initialization & Git Setup** | Directory structure, virtual environment, Git remote, `.gitignore`, licensing, baseline config. |
| **1** | **Dataset Understanding** | PaySim dataset exploration, column metadata, class distribution, data quality checks (`01_dataset_understanding.ipynb`). |
| **2** | **Data Preprocessing** | Leakage-safe transformers, train/test splitting, encoding, identifier analysis, modular pipeline (`02_data_preprocessing.ipynb`). |
| **3** | **Exploratory Data Analysis** | Statistical distributions, fraud patterns by transaction type/time, correlation heatmaps, saved figures (`03_exploratory_data_analysis.ipynb`). |
| **4** | **Baseline ML Models** | Training Logistic Regression, Decision Tree, Random Forest, Naive Bayes, SVM, KNN with hardware-conscious sampling (`04_baseline_models.ipynb`). |
| **5** | **Model Evaluation & Comparison** | Precision, Recall, F1, ROC-AUC, PR-AUC, confusion matrices, trade-off analysis (`reports/model_comparison.csv`). |
| **6** | **Class Imbalance & Optimization** | Resampling (SMOTE, undersampling), class weighting, hyperparameter search, threshold calibration (`06_imbalance_handling.ipynb`). |
| **7** | **Explainable AI (SHAP)** | Global summary plots, individual prediction waterfalls, plain-language risk attribution (`07_model_explainability.ipynb`). |
| **8** | **Production Prediction Pipeline** | Reusable prediction module in `src/`, model serialization, risk score calculation, input validation. |
| **9** | **FastAPI REST Service** | REST endpoints (`/predict`, `/predict/batch`, `/health`), Pydantic validation, interactive Swagger docs. |
| **10** | **Next.js Dashboard & Design System** | Modern fintech dashboard, Light & Dark themes, interactive charting, multi-dataset manager. |
| **11** | **Testing & Quality Assurance** | Automated Pytest suite testing preprocessing, leakage guards, inference stability, and API contracts. |
| **12** | **Documentation & Final Release** | Project report, architecture guide, verified commit history, tagged release `v1.0.0`. |
