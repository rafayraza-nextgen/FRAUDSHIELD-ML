# PaySim Dataset — Data Dictionary & Column Guide

## 1. Overview
The **PaySim Synthetic Financial Dataset** simulates mobile money transactions based on a sample of real transactions from a financial service implemented in an African country. It was generated using the **PaySim** simulator to address the lack of publicly available real financial transaction data due to confidentiality.

- **Primary Source**: [Kaggle - PaySim Synthetic Financial Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1)
- **Total Records**: 6,362,620 transactions
- **Target Variable**: `isFraud` (Binary classification: `0` = Legitimate, `1` = Fraudulent)
- **Domain**: Mobile Money (e.g., transfers, cash-outs, cash-ins, merchant payments, debits)

---

## 2. Detailed Column Descriptions

| Column Name | Data Type | Description | Machine Learning & Leakage Considerations |
| :--- | :--- | :--- | :--- |
| `step` | Integer | Maps 1 unit of time to 1 hour. Total simulation lasts 744 steps (~30 days). | Useful for analyzing cyclical temporal patterns (time-of-day, day-of-month). |
| `type` | Categorical | Type of transaction (`CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, `TRANSFER`). | Critical feature. Fraud in PaySim is predominantly concentrated in `TRANSFER` and `CASH_OUT`. |
| `amount` | Float | The transaction amount in the mobile money currency. | Numerical feature; high skewness; transactions with very large amounts are often correlated with fraud. |
| `nameOrig` | String | Unique customer identifier of the transaction sender (e.g., `C123456789`). | High-cardinality ID. Not directly used as a raw numerical feature; can be used for behavioral aggregation. |
| `oldbalanceOrg` | Float | Sender's account balance before the transaction took place. | Key financial feature. Fraudsters often empty accounts completely (`newbalanceOrig` drops to `0`). |
| `newbalanceOrig` | Float | Sender's account balance after the transaction took place. | Should mathematically correlate with `oldbalanceOrg - amount`, except in errors or fraud. |
| `nameDest` | String | Unique recipient identifier. Starts with `C` (Customer) or `M` (Merchant). | Merchant recipients (`M...`) never suffer fraud in this simulation. |
| `oldbalanceDest` | Float | Recipient's account balance before the transaction took place. | May be `0` for merchants or external accounts. |
| `newbalanceDest` | Float | Recipient's account balance after the transaction took place. | Can reveal if money was actually credited or funneled out. |
| `isFraud` | Integer (`0` or `1`) | **Target variable.** `1` indicates transactions deemed fraudulent by the simulator rules. | Ground truth label. Class is heavily imbalanced (~0.13% fraud). |
| `isFlaggedFraud` | Integer (`0` or `1`) | Business rule flag: attempts to flag unauthorized transactions > 200,000 currency units. | **POTENTIAL LEAKAGE / HEURISTIC FLAG**: Must NOT be fed directly as a predictor without careful validation. |

---

## 3. Data Integrity & Reality Nuances

1. **Synthetic Nature**: PaySim is a simulation. It exhibits specific rule-based behaviors that may differ from modern banking rails (e.g. credit card 3D Secure).
2. **Extreme Class Imbalance**: Out of 6.36 million transactions, only 8,213 (~0.129%) are fraudulent. Naive models predicting all `0`s achieve 99.87% accuracy while missing 100% of fraud. Precision-Recall curves and PR-AUC are required.
3. **Missing / Unrecorded Balances**: Noticeably, merchant destination accounts (`nameDest` starting with `M`) frequently have `0.0` recorded for both `oldbalanceDest` and `newbalanceDest`.
