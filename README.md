# Customer Churn Prediction & Lifetime Value Modeling
 
An end-to-end customer analytics pipeline built on real e-commerce transaction data — combining behavioral segmentation, churn prediction, and probabilistic lifetime value (LTV) modeling into a single interactive dashboard.
 
 
> ⚠️ Hosted on Streamlit Community Cloud's free tier — the app may take a few seconds to wake up if it hasn't been visited recently.
 
---
 
### Dashboard Overview

![Dashboard View 1](assets/Screenshot%201.png)

![Dashboard View 2](assets/screenshot%202.png)

![Dashboard View 3](assets/Screenshot%203.png)

 
---
 
## Overview
 
This project analyzes ~540K transactions from a UK-based online retailer (Kaggle/UCI Online Retail dataset) to answer three questions a real e-commerce business would ask:
 
1. **Who are our customers, behaviorally?** — RFM segmentation
2. **Who's about to leave?** — Supervised churn prediction with explainability
3. **What is each customer actually worth?** — Probabilistic lifetime value forecasting
The result is packaged into a 3-tab Streamlit dashboard usable by a non-technical stakeholder, backed by a fully reproducible modeling pipeline.
 
## Key Differentiators
 
- **Probabilistic LTV modeling** (BG/NBD + Gamma-Gamma via the `lifetimes` library) — goes beyond simple historical averaging to forecast *future* transaction counts and monetary value per customer, with 3/6/12-month projections.
- **Multi-model clustering validation** — K-Means, Gaussian Mixture Models, and Hierarchical Clustering were all fit and compared (silhouette score, BIC/AIC, and visual cluster geometry) before selecting a production model, rather than defaulting to one algorithm.
- **Leakage-aware churn modeling** — features are reconstructed at a strict temporal cutoff (not the full dataset), and XGBoost's early stopping uses a dedicated validation split so the reported test metrics are never contaminated by model-selection decisions.
- **SHAP explainability at both the global and individual-customer level** — the dashboard doesn't just say *who* is at risk, it shows *why*, per customer, via a live SHAP waterfall plot.
## Tech Stack
 
| Category | Tools |
|---|---|
| Language | Python |
| Data Processing | pandas, NumPy |
| Machine Learning | scikit-learn, XGBoost |
| Explainability | SHAP |
| Probabilistic Modeling | `lifetimes` (BG/NBD, Gamma-Gamma) |
| Visualization | Plotly, Matplotlib, Seaborn |
| Dashboard / Deployment | Streamlit, Streamlit Community Cloud |
| Dataset | [Online Retail Dataset](https://www.kaggle.com/datasets/vijayuv/onlineretail) (Kaggle/UCI) |
 
## Project Architecture
 
```
churn-prediction-customer-lifetime-value-modeling/
├── notebooks/
│   ├── 01_data_cleaning_eda.ipynb        # Phase 1
│   ├── 02_rfm_segmentation.ipynb         # Phase 2
│   ├── 03_churn_prediction.ipynb         # Phase 3
│   └── 04_ltv_modeling.ipynb             # Phase 4
├── data/
│   ├── raw/                              # original Kaggle download (gitignored)
│   └── processed/                        # cleaned/derived outputs consumed by app.py
├── models/                               # trained model artifacts (.pkl)
├── assets/                               # dashboard screenshots for this README
├── app.py                                # Phase 5 — Streamlit dashboard
├── requirements.txt
└── README.md
```
 
## Methodology
 
### Phase 1 — Data Cleaning & Exploratory Analysis
Raw transactional data (541,909 rows) was cleaned by removing cancelled orders, duplicate records, invalid/missing values, and statistical outliers in both quantity and unit price. A monthly cohort retention analysis was built to visualize customer drop-off over the customer lifecycle.
 
### Phase 2 — RFM Behavioral Segmentation
Customers were segmented on Recency, Frequency, and Monetary value. Three clustering algorithms (K-Means, GMM, Hierarchical) were fit and evaluated head-to-head; **K-Means (k=4)** was selected as the production model based on a decisively higher silhouette score (0.331 vs. 0.233 Hierarchical vs. 0.203 GMM) and visual cluster separation, then profiled into business segments — **Champions, At-Risk, Hibernating, New Customers**.
 
### Phase 3 — Predictive Churn Modeling
Churn was defined using a 90-day inactivity window. Features were rebuilt strictly at a historical cutoff date to prevent target leakage, and class imbalance was handled via `scale_pos_weight` (XGBoost) and `class_weight='balanced'` (Logistic Regression baseline). Model evaluation used PR-AUC given the class imbalance, with a dedicated validation split isolating early-stopping decisions from the final held-out test set.
 
**Final Model Performance:** *[Insert final XGBoost PR-AUC / precision / recall from your classification report here]*
 
### Phase 4 — Probabilistic Lifetime Value Modeling
BG/NBD models expected future transaction volume; Gamma-Gamma models expected monetary value per transaction, applied only to the 63.9% of customers with repeat purchases. Both models were validated on a calibration/holdout split before being used to generate 3/6/12-month LTV forecasts per customer.
 
### Phase 5 — Interactive Dashboard
A 3-tab Streamlit application:
- **Executive Insights** — high-level KPIs and the cohort retention heatmap
- **Strategic Segments** — filterable segment explorer with targeted campaign recommendations
- **Customer Lookup** — individual customer search showing churn risk, predicted LTV, and a live SHAP waterfall explanation
## Notable Engineering Challenges
 
Building this pipeline surfaced several non-trivial issues, each diagnosed and resolved:
 
- **Silent out-of-order execution bug** — a revenue calculation silently returned `$0.00` due to stale kernel state, not a code error; resolved by enforcing clean top-to-bottom notebook execution.
- **BG/NBD parameter degeneracy** — the dropout parameters (`a`, `b`) repeatedly converged to statistically meaningless boundary values, traced to the ~36% one-time-buyer share weakly identifying the dropout component; addressed via a systematic penalizer sweep with an explicit stability check (rejecting fits with invalid confidence intervals, not just non-`NaN` standard errors).
- **Test-set leakage via early stopping** — XGBoost's early stopping was initially using the test set as its evaluation set, inflating reported metrics; fixed with a proper three-way train/validation/test split.
- **Cross-library naming collisions** — the `lifetimes` library's internal `frequency`/`recency` definitions differ entirely from the project's own RFM definitions despite sharing column names; handled by keeping these namespaces explicitly separate throughout the pipeline.
- **Deployment dependency conflicts** — resolved SHAP/XGBoost serialization incompatibilities and NumPy 2.x binary incompatibilities encountered during Streamlit Cloud deployment.
## Setup & Local Installation
 
```bash
# Clone the repository
git clone https://github.com/deepakp-tech/churn-prediction-customer-lifetime-value-modeling.git
cd churn-prediction-customer-lifetime-value-modeling
 
# Install dependencies
pip install -r requirements.txt
 
# Run the dashboard locally
streamlit run app.py
```
 
The notebooks in `notebooks/` should be run in order (01 → 04) if regenerating the processed data/model artifacts from scratch; otherwise, the pre-computed files in `data/processed/` and `models/` are sufficient to run the dashboard directly.
 
## Limitations
 
- Churn is defined via a fixed 90-day inactivity threshold — a simplification of true non-contractual churn behavior, which in reality varies by customer purchase cycle.
- Gamma-Gamma's independence assumption (spend independent of frequency) shows a mild violation in this dataset (correlation ≈ 0.17), which may introduce slight bias in LTV estimates for the highest-frequency customers.
- LTV estimates apply only to repeat customers (63.9% of the base); one-time buyers receive a transaction-count forecast but no monetary LTV, by design.
## Author
 
**Deepak**

 
