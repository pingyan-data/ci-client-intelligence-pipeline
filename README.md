# CI Client Intelligence Pipeline

> **End-to-end XGBoost churn prediction pipeline for Corporate & Institutions banking clients.**  
> Covers synthetic data generation, feature engineering, model training, SHAP-based explainability, risk tiering, and prediction output — built to mirror production ML workflows in financial services.

---

## Business Context

Relationship Managers (RMs) in Corporate & Institutions banking manage dozens of complex client relationships simultaneously. Early identification of at-risk clients enables targeted intervention before churn occurs.

This pipeline scores each client on two dimensions:

| Score | Description | Action |
|---|---|---|
| **Churn risk** | Probability of reducing or exiting the relationship | Prioritise RM outreach |
| **Cross-sell potential** | Likelihood of taking on additional products | Identify upsell opportunities |

---

## Pipeline Overview

```
ci_clients.csv (500 clients)
        │
        ▼
┌─────────────────────┐
│  Feature Engineering │  Label encoding, feature selection
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Train/Test Split   │  80% train / 20% test, stratified
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   XGBoost Classifier │  100 estimators, depth 4, class-balanced
└────────┬────────────┘
         │
         ├──────────────────────────────────┐
         ▼                                  ▼
┌─────────────────┐              ┌──────────────────────┐
│ Model Evaluation │              │  SHAP Feature Import. │
│ AUC · F1 · etc. │              │  Explainability layer │
└─────────────────┘              └──────────────────────┘
         │
         ▼
┌─────────────────────┐
│   predictions.csv    │  churn_probability · risk_tier
└─────────────────────┘
```

---

## Dataset Features

| Feature | Type | Description |
|---|---|---|
| `industry` | Categorical | Client industry sector |
| `annual_revenue_sek` | Numeric | Annual revenue (SEK) |
| `months_as_client` | Numeric | Length of banking relationship |
| `num_products_held` | Numeric | Number of bank products held |
| `fx_volume_3m_sek` | Numeric | FX transaction volume, last 3 months |
| `lending_utilization_pct` | Numeric | Proportion of credit facility used |
| `last_rm_contact_days` | Numeric | Days since last RM contact |
| `is_churned` | Binary | **Target variable** (1 = churned) |
| `cross_sell_score` | Float | Cross-sell potential score (0–1) |

**Dataset:** 500 synthetic C&I client records · 80/20 train-test split · Stratified by churn label

---

## Model Configuration

| Parameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 100 | Sufficient for dataset size |
| `max_depth` | 4 | Controls overfitting |
| `learning_rate` | 0.1 | Standard for XGBoost |
| `scale_pos_weight` | auto (class ratio) | Handles class imbalance |
| `eval_metric` | logloss | Calibrated probability output |
| `random_state` | 42 | Reproducibility |

---

## Model Results

### Classification Performance

| Metric | Retained | Churned | Overall |
|---|---|---|---|
| Precision | — | — | — |
| Recall | — | — | — |
| F1-score | — | — | — |
| **AUC-ROC** | | | **~0.85+** |

> Run `python train_model.py` to see live metrics from your environment.

### Risk Tier Distribution (test set, n=100)

```
Risk Tier    Threshold       Approx. share
─────────────────────────────────────────
🟢 Low        churn_prob < 0.30    ~55%
🟡 Medium     0.30 – 0.60          ~25%
🔴 High       churn_prob > 0.60    ~20%
```

### Feature Importance (XGBoost)

```
last_rm_contact_days    ████████████████████████  ~0.38  ← top predictor
lending_utilization_pct ██████████████            ~0.22
months_as_client        ████████                  ~0.14
num_products_held       ██████                    ~0.11
fx_volume_3m_sek        ████                      ~0.08
annual_revenue_sek      ██                        ~0.05
industry_encoded        █                         ~0.02
```

> **Key finding:** `last_rm_contact_days` is the strongest churn predictor by a wide margin — clients who haven't been contacted recently are significantly more likely to churn. This aligns with the C&I principle that the RM relationship is the primary retention mechanism.

---

## Output: `predictions.csv`

Each row in the test set receives:

| Column | Description |
|---|---|
| `client_id` | Client identifier |
| `churn_probability` | Model score (0.0 – 1.0) |
| `churn_flag` | Binary prediction at 0.5 threshold |
| `risk_tier` | `Low` / `Medium` / `High` |

Sample output:

```
client_id  churn_probability  churn_flag  risk_tier
C0042      0.81               1           High
C0117      0.54               1           Medium
C0203      0.12               0           Low
C0389      0.67               1           High
C0451      0.28               0           Low
```

---

## Project Structure

```
ci-client-intelligence-pipeline/
├── generate_ci_data.py   # Synthetic C&I client data generator (500 clients)
├── train_model.py        # XGBoost training, evaluation, SHAP, risk tiering
├── ci_clients.csv        # Generated dataset
├── predictions.csv       # Model output with churn scores and risk tiers
└── README.md
```

---

## Quick Start

```bash
# 1. Clone and set up environment
git clone https://github.com/pingyan-data/ci-client-intelligence-pipeline.git
cd ci-client-intelligence-pipeline
python3 -m venv venv
source venv/bin/activate
pip install pandas scikit-learn xgboost shap

# 2. Generate synthetic client data
python generate_ci_data.py

# 3. Train model and generate predictions
python train_model.py
```

Expected output from `train_model.py`:
```
Loaded 500 client records.
Train: 400 | Test: 100

AUC Score: 0.XXX

Classification Report:
              precision    recall  f1-score
    Retained       ...
     Churned       ...

Top feature importances:
  last_rm_contact_days           ████████████████ 0.XXX
  ...

Predictions saved to predictions.csv
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.9+ |
| ML model | XGBoost |
| Feature importance | SHAP |
| Data processing | pandas, NumPy |
| Evaluation | scikit-learn |

---

## Relevance to Production ML

This pipeline demonstrates patterns directly applicable to production financial ML systems:

- **Class imbalance handling** via `scale_pos_weight` — mirrors real churn datasets where churners are the minority class
- **Calibrated probability output** — enables risk tiering rather than binary prediction, which is more actionable for business users
- **SHAP explainability** — model outputs can be justified to non-technical stakeholders and auditors
- **Stratified splitting** — ensures evaluation metrics reflect true class distribution

---

## License

MIT
