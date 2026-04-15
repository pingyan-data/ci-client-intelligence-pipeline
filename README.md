# CI Client Intelligence Pipeline

End-to-end ML pipeline for predicting client churn and cross-sell
potential in Corporate & Institutions banking.

Built with Python, XGBoost, and scikit-learn — from synthetic data
generation to model training, evaluation, and prediction output.

## Background

Relationship Managers (RMs) in Corporate & Institutions banking manage
dozens of complex client relationships simultaneously. This pipeline
helps prioritise which clients need attention by scoring each client on:

- **Churn risk** — likelihood of reducing or exiting the relationship
- **Cross-sell potential** — likelihood of taking on additional products

## Project structure

    ci-client-intelligence-pipeline/
    ├── generate_ci_data.py   # Synthetic C&I client data generator
    ├── train_model.py        # XGBoost churn model training + evaluation
    ├── ci_clients.csv        # Generated dataset (500 clients)
    └── predictions.csv       # Model output with risk tiers

## Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/pingyan-data/ci-client-intelligence-pipeline.git
cd ci-client-intelligence-pipeline
python3 -m venv venv
source venv/bin/activate
pip install pandas scikit-learn xgboost shap
```

## Usage

**Step 1 — Generate synthetic client data:**

```bash
python generate_ci_data.py
```

**Step 2 — Train the churn prediction model:**

```bash
python train_model.py
```

## Key findings

The most important predictor of churn is `last_rm_contact_days` —
clients who have not been contacted recently are significantly more
likely to churn. This aligns with the principle that in C&I banking,
the strength of the RM relationship is the primary retention mechanism.

## Dataset features

| Feature | Description |
|---|---|
| `industry` | Client industry sector |
| `annual_revenue_sek` | Annual revenue in SEK |
| `months_as_client` | Length of banking relationship |
| `num_products_held` | Number of bank products held |
| `fx_volume_3m_sek` | FX transaction volume last 3 months |
| `lending_utilization_pct` | Proportion of credit facility used |
| `last_rm_contact_days` | Days since last RM contact |
| `is_churned` | Target variable (1 = churned) |
| `cross_sell_score` | Cross-sell potential score (0–1) |

## Tech stack

- Python 3.9+
- pandas, scikit-learn, XGBoost, SHAP

## License

MIT