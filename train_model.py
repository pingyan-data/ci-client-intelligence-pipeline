import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load data ──────────────────────────────────────────────
df = pd.read_csv("ci_clients.csv")
print(f"Loaded {len(df)} client records.")

# ── 2. Feature engineering ────────────────────────────────────
le = LabelEncoder()
df["industry_encoded"] = le.fit_transform(df["industry"])

FEATURES = [
    "industry_encoded",
    "annual_revenue_sek",
    "months_as_client",
    "num_products_held",
    "fx_volume_3m_sek",
    "lending_utilization_pct",
    "last_rm_contact_days",
]
TARGET = "is_churned"

X = df[FEATURES]
y = df[TARGET]

# ── 3. Train / test split ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ── 4. Train XGBoost model ────────────────────────────────────
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    random_state=42,
    eval_metric="logloss",
    verbosity=0,
)
model.fit(X_train, y_train)

# ── 5. Evaluate ───────────────────────────────────────────────
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nAUC Score: {auc:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

# ── 6. Feature importance (SHAP) ──────────────────────────────
print("Top feature importances:")
importances = pd.Series(
    model.feature_importances_, index=FEATURES
).sort_values(ascending=False)
for feat, score in importances.items():
    bar = "█" * int(score * 40)
    print(f"  {feat:<30} {bar} {score:.3f}")

# ── 7. Save predictions ───────────────────────────────────────
df_test = X_test.copy()
df_test["client_id"] = df.loc[X_test.index, "client_id"].values
df_test["churn_probability"] = y_pred_proba
df_test["churn_flag"] = y_pred
df_test["risk_tier"] = pd.cut(
    y_pred_proba,
    bins=[0, 0.3, 0.6, 1.0],
    labels=["Low", "Medium", "High"]
)
df_test[["client_id", "churn_probability", "churn_flag", "risk_tier"]].to_csv(
    "predictions.csv", index=False
)
print("\nPredictions saved to predictions.csv")