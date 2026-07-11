"""
Logistic Regression Model - Early Warning of Systemic Contagion (India)
========================================================================
Binary classification: predict high_stress_next_30d
Time-based train/test split at 2022-01-01.

Run from project root:
    python models/logistic_regression_model.py
"""

import os
import json
import pickle
import pathlib

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH   = ROOT / "data" / "processed" / "features_india.csv"
FEAT_LIST   = ROOT / "data" / "processed" / "feature_list.txt"
MODEL_OUT   = ROOT / "models" / "logistic_regression.pkl"
SCALER_OUT  = ROOT / "models" / "scaler_lr.pkl"
METRICS_OUT = ROOT / "models" / "logistic_regression_metrics.json"

TARGET_COL  = "high_stress_next_30d"
SPLIT_DATE  = pd.Timestamp("2022-01-01")


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("Loading data from:", DATA_PATH)
df = pd.read_csv(DATA_PATH, parse_dates=["date"], index_col="date")

# ---------------------------------------------------------------------------
# 2. Determine feature columns
# ---------------------------------------------------------------------------
if FEAT_LIST.exists():
    print("Loading feature list from:", FEAT_LIST)
    with open(FEAT_LIST, "r") as f:
        feature_cols = [line.strip() for line in f if line.strip()]
    # Keep only columns that exist in the dataframe
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"  Features loaded from file: {len(feature_cols)}")
else:
    non_feature_cols = {TARGET_COL}
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    print(f"  No feature_list.txt found. Using all {len(feature_cols)} non-label columns.")

# ---------------------------------------------------------------------------
# 3. Drop NaN rows (features + target)
# ---------------------------------------------------------------------------
cols_needed = feature_cols + [TARGET_COL]
df = df[cols_needed].dropna()
print(f"  Rows after dropping NaN: {len(df)}")

# ---------------------------------------------------------------------------
# 4. Time-based train/test split
# ---------------------------------------------------------------------------
train_df = df[df.index < SPLIT_DATE]
test_df  = df[df.index >= SPLIT_DATE]

print(f"  Train samples: {len(train_df)}  (before {SPLIT_DATE.date()})")
print(f"  Test  samples: {len(test_df)}   (from  {SPLIT_DATE.date()})")

if len(train_df) == 0 or len(test_df) == 0:
    raise ValueError("Train or test set is empty after the time split. Check SPLIT_DATE.")

X_train = train_df[feature_cols].values
y_train = train_df[TARGET_COL].values.astype(int)
X_test  = test_df[feature_cols].values
y_test  = test_df[TARGET_COL].values.astype(int)

# ---------------------------------------------------------------------------
# 5. Standardise features
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 6. Train Logistic Regression
# ---------------------------------------------------------------------------
print("\nTraining LogisticRegression ...")
model = LogisticRegression(
    class_weight="balanced",
    max_iter=2000,
    C=0.1,
    solver="lbfgs",
    random_state=42,
)
model.fit(X_train_sc, y_train)
print("  Training complete.")

# ---------------------------------------------------------------------------
# 7. Evaluate on test set
# ---------------------------------------------------------------------------
y_pred      = model.predict(X_test_sc)
y_prob      = model.predict_proba(X_test_sc)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)
roc_auc   = roc_auc_score(y_test, y_prob)

metrics = {
    "model":     "LogisticRegression",
    "split_date": str(SPLIT_DATE.date()),
    "n_train":   int(len(y_train)),
    "n_test":    int(len(y_test)),
    "accuracy":  round(accuracy,  4),
    "precision": round(precision, 4),
    "recall":    round(recall,    4),
    "f1":        round(f1,        4),
    "roc_auc":   round(roc_auc,   4),
}

# ---------------------------------------------------------------------------
# 8. Save artefacts
# ---------------------------------------------------------------------------
with open(MODEL_OUT,   "wb") as f:
    pickle.dump(model,  f)

with open(SCALER_OUT,  "wb") as f:
    pickle.dump(scaler, f)

with open(METRICS_OUT, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n  Model  saved -> {MODEL_OUT}")
print(f"  Scaler saved -> {SCALER_OUT}")
print(f"  Metrics saved -> {METRICS_OUT}")

# ---------------------------------------------------------------------------
# 9. Summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 50)
print("  LOGISTIC REGRESSION - TEST SET RESULTS")
print("=" * 50)
print(f"  {'Metric':<15} {'Value':>10}")
print(f"  {'-'*25}")
print(f"  {'Split Date':<15} {str(SPLIT_DATE.date()):>10}")
print(f"  {'Train Samples':<15} {len(y_train):>10,}")
print(f"  {'Test Samples':<15} {len(y_test):>10,}")
print(f"  {'Accuracy':<15} {accuracy:>10.4f}")
print(f"  {'Precision':<15} {precision:>10.4f}")
print(f"  {'Recall':<15} {recall:>10.4f}")
print(f"  {'F1 Score':<15} {f1:>10.4f}")
print(f"  {'ROC-AUC':<15} {roc_auc:>10.4f}")
print("=" * 50)
