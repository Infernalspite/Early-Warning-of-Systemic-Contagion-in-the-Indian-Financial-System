"""
XGBoost Model - Early Warning of Systemic Contagion (India)
============================================================
Binary classification: predict high_stress_next_30d
Time-based train/test split at 2022-01-01.
Handles class imbalance via scale_pos_weight.

Run from project root:
    python models/xgboost_model.py
"""

import os
import json
import pickle
import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

try:
    from xgboost import XGBClassifier
except ImportError:
    print("XGBoost is not installed.")
    print("Install with: pip install xgboost")
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH   = ROOT / "data" / "processed" / "features_india.csv"
FEAT_LIST   = ROOT / "data" / "processed" / "feature_list.txt"
MODEL_OUT   = ROOT / "models" / "xgboost.pkl"
METRICS_OUT = ROOT / "models" / "xgboost_metrics.json"

TARGET_COL = "high_stress_next_30d"
SPLIT_DATE = pd.Timestamp("2022-01-01")
TOP_N_FEAT = 10


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("Loading data from:", DATA_PATH)
df = pd.read_csv(DATA_PATH, parse_dates=["Date"], index_col="Date")

# ---------------------------------------------------------------------------
# 2. Determine feature columns
# ---------------------------------------------------------------------------
if FEAT_LIST.exists():
    print("Loading feature list from:", FEAT_LIST)
    with open(FEAT_LIST, "r") as f:
        feature_cols = [line.strip() for line in f if line.strip()]
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"  Features loaded from file: {len(feature_cols)}")
else:
    non_feature_cols = {TARGET_COL}
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    print(f"  No feature_list.txt found. Using all {len(feature_cols)} non-label columns.")

# ---------------------------------------------------------------------------
# 3. Drop NaN rows
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
# 5. Compute scale_pos_weight for class imbalance
# ---------------------------------------------------------------------------
n_neg = int(np.sum(y_train == 0))
n_pos = int(np.sum(y_train == 1))
if n_pos == 0:
    raise ValueError("No positive samples in the training set. Cannot compute scale_pos_weight.")
scale_pos_weight = n_neg / n_pos
print(f"\n  Class distribution (train): neg={n_neg}, pos={n_pos}")
print(f"  scale_pos_weight = {scale_pos_weight:.4f}")

# ---------------------------------------------------------------------------
# 6. Train XGBClassifier
# ---------------------------------------------------------------------------
print("\nTraining XGBClassifier ...")
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    verbosity=0,
)
model.fit(X_train, y_train)
print("  Training complete.")

# ---------------------------------------------------------------------------
# 7. Evaluate on test set
# ---------------------------------------------------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)
roc_auc   = roc_auc_score(y_test, y_prob)

metrics = {
    "model":            "XGBClassifier",
    "split_date":       str(SPLIT_DATE.date()),
    "n_train":          int(len(y_train)),
    "n_test":           int(len(y_test)),
    "scale_pos_weight": round(scale_pos_weight, 4),
    "accuracy":         round(accuracy,  4),
    "precision":        round(precision, 4),
    "recall":           round(recall,    4),
    "f1":               round(f1,        4),
    "roc_auc":          round(roc_auc,   4),
}

# ---------------------------------------------------------------------------
# 8. Save artefacts
# ---------------------------------------------------------------------------
with open(MODEL_OUT,   "wb") as f:
    pickle.dump(model, f)

with open(METRICS_OUT, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n  Model   saved -> {MODEL_OUT}")
print(f"  Metrics saved -> {METRICS_OUT}")

# ---------------------------------------------------------------------------
# 9. Feature importances (top N)
# ---------------------------------------------------------------------------
importances = model.feature_importances_
feat_imp_df = pd.DataFrame({
    "feature":    feature_cols,
    "importance": importances,
}).sort_values("importance", ascending=False).reset_index(drop=True)

top_n = min(TOP_N_FEAT, len(feat_imp_df))
print(f"\n  Top {top_n} Feature Importances")
print(f"  {'Rank':<6} {'Feature':<40} {'Importance':>10}")
print(f"  {'-'*58}")
for i, row in feat_imp_df.head(top_n).iterrows():
    print(f"  {i+1:<6} {row['feature']:<40} {row['importance']:>10.6f}")

# ---------------------------------------------------------------------------
# 10. Summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 50)
print("  XGBOOST - TEST SET RESULTS")
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
