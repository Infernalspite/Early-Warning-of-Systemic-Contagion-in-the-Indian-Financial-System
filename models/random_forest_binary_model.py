"""
Random Forest Model (binary) - Early Warning of Systemic Contagion (India)
============================================================================
Binary classification: predict high_stress_next_30d
Time-based train/test split at 2022-01-01.
Standard tabular baseline; also yields feature importances.

Run from project root:
    python models/random_forest_binary_model.py
"""

import json
import pickle
import pathlib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH   = ROOT / "data" / "processed" / "features_india.csv"
FEAT_LIST   = ROOT / "data" / "processed" / "feature_list.txt"
MODEL_OUT   = ROOT / "models" / "random_forest_binary.pkl"
METRICS_OUT = ROOT / "models" / "random_forest_metrics.json"

TARGET_COL  = "high_stress_next_30d"
SPLIT_DATE  = pd.Timestamp("2022-01-01")

print("Loading data from:", DATA_PATH)
df = pd.read_csv(DATA_PATH, parse_dates=["Date"], index_col="Date")

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

cols_needed = feature_cols + [TARGET_COL]
df = df[cols_needed].dropna()
print(f"  Rows after dropping NaN: {len(df)}")

train_df = df[df.index < SPLIT_DATE]
test_df  = df[df.index >= SPLIT_DATE]

print(f"  Train samples: {len(train_df)}  (before {SPLIT_DATE.date()})")
print(f"  Test  samples: {len(test_df)}   (from  {SPLIT_DATE.date()})")

X_train = train_df[feature_cols].values
y_train = train_df[TARGET_COL].values.astype(int)
X_test  = test_df[feature_cols].values
y_test  = test_df[TARGET_COL].values.astype(int)

print("\nTraining RandomForestClassifier ...")
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=3,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)
print("  Training complete.")

with open(MODEL_OUT, "wb") as f:
    pickle.dump(model, f)
print(f"\n  Model saved -> {MODEL_OUT}")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
try:
    auc = roc_auc_score(y_test, y_prob)
except ValueError:
    auc = float("nan")

importances = sorted(
    zip(feature_cols, model.feature_importances_),
    key=lambda x: -x[1]
)[:10]

metrics = {
    "model": "Random Forest",
    "split_date": str(SPLIT_DATE.date()),
    "train_samples": len(train_df),
    "test_samples": len(test_df),
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1": f1,
    "roc_auc": auc,
    "top_features": [{"feature": k, "importance": float(v)} for k, v in importances],
}
with open(METRICS_OUT, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"  Metrics saved -> {METRICS_OUT}")

print("\n" + "=" * 50)
print("  RANDOM FOREST - TEST SET RESULTS")
print("=" * 50)
print(f"  Split Date      {SPLIT_DATE.date()}")
print(f"  Train Samples   {len(train_df):>7,}")
print(f"  Test Samples    {len(test_df):>7,}")
print(f"  Accuracy        {acc:.4f}")
print(f"  Precision       {prec:.4f}")
print(f"  Recall          {rec:.4f}")
print(f"  F1 Score        {f1:.4f}")
print(f"  ROC-AUC         {auc:.4f}")
print("=" * 50)
