"""
train_model.py
==============
Trains a Random Forest model to predict
Indian banking crisis periods.

Input  : features from feature_engineering.py
Output : Trained model + performance metrics

Run: python train_model.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import os
import pickle
import json

os.makedirs("models", exist_ok=True)
os.makedirs("outputs/charts", exist_ok=True)

print("="*60)
print("RANDOM FOREST MODEL -- INDIAN RISK ENGINE")
print("="*60)

# ================================================================
# LOAD FEATURES
# ================================================================

print("\n[1] Loading features...")

features = pd.read_csv(
    "data/processed/features_india.csv",
    index_col=0,
    parse_dates=True
)
features.index = pd.to_datetime(features.index, dayfirst=True)
features = features.sort_index()

print(f"    Loaded: {features.shape}")

# ================================================================
# PREPARE X AND y
# ================================================================

print("\n[2] Preparing X and y...")

FEATURE_COLS = [
    "avg_volatility_5d",
    "max_volatility_5d",
    "avg_volatility_10d",
    "max_volatility_10d",
    "avg_volatility_30d",
    "max_volatility_30d",
    "avg_return_5d",
    "min_return_5d",
    "avg_return_10d",
    "min_return_10d",
    "avg_return_30d",
    "min_return_30d",
    "avg_pairwise_correlation",
    "network_density_06",
    "absorption_ratio",
    "india_vix",
    "india_vix_change",
    "nifty_bank_return_5d",
    "nifty_bank_drawdown_30d",
    "nifty_bank_rsi",
    "nifty_bank_ema_crossover",
    "nifty_fin_return_5d",
    "nifty_it_return_5d",
    "inr_usd",
    "inr_usd_change",
    "rbi_repo_rate",
    "rbi_repo_rate_change_30d",
    "yes_bank_vol_10d",
    "yes_bank_return_5d",
    "sbi_vol_10d",
    "hdfc_vol_10d",
]

# Keep only columns that exist in the dataset
FEATURE_COLS = [c for c in FEATURE_COLS if c in features.columns]
print(f"    Using {len(FEATURE_COLS)} features")

X = features[FEATURE_COLS].copy()
y = features["label"].copy()

# Drop rows with any missing values
mask = X.notna().all(axis=1)
X    = X[mask]
y    = y[mask]

print(f"    X shape: {X.shape}")
print(f"    y shape: {y.shape}")
print(f"\n    Label distribution:")
for label, count in y.value_counts().sort_index().items():
    names = {0: "Normal", 1: "Pre-Crisis", 2: "Crisis"}
    pct   = count / len(y) * 100
    print(f"    {label} ({names[label]:10s}): {count} days ({pct:.1f}%)")

# ================================================================
# TRAIN / TEST SPLIT (time-based, NOT random)
# Train: 2014-2021  |  Test: 2022-2026
# ================================================================

print("\n[3] Splitting data (time-based)...")

split_date = "2022-01-01"

X_train = X[X.index <  split_date]
X_test  = X[X.index >= split_date]
y_train = y[y.index <  split_date]
y_test  = y[y.index >= split_date]

print(f"    Train: {len(X_train)} days  ({X_train.index[0].date()} to {X_train.index[-1].date()})")
print(f"    Test : {len(X_test)}  days  ({X_test.index[0].date()}  to {X_test.index[-1].date()})")

# ================================================================
# TRAIN RANDOM FOREST
# ================================================================

print("\n[4] Training Random Forest...")

model = RandomForestClassifier(
    n_estimators     = 200,
    max_depth        = 10,
    min_samples_leaf = 5,
    class_weight     = "balanced",
    random_state     = 42,
    n_jobs           = -1
)

model.fit(X_train, y_train)
print("    Model trained!")

# ================================================================
# EVALUATE MODEL
# ================================================================

print("\n[5] Evaluating model...")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

print("\n--- CLASSIFICATION REPORT ---")
print("-"*50)
print(classification_report(
    y_test,
    y_pred,
    target_names=["Normal", "Pre-Crisis", "Crisis"]
))

accuracy = (y_pred == y_test).mean()
print(f"Overall Accuracy: {accuracy*100:.1f}%")

# ================================================================
# FEATURE IMPORTANCE
# ================================================================

print("\n[6] Feature Importances (Top 10):")
print("-"*40)

importance_df = pd.DataFrame({
    "feature"   : FEATURE_COLS,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

for i, row in importance_df.head(10).iterrows():
    bar = "#" * int(row["importance"] * 200)
    print(f"   {row['feature']:30s} {bar} {row['importance']:.4f}")

# ================================================================
# SAVE MODEL
# ================================================================

with open("models/random_forest_india.pkl", "wb") as f:
    pickle.dump(model, f)

print(f"\nModel saved to: models/random_forest_india.pkl")

# ================================================================
# SAVE CHARTS
# ================================================================

print("\n[7] Creating evaluation charts...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Pre-Crisis", "Crisis"]
)
disp.plot(ax=axes[0], cmap="Blues", colorbar=False)
axes[0].set_title("Confusion Matrix", fontsize=12, fontweight="bold")

top10 = importance_df.head(10)
axes[1].barh(top10["feature"][::-1], top10["importance"][::-1], color="steelblue")
axes[1].set_title("Top 10 Feature Importances", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig("outputs/charts/09_model_evaluation.png", dpi=150)
plt.close()
print("    Chart saved: outputs/charts/09_model_evaluation.png")

# Predicted vs Actual chart
try:
    indices_df = pd.read_csv("data/raw/nifty_indices.csv", index_col=0, parse_dates=True)
    indices_df.index = pd.to_datetime(indices_df.index, dayfirst=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    if "Nifty Bank" in indices_df.columns:
        nifty_test = indices_df["Nifty Bank"][indices_df.index >= split_date]
        ax1.plot(nifty_test.index, nifty_test.values, color="royalblue", linewidth=1.2)
        ax1.set_ylabel("Nifty Bank")
        ax1.set_title("Actual vs Predicted Crisis Periods (Test: 2022-2026)", fontsize=12, fontweight="bold")

    for date, label in y_test.items():
        if label == 2:
            ax1.axvspan(date, date + pd.Timedelta(days=1), alpha=0.3, color="red")
        elif label == 1:
            ax1.axvspan(date, date + pd.Timedelta(days=1), alpha=0.3, color="yellow")

    pred_series = pd.Series(y_pred, index=y_test.index)
    ax2.fill_between(pred_series.index, pred_series.values, alpha=0.6, color="darkorange", step="mid")
    ax2.set_ylabel("Predicted Label\n(0=Normal, 1=Pre, 2=Crisis)")
    ax2.set_xlabel("Date")
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(["Normal", "Pre-Crisis", "Crisis"])
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    from matplotlib.patches import Patch
    ax1.legend(handles=[
        Patch(facecolor="red", alpha=0.4, label="Actual Crisis"),
        Patch(facecolor="yellow", alpha=0.5, label="Actual Pre-Crisis"),
    ], loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig("outputs/charts/10_predictions_vs_actual.png", dpi=150)
    plt.close()
    print("    Chart saved: outputs/charts/10_predictions_vs_actual.png")
except Exception as e:
    print(f"    Warning: Could not create predictions chart: {e}")

# ================================================================
# EXPORT DASHBOARD DATA JSON
# ================================================================
print("\n[8] Compiling and exporting dashboard data...")

X_all       = features[FEATURE_COLS].dropna()
X_all_clean = X_all.ffill().bfill().fillna(0)
probs       = model.predict_proba(X_all_clean)
classes     = list(model.classes_)
y_pred_all  = model.predict(X_all_clean)

pre_prob    = probs[:, classes.index(1)] if 1 in classes else np.zeros(len(probs))
crisis_prob = probs[:, classes.index(2)] if 2 in classes else probs[:, -1]
cri         = (crisis_prob * 0.7 + pre_prob * 0.3) * 100

# Bank prices
prices_df = pd.read_csv("data/raw/bank_prices_nse.csv", index_col=0, parse_dates=True)
prices_df.index = pd.to_datetime(prices_df.index, dayfirst=True)
prices_df = prices_df.sort_index().reindex(X_all.index).ffill().bfill()

prices_norm = (prices_df / prices_df.iloc[0]) * 100
prices_dict = {}
for col in prices_df.columns:
    raw_vals  = prices_df[col].round(2).tolist()
    norm_vals = prices_norm[col].round(2).tolist()
    prices_dict[col] = {"raw": raw_vals, "norm": norm_vals}

# Indices (macro)
indices_df = pd.read_csv("data/raw/nifty_indices.csv", index_col=0, parse_dates=True)
indices_df.index = pd.to_datetime(indices_df.index, dayfirst=True)
indices_df = indices_df.sort_index().reindex(X_all.index).ffill().bfill()

macro_df = pd.read_csv("data/raw/macro_indicators.csv", index_col=0, parse_dates=True)
macro_df.index = pd.to_datetime(macro_df.index, dayfirst=True)
macro_df = macro_df.sort_index().reindex(X_all.index).ffill().bfill()

# Returns (for network graph)
returns_df = pd.read_csv("data/processed/bank_returns_nse.csv", index_col=0, parse_dates=True)
returns_df.index = pd.to_datetime(returns_df.index, dayfirst=True)
returns_df = returns_df.sort_index()
bank_names = list(returns_df.columns)

# Build network history (downsample every 5 days + crisis dates)
crisis_dates_str = ["2018-09-21", "2020-03-05", "2020-03-23", "2020-11-17", "2023-01-24"]
crisis_ts = {pd.Timestamp(d) for d in crisis_dates_str}

step_indices = list(range(30, len(returns_df), 5))
dates_to_compute = sorted(
    {returns_df.index[i] for i in step_indices} | (crisis_ts & set(returns_df.index))
)

network_history = []
for dt in dates_to_compute:
    try:
        i = returns_df.index.get_loc(dt)
        if isinstance(i, slice):
            i = i.start
        if i < 30:
            continue
        window = returns_df.iloc[i-30:i].fillna(0)
        corr   = window.corr().fillna(0)

        links = []
        for ai, ba in enumerate(bank_names):
            for bi, bb in enumerate(bank_names):
                if ai < bi:
                    val = corr.iloc[ai, bi]
                    if val > 0.40:
                        links.append({
                            "source": ba,
                            "target": bb,
                            "value": round(float(val), 3)
                        })
        network_history.append({"date": dt.strftime("%Y-%m-%d"), "links": links})
    except Exception:
        continue

export_data = {
    "dates"             : [d.strftime("%Y-%m-%d") for d in X_all.index],
    "cri"               : cri.round(1).tolist(),
    "predicted_labels"  : y_pred_all.tolist(),
    "actual_labels"     : y.loc[X_all.index].tolist(),
    "bank_names"        : bank_names,
    "bank_prices"       : prices_dict,
    "macro": {
        "nifty_bank" : (indices_df["Nifty Bank"].round(2).tolist()                   if "Nifty Bank"               in indices_df else []),
        "nifty_50"   : (indices_df["Nifty 50"].round(2).tolist()                     if "Nifty 50"                 in indices_df else []),
        "india_vix"  : (indices_df["India VIX"].round(2).tolist()                    if "India VIX"                in indices_df else []),
        "nifty_fin"  : (indices_df["Nifty Financial Services"].round(2).tolist()     if "Nifty Financial Services" in indices_df else []),
        "inr_usd"    : (macro_df["INR_USD_Rate"].round(4).tolist()                   if "INR_USD_Rate"             in macro_df   else []),
        "repo_rate"  : (macro_df["RBI_Repo_Rate"].round(2).tolist()                  if "RBI_Repo_Rate"            in macro_df   else []),
    },
    "network_history"   : network_history,
    "feature_importances": [
        {"feature": f, "importance": round(float(imp), 4)}
        for f, imp in zip(FEATURE_COLS, model.feature_importances_)
    ],
    "model_metrics": {
        "accuracy"     : round(float(accuracy), 4),
        "features_used": FEATURE_COLS
    }
}

os.makedirs("web/data", exist_ok=True)
with open("web/data/dashboard_data.json", "w") as f:
    json.dump(export_data, f)

print("    Dashboard data exported to: web/data/dashboard_data.json")

# ================================================================
# SUMMARY
# ================================================================

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"  Model Accuracy  : {accuracy*100:.1f}%")
print(f"  Features Used   : {len(FEATURE_COLS)}")
print(f"  Training Days   : {len(X_train)}")
print(f"  Testing Days    : {len(X_test)}")
print(f"  Network Frames  : {len(network_history)}")
print("\n  Files saved:")
print("    models/random_forest_india.pkl")
print("    outputs/charts/09_model_evaluation.png")
print("    web/data/dashboard_data.json")
print("\nDone! Start the dashboard with:  python server.py")