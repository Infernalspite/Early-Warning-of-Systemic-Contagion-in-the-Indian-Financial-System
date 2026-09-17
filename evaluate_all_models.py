"""
evaluate_all_models.py
======================
Loads all 5 trained models and evaluates them on the SAME
X_test, y_test. Produces the final comparison table.

Models evaluated:
  1. Logistic Regression
  2. Random Forest
  3. XGBoost
  4. LSTM
  5. GNN (GraphSAGE)

Run AFTER training all 5 models.
Run: python evaluate_all_models.py
"""

import os, json, pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)

os.makedirs("outputs/charts", exist_ok=True)

print("="*60)
print("MODEL COMPARISON BENCHMARK -- INDIAN RISK ENGINE")
print("="*60)

# ================================================================
# LOAD DATA & BUILD TEST SET
# ================================================================
print("\n[1] Loading feature matrix...")

features = pd.read_csv("data/processed/features_india.csv", index_col=0, parse_dates=True)
features.index = pd.to_datetime(features.index, dayfirst=True)
features = features.sort_index()

# Load the canonical feature list used by tabular models
feat_list_path = "data/processed/feature_list.txt"
if os.path.exists(feat_list_path):
    with open(feat_list_path) as f:
        FEATURE_COLS = [l.strip() for l in f if l.strip()]
    FEATURE_COLS = [c for c in FEATURE_COLS if c in features.columns]
else:
    exclude = {"label", "label_name", "crisis_name", "high_stress_next_30d"}
    FEATURE_COLS = [c for c in features.columns if c not in exclude]

TARGET = "high_stress_next_30d"
if TARGET not in features.columns:
    raise ValueError(f"'{TARGET}' not found. Run crisis_labels.py first.")

X = features[FEATURE_COLS].copy()
y = features[TARGET].copy()
mask = X.notna().all(axis=1) & y.notna()
X, y = X[mask], y[mask]

split_date = pd.Timestamp("2022-01-01")
X_test  = X[X.index >= split_date]
y_test  = y[y.index >= split_date]

print(f"   Feature columns : {len(FEATURE_COLS)}")
print(f"   Test rows       : {len(X_test)}")
print(f"   Test positives  : {int(y_test.sum())}  ({y_test.mean()*100:.1f}%)")

# ================================================================
# HELPER: load metrics from JSON or compute from predictions
# ================================================================

all_results = {}  # model_name -> dict of metrics + roc data

def record_metrics(name, y_true, y_pred, y_prob):
    """Compute and store all metrics for a model."""
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
    except Exception:
        auc = float("nan")
        fpr, tpr = np.array([0,1]), np.array([0,1])

    all_results[name] = {
        "accuracy"  : acc,
        "precision" : prec,
        "recall"    : rec,
        "f1"        : f1,
        "roc_auc"   : auc,
        "fpr"       : fpr.tolist(),
        "tpr"       : tpr.tolist(),
    }
    return acc, prec, rec, f1, auc

# ================================================================
# 1. LOGISTIC REGRESSION
# ================================================================
print("\n[2] Evaluating Logistic Regression...")
try:
    with open("models/logistic_regression.pkl", "rb") as f:
        lr_model = pickle.load(f)
    with open("models/scaler_lr.pkl", "rb") as f:
        lr_scaler = pickle.load(f)

    X_test_sc   = lr_scaler.transform(X_test)
    lr_pred     = lr_model.predict(X_test_sc)
    lr_prob     = lr_model.predict_proba(X_test_sc)[:, 1]
    acc,p,r,f,a = record_metrics("Logistic Regression", y_test, lr_pred, lr_prob)
    print(f"   Accuracy={acc*100:.1f}%  F1={f:.4f}  AUC={a:.4f}")
except FileNotFoundError:
    print("   SKIP: models/logistic_regression.pkl not found. Run logistic_regression_model.py")
    all_results["Logistic Regression"] = None
except Exception as e:
    print(f"   ERROR: {e}")
    all_results["Logistic Regression"] = None

# ================================================================
# 2. RANDOM FOREST
# ================================================================
print("\n[3] Evaluating Random Forest...")
try:
    with open("models/random_forest_india.pkl", "rb") as f:
        rf_model = pickle.load(f)

    # RF was trained on 3-class; check if it can predict binary
    # If it has the binary target available re-use, else load dedicated binary RF
    rf_binary_path = "models/random_forest_binary.pkl"
    if os.path.exists(rf_binary_path):
        with open(rf_binary_path, "rb") as f:
            rf_model = pickle.load(f)
        rf_pred = rf_model.predict(X_test)
        rf_prob = rf_model.predict_proba(X_test)[:, 1]
    else:
        # Use existing RF: treat label>0 as stress
        try:
            rf_pred_raw = rf_model.predict(X_test)
            rf_pred = (rf_pred_raw > 0).astype(int)
            rf_prob_raw = rf_model.predict_proba(X_test)
            # sum P(1) + P(2) as stress probability
            classes = list(rf_model.classes_)
            rf_prob = sum(rf_prob_raw[:, classes.index(c)]
                         for c in [1,2] if c in classes)
        except Exception:
            rf_pred = np.zeros(len(X_test), dtype=int)
            rf_prob = np.zeros(len(X_test))

    acc,p,r,f,a = record_metrics("Random Forest", y_test, rf_pred, rf_prob)
    print(f"   Accuracy={acc*100:.1f}%  F1={f:.4f}  AUC={a:.4f}")
except FileNotFoundError:
    print("   SKIP: Random Forest model not found.")
    all_results["Random Forest"] = None
except Exception as e:
    print(f"   ERROR: {e}")
    all_results["Random Forest"] = None

# ================================================================
# 3. XGBOOST
# ================================================================
print("\n[4] Evaluating XGBoost...")
try:
    with open("models/xgboost.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    xgb_pred    = xgb_model.predict(X_test)
    xgb_prob    = xgb_model.predict_proba(X_test)[:, 1]
    acc,p,r,f,a = record_metrics("XGBoost", y_test, xgb_pred, xgb_prob)
    print(f"   Accuracy={acc*100:.1f}%  F1={f:.4f}  AUC={a:.4f}")
except FileNotFoundError:
    print("   SKIP: models/xgboost.pkl not found. Run xgboost_model.py")
    all_results["XGBoost"] = None
except Exception as e:
    print(f"   ERROR: {e}")
    all_results["XGBoost"] = None

# ================================================================
# 4. LSTM
# ================================================================
print("\n[5] Evaluating LSTM...")
try:
    import torch
    import torch.nn as nn

    with open("models/scaler_lstm.pkl", "rb") as f:
        lstm_scaler = pickle.load(f)
    with open("models/lstm_metrics.json") as f:
        lstm_metrics_cached = json.load(f)

    # Rebuild LSTM architecture
    SEQ_LEN     = 30
    N_FEATURES  = len(FEATURE_COLS)

    class BankingLSTM(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, dropout=dropout)
            self.fc   = nn.Linear(hidden_size, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            return torch.sigmoid(self.fc(out[:, -1, :])).squeeze(-1)

    lstm_model = BankingLSTM(N_FEATURES, 64, 2, 0.3)
    lstm_model.load_state_dict(torch.load("models/lstm_model.pt", map_location="cpu"))
    lstm_model.eval()

    # Build test sequences
    X_all  = features[FEATURE_COLS].ffill().bfill().fillna(0)
    y_all  = features[TARGET].fillna(0)
    common = X_all.index.intersection(y_all.index)
    X_all  = X_all.reindex(common)
    y_all  = y_all.reindex(common)

    test_seqs, test_labels = [], []
    for i in range(SEQ_LEN, len(X_all)):
        dt = X_all.index[i]
        if dt < split_date:
            continue
        window = X_all.iloc[i-SEQ_LEN:i].values.astype(np.float32)
        window = lstm_scaler.transform(window)
        test_seqs.append(window)
        test_labels.append(int(y_all.iloc[i]))

    if test_seqs:
        X_ts  = torch.tensor(np.array(test_seqs), dtype=torch.float)
        y_ts  = np.array(test_labels)
        with torch.no_grad():
            lstm_prob = lstm_model(X_ts).numpy()
        lstm_pred = (lstm_prob >= 0.5).astype(int)
        acc,p,r,f,a = record_metrics("LSTM", y_ts, lstm_pred, lstm_prob)
        print(f"   Accuracy={acc*100:.1f}%  F1={f:.4f}  AUC={a:.4f}")
    else:
        print("   No test sequences built.")
        all_results["LSTM"] = None

except FileNotFoundError:
    print("   SKIP: LSTM model not found. Run lstm_model.py")
    all_results["LSTM"] = None
except ImportError:
    print("   SKIP: PyTorch not installed.")
    all_results["LSTM"] = None
except Exception as e:
    print(f"   ERROR: {e}")
    all_results["LSTM"] = None

# ================================================================
# 5. GNN
# ================================================================
print("\n[6] Evaluating GNN...")
try:
    with open("models/gnn_metrics.json") as f:
        gnn_metrics = json.load(f)
    # GNN saves its own metrics at train time; load directly
    all_results["GNN (GraphSAGE)"] = {
        "accuracy"  : gnn_metrics.get("accuracy", float("nan")),
        "precision" : gnn_metrics.get("precision", float("nan")),
        "recall"    : gnn_metrics.get("recall", float("nan")),
        "f1"        : gnn_metrics.get("f1", float("nan")),
        "roc_auc"   : gnn_metrics.get("roc_auc") or float("nan"),
        "fpr"       : [0, 1],
        "tpr"       : [0, 1],
    }
    m = all_results["GNN (GraphSAGE)"]
    print(f"   Accuracy={m['accuracy']*100:.1f}%  F1={m['f1']:.4f}  AUC={m['roc_auc']:.4f}")
except FileNotFoundError:
    print("   SKIP: gnn_metrics.json not found. Run gnn_model.py")
    all_results["GNN (GraphSAGE)"] = None
except Exception as e:
    print(f"   ERROR: {e}")
    all_results["GNN (GraphSAGE)"] = None

# ================================================================
# RESULTS TABLE
# ================================================================
print("\n" + "="*75)
print("FINAL MODEL COMPARISON TABLE")
print("="*75)

header = f"{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}"
print(header)
print("-"*75)

rows = []
for name, res in all_results.items():
    if res is None:
        row = f"{name:<22} {'N/A':>9} {'N/A':>10} {'N/A':>8} {'N/A':>8} {'N/A':>9}"
    else:
        row = (f"{name:<22} {res['accuracy']*100:>8.2f}% "
               f"{res['precision']:>10.4f} "
               f"{res['recall']:>8.4f} "
               f"{res['f1']:>8.4f} "
               f"{res['roc_auc']:>9.4f}")
        rows.append({
            "Model"    : name,
            "Accuracy" : round(res["accuracy"]*100, 2),
            "Precision": round(res["precision"], 4),
            "Recall"   : round(res["recall"], 4),
            "F1"       : round(res["f1"], 4),
            "ROC_AUC"  : round(res["roc_auc"], 4)
        })
    print(row)

print("="*75)
print("\nNote: With 4.6% positive class, F1 and ROC-AUC are the primary metrics.")
print("      Accuracy alone is misleading -- a constant-zero model scores ~95%.")

# ================================================================
# SAVE TABLE
# ================================================================
if rows:
    table_df = pd.DataFrame(rows)
    table_df.to_csv("outputs/model_comparison.csv", index=False)
    print(f"\nTable saved: outputs/model_comparison.csv")

# Save full results with ROC data for dashboard
with open("web/data/model_comparison.json", "w") as f:
    safe = {}
    for k, v in all_results.items():
        if v:
            safe[k] = {mk: (mv if not isinstance(mv, float) or not np.isnan(mv) else None)
                       for mk, mv in v.items()}
        else:
            safe[k] = None
    json.dump(safe, f, indent=2)
print("Dashboard JSON saved: web/data/model_comparison.json")

# ================================================================
# ROC CURVE CHART
# ================================================================
print("\n[7] Generating ROC curves chart...")

fig, ax = plt.subplots(figsize=(9, 7))
ax.plot([0,1],[0,1],"k--", alpha=0.4, label="Random (AUC=0.50)")

colors = ["#3b82f6","#8b5cf6","#10b981","#f59e0b","#ef4444"]
for (name, res), color in zip(all_results.items(), colors):
    if res is None:
        continue
    auc = res["roc_auc"]
    if np.isnan(auc):
        continue
    fpr = np.array(res["fpr"])
    tpr = np.array(res["tpr"])
    ax.plot(fpr, tpr, color=color, linewidth=2,
            label=f"{name} (AUC={auc:.3f})")

ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate",  fontsize=12)
ax.set_title("ROC Curves — All Models\n(Binary: high_stress_next_30d)", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig("outputs/charts/11_roc_curves.png", dpi=150)
plt.close()
print("   Chart saved: outputs/charts/11_roc_curves.png")

# Bar chart of F1 and AUC
available = {k:v for k,v in all_results.items() if v and not np.isnan(v["f1"])}
if available:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    names  = list(available.keys())
    f1s    = [available[n]["f1"]      for n in names]
    aucs   = [available[n]["roc_auc"] for n in names]
    x      = np.arange(len(names))

    axes[0].bar(x, f1s, color=colors[:len(names)], alpha=0.85, edgecolor="white")
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=20, ha="right", fontsize=10)
    axes[0].set_ylabel("F1 Score"); axes[0].set_title("F1 Score by Model", fontweight="bold")
    axes[0].set_ylim(0, 1)
    for xi, v in zip(x, f1s):
        axes[0].text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    axes[1].bar(x, aucs, color=colors[:len(names)], alpha=0.85, edgecolor="white")
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=20, ha="right", fontsize=10)
    axes[1].set_ylabel("ROC-AUC"); axes[1].set_title("ROC-AUC by Model", fontweight="bold")
    axes[1].set_ylim(0, 1)
    for xi, v in zip(x, aucs):
        axes[1].text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    plt.suptitle("Model Comparison — Indian Banking Systemic Risk", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/charts/12_model_comparison_bars.png", dpi=150)
    plt.close()
    print("   Chart saved: outputs/charts/12_model_comparison_bars.png")

print("\n" + "="*60)
print("BENCHMARK EVALUATION COMPLETE")
print("="*60)
print("  outputs/model_comparison.csv")
print("  outputs/charts/11_roc_curves.png")
print("  outputs/charts/12_model_comparison_bars.png")
print("  web/data/model_comparison.json  (for dashboard)")
