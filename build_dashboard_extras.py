"""
build_dashboard_extras.py
==========================
Computes the extra artifacts the dashboard needs beyond the plain
accuracy/F1/AUC table: confusion matrices, per-model probability
histograms, a feature correlation matrix, and feature importances
for every model that can produce one. Reuses the same X_test/y_test
construction as evaluate_all_models.py so nothing here can silently
drift from the benchmark table.
Writes: web/data/dashboard_extras.json
"""
import os, json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

features = pd.read_csv("data/processed/features_india.csv", index_col=0, parse_dates=True)
features.index = pd.to_datetime(features.index, dayfirst=True)
features = features.sort_index()

with open("data/processed/feature_list.txt") as f:
    FEATURE_COLS = [l.strip() for l in f if l.strip()]
FEATURE_COLS = [c for c in FEATURE_COLS if c in features.columns]

TARGET = "high_stress_next_30d"
X = features[FEATURE_COLS].copy()
y = features[TARGET].copy()
mask = X.notna().all(axis=1) & y.notna()
X, y = X[mask], y[mask]

split_date = pd.Timestamp("2022-01-01")
X_test = X[X.index >= split_date]
y_test = y[y.index >= split_date]

out = {"confusion": {}, "prob_hist": {}, "feature_importance": {}, "correlation": {}}

def cm_block(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

def hist_block(y_true, y_prob, bins=10):
    edges = np.linspace(0, 1, bins+1)
    h_neg, _ = np.histogram(np.array(y_prob)[np.array(y_true)==0], bins=edges)
    h_pos, _ = np.histogram(np.array(y_prob)[np.array(y_true)==1], bins=edges)
    return {"edges": edges.tolist(), "neg": h_neg.tolist(), "pos": h_pos.tolist()}

# ---------- Logistic Regression ----------
try:
    with open("models/logistic_regression.pkl","rb") as f: lr = pickle.load(f)
    with open("models/scaler_lr.pkl","rb") as f: sc = pickle.load(f)
    Xs = sc.transform(X_test)
    pred = lr.predict(Xs); prob = lr.predict_proba(Xs)[:,1]
    out["confusion"]["Logistic Regression"] = cm_block(y_test, pred)
    out["prob_hist"]["Logistic Regression"] = hist_block(y_test, prob)
    coefs = lr.coef_[0]
    order = np.argsort(-np.abs(coefs))[:8]
    out["feature_importance"]["Logistic Regression"] = [
        {"feature": FEATURE_COLS[i], "importance": float(abs(coefs[i])), "sign": "+" if coefs[i] >= 0 else "-"}
        for i in order
    ]
except Exception as e:
    print("LR skip:", e)

# ---------- Random Forest ----------
try:
    with open("models/random_forest_binary.pkl","rb") as f: rf = pickle.load(f)
    pred = rf.predict(X_test); prob = rf.predict_proba(X_test)[:,1]
    out["confusion"]["Random Forest"] = cm_block(y_test, pred)
    out["prob_hist"]["Random Forest"] = hist_block(y_test, prob)
    imp = rf.feature_importances_
    order = np.argsort(-imp)[:8]
    out["feature_importance"]["Random Forest"] = [
        {"feature": FEATURE_COLS[i], "importance": float(imp[i])} for i in order
    ]
except Exception as e:
    print("RF skip:", e)

# ---------- XGBoost ----------
try:
    with open("models/xgboost.pkl","rb") as f: xgb = pickle.load(f)
    pred = xgb.predict(X_test); prob = xgb.predict_proba(X_test)[:,1]
    out["confusion"]["XGBoost"] = cm_block(y_test, pred)
    out["prob_hist"]["XGBoost"] = hist_block(y_test, prob)
    imp = xgb.feature_importances_
    order = np.argsort(-imp)[:8]
    out["feature_importance"]["XGBoost"] = [
        {"feature": FEATURE_COLS[i], "importance": float(imp[i])} for i in order
    ]
except Exception as e:
    print("XGB skip:", e)

# ---------- LSTM ----------
try:
    import torch, torch.nn as nn
    with open("models/scaler_lstm.pkl","rb") as f: lstm_scaler = pickle.load(f)
    SEQ_LEN = 30; N_FEATURES = len(FEATURE_COLS)
    class BankingLSTM(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_size, 1)
        def forward(self, x):
            out,_ = self.lstm(x)
            return torch.sigmoid(self.fc(out[:,-1,:])).squeeze(-1)
    m = BankingLSTM(N_FEATURES,64,2,0.3)
    m.load_state_dict(torch.load("models/lstm_model.pt", map_location="cpu"))
    m.eval()
    X_all = features[FEATURE_COLS].ffill().bfill().fillna(0)
    y_all = features[TARGET].fillna(0)
    common = X_all.index.intersection(y_all.index)
    X_all, y_all = X_all.reindex(common), y_all.reindex(common)
    seqs, labs = [], []
    for i in range(SEQ_LEN, len(X_all)):
        if X_all.index[i] < split_date: continue
        w = X_all.iloc[i-SEQ_LEN:i].values.astype(np.float32)
        seqs.append(lstm_scaler.transform(w)); labs.append(int(y_all.iloc[i]))
    Xt = torch.tensor(np.array(seqs), dtype=torch.float)
    with torch.no_grad():
        prob = m(Xt).numpy()
    pred = (prob >= 0.5).astype(int)
    out["confusion"]["LSTM"] = cm_block(labs, pred)
    out["prob_hist"]["LSTM"] = hist_block(labs, prob)
except Exception as e:
    print("LSTM skip:", e)

# ---------- GNN (metrics already final, no per-sample access needed) ----------
try:
    with open("models/gnn_metrics.json") as f:
        gm = json.load(f)
    out["confusion"]["GNN (GraphSAGE)"] = None  # not exposed by training script
except Exception as e:
    print("GNN skip:", e)

# ---------- Correlation matrix (key features, for the network/correlation view) ----------
key_feats = ["avg_pairwise_correlation","network_density_06","absorption_ratio",
             "covar_system","srisk_proxy","mes_avg","india_vix","inr_usd",
             "rbi_repo_rate","granger_count","avg_volatility_30d","nifty_bank_drawdown_30d"]
key_feats = [k for k in key_feats if k in features.columns]
corr = features[key_feats].corr().round(3)
out["correlation"] = {"features": key_feats, "matrix": corr.values.tolist()}

os.makedirs("web/data", exist_ok=True)
with open("web/data/dashboard_extras.json","w") as f:
    json.dump(out, f, indent=2)
print("Saved web/data/dashboard_extras.json")
print("Keys:", list(out.keys()))
for k in ["confusion","feature_importance"]:
    print(k, "->", list(out[k].keys()))
