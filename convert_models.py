"""
convert_models.py
-----------------
Retrains LR, RF, XGBoost from the repo data and exports each to a pure-JSON
format that the Vercel serverless function can load with zero sklearn/xgboost
dependency at runtime.

Run once:
    python convert_models.py

Outputs (added to git):
    models/logistic_regression.json
    models/random_forest_binary.json
    models/xgboost.json
"""
import os, json, pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

ROOT      = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "data", "processed", "features_india.csv")
FEAT_LIST = os.path.join(ROOT, "data", "processed", "feature_list.txt")
MODELS    = os.path.join(ROOT, "models")

TARGET     = "high_stress_next_30d"
SPLIT_DATE = pd.Timestamp("2022-01-01")

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, parse_dates=["Date"], index_col="Date")
with open(FEAT_LIST) as f:
    feature_cols = [l.strip() for l in f if l.strip()]
feature_cols = [c for c in feature_cols if c in df.columns]

df = df[feature_cols + [TARGET]].dropna()
train = df[df.index < SPLIT_DATE]
X_train = train[feature_cols].values
y_train = train[TARGET].values.astype(int)

# ── 1. Logistic Regression + StandardScaler ───────────────────────────────
print("Training LR …")
sc = StandardScaler()
Xs = sc.fit_transform(X_train)
lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
lr.fit(Xs, y_train)

with open(os.path.join(MODELS, "logistic_regression.json"), "w") as f:
    json.dump({
        "features":     feature_cols,
        "coef":         lr.coef_[0].tolist(),
        "intercept":    float(lr.intercept_[0]),
        "scaler_mean":  sc.mean_.tolist(),
        "scaler_scale": sc.scale_.tolist(),
    }, f)
print("  → models/logistic_regression.json")

# ── 2. Random Forest ─────────────────────────────────────────────────────────
print("Training RF (300 trees) …")
rf = RandomForestClassifier(
    n_estimators=300, max_depth=8, min_samples_leaf=3,
    class_weight="balanced_subsample", random_state=42, n_jobs=-1,
)
rf.fit(X_train, y_train)

def _serialize_tree(est):
    t = est.tree_
    return [
        {"left": int(t.children_left[i]), "right": int(t.children_right[i]),
         "feature": int(t.feature[i]),    "threshold": float(t.threshold[i]),
         "value": t.value[i][0].tolist()}
        for i in range(t.node_count)
    ]

with open(os.path.join(MODELS, "random_forest_binary.json"), "w") as f:
    json.dump([_serialize_tree(e) for e in rf.estimators_], f)
print("  → models/random_forest_binary.json")

# ── 3. XGBoost ────────────────────────────────────────────────────────────────
print("Training XGBoost (300 trees) …")
n_neg, n_pos = int((y_train==0).sum()), int((y_train==1).sum())
xgb = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=n_neg/n_pos,
    use_label_encoder=False, eval_metric="logloss",
    random_state=42, verbosity=0,
)
xgb.fit(X_train, y_train)

try:
    cfg = json.loads(xgb.get_booster().save_config())
    raw = cfg["learner"]["learner_model_param"]["base_score"]
    # newer XGBoost encodes base_score as '[5E-1]'
    raw = raw.strip("[]")
    base_score = float(raw)
except Exception:
    base_score = 0.5
trees_json = [json.loads(t) for t in xgb.get_booster().get_dump(dump_format="json")]

with open(os.path.join(MODELS, "xgboost.json"), "w") as f:
    json.dump({"base_score": base_score,
               "feature_names": feature_cols,
               "trees": trees_json}, f)
print("  → models/xgboost.json")

# ── 4. Verify identical predictions ──────────────────────────────────────────
print("\nVerifying …")
test = df[df.index >= SPLIT_DATE]
X_test = test[feature_cols].values[:5]

with open(os.path.join(MODELS, "logistic_regression.json")) as f: lr_j = json.load(f)
with open(os.path.join(MODELS, "random_forest_binary.json")) as f: rf_j = json.load(f)
with open(os.path.join(MODELS, "xgboost.json")) as f: xgb_j = json.load(f)

def lr_pred(j, x):
    z = j["intercept"]
    for i, col in enumerate(j["features"]):
        idx = feature_cols.index(col)
        z += ((x[idx] - j["scaler_mean"][i]) / j["scaler_scale"][i]) * j["coef"][i]
    return 1/(1+np.exp(-z))

def rf_pred(trees, x):
    tot = None
    for nodes in trees:
        nid = 0
        while nodes[nid]["left"] != -1:
            nid = nodes[nid]["left"] if x[nodes[nid]["feature"]] <= nodes[nid]["threshold"] else nodes[nid]["right"]
        v = nodes[nid]["value"]
        tot = v if tot is None else [tot[i]+v[i] for i in range(len(v))]
    s = sum(tot); p = [v/s for v in tot]; return p[1]

def xgb_pred(j, x):
    fn = j["feature_names"]; bs = j["base_score"]
    def walk(n):
        if "leaf" in n: return n["leaf"]
        fi = fn.index(n["split"]) if n["split"] in fn else int(n["split"][1:])
        tid = n["yes"] if x[fi] < n["split_condition"] else n["no"]
        for c in n["children"]:
            if c["nodeid"]==tid: return walk(c)
    s = sum(walk(t) for t in j["trees"])
    off = np.log(bs/(1-bs)) if 0<bs<1 else 0.0
    return 1/(1+np.exp(-(off+s)))

lr_pkl_p  = lr.predict_proba(sc.transform(X_test))[:,1]   # LR needs scaled input
rf_pkl_p  = rf.predict_proba(X_test)[:,1]
xgb_pkl_p = xgb.predict_proba(X_test)[:,1]

all_ok = True
for i in range(5):
    x = X_test[i]
    lp = lr_pred(lr_j, x); rp = rf_pred(rf_j, x); xp = xgb_pred(xgb_j, x)
    ok = abs(lp-lr_pkl_p[i])<1e-6 and abs(rp-rf_pkl_p[i])<1e-6 and abs(xp-xgb_pkl_p[i])<1e-5
    if not ok: all_ok = False
    print(f"  row {i}: LR {abs(lp-lr_pkl_p[i]):.1e} | RF {abs(rp-rf_pkl_p[i]):.1e} | XGB {abs(xp-xgb_pkl_p[i]):.1e} {'✓' if ok else '✗'}")

print("\nAll predictions match!" if all_ok else "\n⚠ Mismatch found!")
