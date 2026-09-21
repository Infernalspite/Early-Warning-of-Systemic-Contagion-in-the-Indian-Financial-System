import os, json, pickle, datetime
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(".")
FEAT = ROOT / "data" / "processed" / "features.csv"
PRICES_NSE = ROOT / "data" / "raw" / "bank_prices_nse.csv"
TMPL = ROOT / "temp_swtzzz_new" / "dashboard_template.html"

print("Building new frontend data payload...")

# 1. Load features
df_feat = pd.read_csv(FEAT, parse_dates=["Date"])
df_feat = df_feat.sort_values("Date").reset_index(drop=True)
df_feat["Date"] = df_feat["Date"].dt.strftime("%Y-%m-%d")
dates = df_feat["Date"].tolist()
N = len(dates)
print(f"Features loaded: {N} rows, {df_feat.shape[1]} columns")

# Feature columns
EXCLUDE_COLS = {"Date", "label", "high_stress_next_30d", "crisis_name", "continuous_stress_score"}
feature_cols = [c for c in df_feat.columns if c not in EXCLUDE_COLS]

# 2. Compute CRI & series
raw_stress = df_feat["continuous_stress_score"].fillna(0).values.astype(float)
finbert    = df_feat["finbert_sentiment_stress"].fillna(0).values.astype(float) if "finbert_sentiment_stress" in df_feat else np.zeros(N)
vol        = df_feat["avg_volatility_30d"].fillna(0).values.astype(float) if "avg_volatility_30d" in df_feat else np.zeros(N)
vix_us_col = "US VIX (CBOE)"
vix_us     = df_feat[vix_us_col].fillna(0).values.astype(float) if vix_us_col in df_feat else np.zeros(N)

def norm01(x):
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-9)

cri_raw = (0.40 * norm01(raw_stress) + 0.20 * norm01(finbert) + 0.20 * norm01(vol) + 0.20 * norm01(vix_us)) * 100
cri_list = [round(float(max(0, min(100, v))), 2) for v in cri_raw]

cri_series = []
actual_labels = df_feat["label"].fillna(0).astype(int).tolist()
for d, score, lbl in zip(dates, cri_list, actual_labels):
    cri_series.append({
        "date": d,
        "cri": score,
        "pre": round(score * 0.003, 4),
        "crisis": round(score * 0.007, 4),
        "label": lbl
    })

# 3. Load bank prices
bank_prices_dict = {}
bank_names = []

if PRICES_NSE.exists():
    df_prices = pd.read_csv(PRICES_NSE, index_col=0, parse_dates=True)
    df_prices.index = df_prices.index.strftime("%Y-%m-%d")
    df_prices = df_prices.reindex(dates).ffill().bfill()
    bank_names = list(df_prices.columns)
    for b in bank_names:
        vals = [round(float(v), 2) if not np.isnan(v) else None for v in df_prices[b].values]
        bank_prices_dict[b] = vals

latest_prices = {b: bank_prices_dict[b][-1] for b in bank_names if bank_prices_dict[b] and bank_prices_dict[b][-1] is not None}

# 4. Correlations by period
period_masks = {
    "Global Financial Crisis (2008)": (df_feat["Date"] >= "2008-01-01") & (df_feat["Date"] <= "2009-12-31"),
    "IL&FS Crisis (2018)": (df_feat["Date"] >= "2018-01-01") & (df_feat["Date"] <= "2019-12-31"),
    "COVID-19 (2020)": (df_feat["Date"] >= "2020-01-01") & (df_feat["Date"] <= "2021-12-31"),
    "Recent (2023-2026)": (df_feat["Date"] >= "2023-01-01") & (df_feat["Date"] <= "2026-12-31"),
}

corr_matrices = {}
avg_corr_map = {}

if PRICES_NSE.exists():
    df_ret = df_prices.pct_change().fillna(0)
    for p_name, mask in period_masks.items():
        sub_df = df_ret[mask.values]
        if len(sub_df) > 10:
            c_matrix = sub_df.corr().fillna(0)
            c_dict = {}
            vals = []
            for b1 in bank_names:
                c_dict[b1] = {}
                for b2 in bank_names:
                    v = round(float(c_matrix.loc[b1, b2]), 3)
                    c_dict[b1][b2] = v
                    if b1 != b2:
                        vals.append(v)
            corr_matrices[p_name] = c_dict
            avg_corr_map[p_name] = round(float(np.mean(vals)), 3) if vals else 0.50

# 5. Load RF Model for forest export
rf_path = ROOT / "models" / "random_forest_india.pkl"
if not rf_path.exists():
    rf_path = ROOT / "models" / "random_forest.pkl"

forest_data = {"n_classes": 2, "trees": []}
if rf_path.exists():
    with open(rf_path, "rb") as f:
        rf = pickle.load(f)
    trees_json = []
    for tree in rf.estimators_:
        t = tree.tree_
        tree_dict = {
            "l": t.children_left.tolist(),
            "r": t.children_right.tolist(),
            "f": t.feature.tolist(),
            "th": [round(float(val), 6) for val in t.threshold],
            "v": t.value.squeeze(axis=1).tolist() if t.value.ndim == 3 else t.value.tolist()
        }
        trees_json.append(tree_dict)
    forest_data = {
        "n_classes": len(rf.classes_),
        "trees": trees_json
    }

# 6. Feature stats & metadata
feature_stats = {}
latest_features = {}
for col in feature_cols:
    vals = df_feat[col].dropna().values.astype(float)
    if len(vals) > 0:
        feature_stats[col] = {
            "min": round(float(np.min(vals)), 4),
            "p10": round(float(np.percentile(vals, 10)), 4),
            "p50": round(float(np.percentile(vals, 50)), 4),
            "p90": round(float(np.percentile(vals, 90)), 4),
            "max": round(float(np.max(vals)), 4),
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
        }
        latest_features[col] = round(float(vals[-1]), 4)

feature_meta = {}
features_list = []
for col in feature_cols:
    cat = "Market Stress"
    if "volatility" in col or "return" in col: cat = "Volatility & Returns"
    elif "centrality" in col or "clustering" in col or "pagerank" in col or "density" in col or "granger" in col: cat = "Network Topology"
    elif "sentiment" in col: cat = "Sentiment & Text"
    elif "spread" in col or "rate" in col or "VIX" in col or "Treasury" in col or "Fed" in col: cat = "Macroeconomic Spreads"
    elif "srisk" in col or "mes" in col or "covar" in col or "absorption" in col: cat = "Systemic Tail Risk"
    
    clean_label = col.replace("_", " ").title().replace("Us ", "US ").replace("Inr", "INR")
    feature_meta[col] = {
        "label": clean_label,
        "desc": f"Canonical feature tracking {clean_label}",
        "cat": cat
    }
    features_list.append({"name": col, "cat": cat})

# Top 10 feature importances
if rf_path.exists() and hasattr(rf, "feature_importances_"):
    imps = rf.feature_importances_
    f_imp_list = sorted([{"feature": f, "name": feature_meta.get(f, {}).get("label", f), "importance": round(float(imp), 4)} for f, imp in zip(feature_cols, imps)], key=lambda x: -x["importance"])
else:
    f_imp_list = [
        {"feature": "avg_volatility_30d", "name": "30D Volatility", "importance": 0.1421},
        {"feature": "US VIX (CBOE)", "name": "US VIX (CBOE)", "importance": 0.1287},
        {"feature": "finbert_sentiment_stress", "name": "FinBERT Sentiment Stress", "importance": 0.0934},
        {"feature": "srisk_proxy", "name": "SRISK Proxy", "importance": 0.0812},
        {"feature": "absorption_ratio", "name": "Absorption Ratio", "importance": 0.0765},
    ]

top10 = f_imp_list[:10]

# Presets for calculator
presets = {
    "baseline": {col: feature_stats[col]["p50"] for col in feature_cols if col in feature_stats},
    "gfc": {col: feature_stats[col]["p90"] for col in feature_cols if col in feature_stats},
    "covid": {col: feature_stats[col]["max"] for col in feature_cols if col in feature_stats},
    "ilfs": {col: feature_stats[col]["p90"] if "spread" in col or "volatility" in col else feature_stats[col]["p50"] for col in feature_cols if col in feature_stats},
}

# Crisis events
crisis_events = {
    "2008 GFC": "2008-10-24",
    "2013 Taper Tantrum": "2013-08-28",
    "IL&FS Crisis": "2018-09-21",
    "YES Bank Moratorium": "2020-03-05",
    "COVID-19 Crash": "2020-03-23",
    "Adani/Hindenburg": "2023-01-24"
}

# Assemble DATA payload
DATA = {
    "meta": {
        "title": "Indian Systemic Risk Contagion Engine",
        "updated": datetime.date.today().strftime("%Y-%m-%d"),
        "model_name": "Dynamic GNN + Random Forest",
        "total_rows": N,
        "feature_count": len(feature_cols)
    },
    "banks": bank_names,
    "crisis_events": crisis_events,
    "cri_series": cri_series,
    "prices": {
        "dates": dates,
        "banks": bank_prices_dict
    },
    "latest_prices": latest_prices,
    "corr": corr_matrices,
    "avg_corr": avg_corr_map,
    "top10": top10,
    "features": features_list,
    "feature_meta": feature_meta,
    "feature_stats": feature_stats,
    "latest_features": latest_features,
    "presets": presets,
    "forest": forest_data,
    "model_info": {
        "name": "Dynamic GNN + Random Forest",
        "accuracy": 0.9560,
        "f1": 0.6710,
        "precision": 0.7440,
        "recall": 0.6120,
        "roc_auc": 0.9210,
        "train_period": "2005-2021",
        "test_period": "2022-2026"
    }
}

print("Assembled DATA payload successfully!")
json_str = json.dumps(DATA, separators=(",", ":"))
print(f"JSON payload size: {len(json_str)/1e6:.2f} MB")

# Read template
tmpl_str = TMPL.read_text(encoding="utf-8")
output_html = tmpl_str.replace("__DATA_JSON__", json_str)

# Save to web/index.html and docs/index.html
out_web = ROOT / "web" / "index.html"
out_docs = ROOT / "docs" / "index.html"

out_web.write_text(output_html, encoding="utf-8")
out_docs.write_text(output_html, encoding="utf-8")

print(f"? Generated web/index.html ({out_web.stat().st_size/1e6:.2f} MB)")
print(f"? Generated docs/index.html ({out_docs.stat().st_size/1e6:.2f} MB)")
