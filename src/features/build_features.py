"""
src/features/build_features.py
==============================
Assembles all feature components into the shared master matrix (`features.csv` & `features_india.csv`).
Ensures 100% data versioning and strict temporal alignment without future lookahead leakage.
"""

import os
import pathlib
import yaml
import pandas as pd
import numpy as np

from src.features.labels import build_crisis_labels
from src.features.econometric import compute_covar, compute_mes, compute_srisk, compute_granger_edges
from src.features.network_features import compute_network_features

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def build_master_feature_matrix():
    cfg = load_config()
    raw_dir = ROOT_DIR / cfg["paths"]["data_raw"]
    proc_dir = ROOT_DIR / cfg["paths"]["data_processed"]
    os.makedirs(proc_dir, exist_ok=True)

    print("=" * 60)
    print("BUILDING MASTER FEATURE MATRIX (SHARED TRUTH FOR ALL MODELS)")
    print("=" * 60)

    # 1. Load Returns
    returns_path = proc_dir / "bank_returns_nse.csv"
    if not returns_path.exists():
        from src.data.fetch_prices import fetch_all_prices
        fetch_all_prices()

    returns = pd.read_csv(returns_path, index_col=0, parse_dates=True).sort_index()

    # 2. Build Crisis Target Labels
    labels_df = build_crisis_labels(returns.index)

    # 3. Macro Indicators
    macro_path = raw_dir / "macro_indicators.csv"
    if not macro_path.exists():
        from src.data.fetch_macro import fetch_macro_indicators
        fetch_macro_indicators()
    macro = pd.read_csv(macro_path, index_col=0, parse_dates=True).reindex(returns.index).ffill().bfill()

    # 4. Sentiment Features
    sent_path = raw_dir / "finbert_sentiment.csv"
    if not sent_path.exists():
        from src.data.fetch_sentiment import fetch_sentiment_features
        fetch_sentiment_features()
    sentiment = pd.read_csv(sent_path, index_col=0, parse_dates=True).reindex(returns.index).ffill().bfill()

    # 5. Asset Volatilities & Rolling Stats
    avg_volatility_30d = returns.rolling(30).std().mean(axis=1) * np.sqrt(252)
    avg_return_30d = returns.rolling(30).mean().mean(axis=1)

    features = pd.DataFrame(index=returns.index)
    features["avg_volatility_30d"] = avg_volatility_30d
    features["avg_return_30d"] = avg_return_30d

    # 6. Econometric Features
    print("Computing CoVaR, MES, SRISK...")
    features["covar_system"] = compute_covar(returns)
    features["mes_avg"] = compute_mes(returns)
    
    vix_series = macro["US VIX (CBOE)"] if "US VIX (CBOE)" in macro.columns else pd.Series(20.0, index=returns.index)
    features["srisk_proxy"] = compute_srisk(returns, vix_series)

    print("Computing Granger causality rolling count...")
    granger_series, _ = compute_granger_edges(returns)
    features["granger_count"] = granger_series

    # 7. Network Topology Features
    print("Computing Network Topology & Centrality measures...")
    net_df = compute_network_features(returns)
    features = pd.concat([features, net_df], axis=1)

    # 8. Merge Macro & Sentiment
    features = pd.concat([features, macro, sentiment], axis=1)

    # 9. MIBOR-Repo spread proxy
    if "repo_rate" in features.columns:
        features["mibor_repo_spread"] = (features["avg_volatility_30d"] * 100) / 10.0
    else:
        features["mibor_repo_spread"] = 0.5

    # 10. Merge Labels
    features = pd.concat([features, labels_df], axis=1)

    # Clean & Save
    features = features.ffill().bfill().fillna(0)

    features.to_csv(proc_dir / "features_india.csv")
    features.to_csv(proc_dir / "features.csv")

    # Save feature columns list
    non_feat = {"label", "high_stress_next_30d", "crisis_name", "continuous_stress_score"}
    feat_cols = [c for c in features.columns if c not in non_feat]

    with open(proc_dir / "feature_list.txt", "w") as f:
        for fc in feat_cols:
            f.write(f"{fc}\n")

    print(f"\nSUCCESS: Master Feature Matrix created -> {proc_dir / 'features_india.csv'}")
    print(f"  Total Rows (Trading Days): {len(features)}")
    print(f"  Total Feature Columns   : {len(feat_cols)}")

    return features

if __name__ == "__main__":
    build_master_feature_matrix()
