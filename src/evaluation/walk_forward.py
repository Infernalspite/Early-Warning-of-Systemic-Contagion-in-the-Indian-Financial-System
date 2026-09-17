"""
src/evaluation/walk_forward.py
==============================
Walk-forward backtest evaluation across all 6 labeled crisis events:
1. 2008 GFC
2. 2013 Taper Tantrum
3. 2018 IL&FS
4. 2020 YES Bank Moratorium
5. 2020 COVID Market Crash
6. 2023 Adani / Hindenburg Crisis

Computes walk-forward metric progression (train up to crisis N, test on crisis N+1).
"""

import os
import pathlib
import yaml
import pandas as pd
import numpy as np
from src.evaluation.metrics import evaluate_predictions
from src.models.tabular_models import XGBoostModel, RandomForestModel, LogisticRegressionModel

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def run_walk_forward_backtest():
    cfg = load_config()
    proc_dir = ROOT_DIR / cfg["paths"]["data_processed"]
    tables_dir = ROOT_DIR / cfg["paths"]["tables"]
    os.makedirs(tables_dir, exist_ok=True)

    feat_path = proc_dir / "features_india.csv"
    if not feat_path.exists():
        from src.features.build_features import build_master_feature_matrix
        df = build_master_feature_matrix()
    else:
        df = pd.read_csv(feat_path, parse_dates=["Date"], index_col="Date")

    crises = cfg["crises"]
    non_feat = {"label", "high_stress_next_30d", "crisis_name", "continuous_stress_score"}
    feature_cols = [c for c in df.columns if c not in non_feat]

    results_rows = []

    print("=" * 60)
    print("RUNNING WALK-FORWARD BACKTEST ACROSS 6 HISTORICAL CRISES")
    print("=" * 60)

    for idx in range(1, len(crises)):
        train_crisis = crises[idx - 1]
        test_crisis = crises[idx]

        train_cutoff = pd.to_datetime(train_crisis["event_date"])
        test_event = pd.to_datetime(test_crisis["event_date"])
        test_start = pd.to_datetime(test_crisis["label_start"]) - pd.Timedelta(days=30)
        test_end = test_event + pd.Timedelta(days=30)

        train_df = df[df.index < train_cutoff].dropna()
        test_df = df[(df.index >= test_start) & (df.index <= test_end)].dropna()

        if len(train_df) == 0 or len(test_df) == 0:
            continue

        X_tr = train_df[feature_cols].values
        y_tr = train_df["high_stress_next_30d"].values.astype(int)
        
        X_te = test_df[feature_cols].values
        y_te = test_df["high_stress_next_30d"].values.astype(int)

        if len(np.unique(y_tr)) < 2:
            continue

        # Evaluate XGBoost on this walk-forward fold
        model = XGBoostModel()
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_te)

        res = evaluate_predictions(y_te, probs, model_name=f"WalkForward_XGB_{test_crisis['name']}")
        res["test_crisis"] = test_crisis["name"]
        res["train_until"] = str(train_cutoff.date())
        results_rows.append(res)

        print(f"Fold {idx}: Test Crisis='{test_crisis['name']:<20}' | TrainUntil={str(train_cutoff.date())} | ROC-AUC={res['roc_auc']} | PR-AUC={res['pr_auc']}")

    wf_df = pd.DataFrame(results_rows)
    wf_df.to_csv(tables_dir / "walk_forward_results.csv", index=False)
    print(f"\nWalk-forward results saved -> {tables_dir / 'walk_forward_results.csv'}")

    return wf_df

if __name__ == "__main__":
    run_walk_forward_backtest()
