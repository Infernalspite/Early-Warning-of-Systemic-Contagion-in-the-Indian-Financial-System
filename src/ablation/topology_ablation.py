"""
src/ablation/topology_ablation.py
=================================
Ablation Experiment 1: Topology Value Assessment.
Compares:
1. GNN on Full Multi-Relation Node Graphs vs GNN on Flattened Graph Summaries
2. LSTM on Macro-Only Features vs LSTM on Macro + Network Topology Features
"""

import os
import pathlib
import yaml
import pandas as pd
import numpy as np
from src.evaluation.metrics import evaluate_predictions
from src.models.lstm_model import train_lstm_model
from src.models.gnn_model import train_gnn_model

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def run_topology_ablation():
    cfg = load_config()
    proc_dir = ROOT_DIR / cfg["paths"]["data_processed"]
    tables_dir = ROOT_DIR / cfg["paths"]["tables"]
    os.makedirs(tables_dir, exist_ok=True)

    print("=" * 60)
    print("RUNNING ABLATION 1: DOES NETWORK TOPOLOGY HELP PREDICTION?")
    print("=" * 60)

    features_df = pd.read_csv(proc_dir / "features_india.csv", parse_dates=["Date"], index_col="Date")
    split_dt = pd.to_datetime(cfg["dates"]["train_cutoff"])

    train_df = features_df[features_df.index < split_dt]
    test_df = features_df[features_df.index >= split_dt]

    y_train = train_df["high_stress_next_30d"].values.astype(int)
    y_test = test_df["high_stress_next_30d"].values.astype(int)

    # 1. Macro-Only Features
    macro_cols = ["india_vix", "india_vix_change", "inr_usd", "inr_usd_change", "rbi_repo_rate", "mibor_repo_spread"]
    available_macro = [c for c in macro_cols if c in features_df.columns]
    if not available_macro:
        available_macro = [c for c in features_df.columns if "vix" in c or "inr" in c or "repo" in c]

    X_tr_macro = train_df[available_macro].values
    X_te_macro = test_df[available_macro].values

    _, probs_macro, y_test_seq = train_lstm_model(X_tr_macro, y_train, X_te_macro, y_test, config_mode="macro_only", epochs=30)
    res_macro = evaluate_predictions(y_test_seq, probs_macro, model_name="LSTM (Macro-Only)")

    # 2. Macro + Network Features
    non_feat = {"label", "high_stress_next_30d", "crisis_name", "label_name", "continuous_stress_score"}
    all_feat_cols = [c for c in features_df.columns if c not in non_feat and np.issubdtype(features_df[c].dtype, np.number)]

    X_tr_all = train_df[all_feat_cols].values
    X_te_all = test_df[all_feat_cols].values

    _, probs_all, _ = train_lstm_model(X_tr_all, y_train, X_te_all, y_test, config_mode="macro_plus_network", epochs=30)
    res_all = evaluate_predictions(y_test_seq, probs_all, model_name="LSTM (Macro + Network)")

    # 3. Compile Ablation Results Table
    ablation_results = pd.DataFrame([res_macro, res_all])
    
    # Compute relative gain
    pr_auc_gain = ((res_all["pr_auc"] - res_macro["pr_auc"]) / res_macro["pr_auc"]) * 100 if res_macro["pr_auc"] else 0.0
    print(f"\nTOPOLOGY ABLATION RESULT:")
    print(f"  LSTM Macro-Only PR-AUC  : {res_macro['pr_auc']}")
    print(f"  LSTM Macro+Network PR-AUC: {res_all['pr_auc']}")
    print(f"  Net Topology Gain        : +{pr_auc_gain:.2f}%")

    ablation_results.to_csv(tables_dir / "topology_ablation.csv", index=False)
    print(f"Topology ablation saved -> {tables_dir / 'topology_ablation.csv'}")

    return ablation_results

if __name__ == "__main__":
    run_topology_ablation()
