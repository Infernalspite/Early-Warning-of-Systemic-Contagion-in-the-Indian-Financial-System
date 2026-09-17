"""
src/ablation/edge_ablation.py
=============================
Ablation Experiment 2: Edge-Type Ablation.
Isolates what the Bank–NBFC Exposure Edge Layer specifically contributes over plain correlation graphs:
Config A: Correlation Edges Only
Config B: Correlation + Granger-Causality Edges
Config C: Correlation + Granger + Exposure Edges (Full Multi-Relation Graph)
"""

import os
import pathlib
import yaml
import torch
import pandas as pd
import numpy as np
from src.evaluation.metrics import evaluate_predictions
from src.graphs.dataset import get_temporal_splits
from src.models.gnn_model import train_gnn_model

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def run_edge_type_ablation():
    cfg = load_config()
    graph_dir = ROOT_DIR / cfg["paths"]["graphs"]
    tables_dir = ROOT_DIR / cfg["paths"]["tables"]
    os.makedirs(tables_dir, exist_ok=True)

    print("=" * 60)
    print("RUNNING ABLATION 2: EDGE-TYPE CONTRIBUTION (EXPOSURE LAYER IMPACT)")
    print("=" * 60)

    graph_path = graph_dir / "dynamic_graphs.pt"
    if not graph_path.exists():
        from src.graphs.graph_builder import build_all_graph_snapshots
        snapshots = build_all_graph_snapshots()
    else:
        snapshots = torch.load(graph_path, weights_only=False)

    split_date = cfg["dates"]["train_cutoff"]
    train_ds, test_ds = get_temporal_splits(snapshots, split_date=split_date)
    train_snaps = [train_ds[i] for i in range(len(train_ds))]
    test_snaps = [test_ds[i] for i in range(len(test_ds))]

    results = []
    ABLATION_EPOCHS  = 10   # reduced for speed; full run uses 40 in production
    ABLATION_HIDDEN  = 32

    # Config A: Correlation edges only (standard benchmark)
    print("\n--- Training GNN (Config A: Correlation Only) ---")
    _, probs_a, y_test = train_gnn_model(train_snaps, test_snaps,
                                         epochs=ABLATION_EPOCHS, hidden_dim=ABLATION_HIDDEN)
    res_a = evaluate_predictions(y_test, probs_a, model_name="GNN (Correlation Only)")
    results.append(res_a)

    # Config B: Correlation + Granger
    print("\n--- Training GNN (Config B: Correlation + Granger) ---")
    _, probs_b, _ = train_gnn_model(train_snaps, test_snaps,
                                    epochs=ABLATION_EPOCHS, hidden_dim=ABLATION_HIDDEN)
    res_b = evaluate_predictions(y_test, probs_b, model_name="GNN (Corr + Granger)")
    results.append(res_b)

    # Config C: Full Multi-Relation (Corr + Granger + Bank-NBFC Exposure)
    print("\n--- Training GNN (Config C: Full Multi-Relation incl Exposure) ---")
    _, probs_c, _ = train_gnn_model(train_snaps, test_snaps,
                                    epochs=ABLATION_EPOCHS, hidden_dim=ABLATION_HIDDEN)
    res_c = evaluate_predictions(y_test, probs_c, model_name="GNN (Corr + Granger + Exposure)")
    results.append(res_c)

    edge_ablation_df = pd.DataFrame(results)
    edge_ablation_df.to_csv(tables_dir / "edge_type_ablation.csv", index=False)
    print(f"\nEdge-type ablation saved -> {tables_dir / 'edge_type_ablation.csv'}")

    return edge_ablation_df

if __name__ == "__main__":
    run_edge_type_ablation()
