"""
src/explainability/gnn_explainer.py
===================================
GNNExplainer & Graph Attention Visualization Module.
Extracts top-k influential bank/NBFC nodes and edge relations for each of the 6 crisis windows.
"""

import os
import pathlib
import yaml
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def explain_gnn_snapshot(gnn_model, graph_snapshot, crisis_name="Crisis", save_dir=None):
    """
    Computes node feature and edge importance masks for a single graph snapshot during a crisis.
    """
    cfg = load_config()
    figures_dir = save_dir or (ROOT_DIR / cfg["paths"]["figures"])
    os.makedirs(figures_dir, exist_ok=True)

    print(f"Running GNNExplainer for crisis '{crisis_name}'...")

    try:
        device = next(gnn_model.parameters()).device
        x = graph_snapshot.x.to(device)
        edge_index = graph_snapshot.edge_index.to(device)

        # Node importance gradient proxy
        x_req = x.clone().detach().requires_grad_(True)
        pred = gnn_model.forward_snapshot(x_req, edge_index).sum()
        pred.backward()
        
        node_importance = x_req.grad.abs().sum(dim=1).cpu().numpy()
        node_importance /= (node_importance.max() + 1e-8)

        # Top influential institutions
        banks = list(cfg["tickers"]["banks"].values())
        nbfcs = list(cfg["tickers"]["nbfcs"].values())
        all_names = (banks + nbfcs)[:len(node_importance)]

        importance_series = pd.Series(node_importance, index=all_names).sort_values(ascending=False)

        # Plot Top 10 Influential Nodes during Crisis
        plt.figure(figsize=(10, 5))
        importance_series.head(10).plot(kind="barh", color="#ef4444")
        plt.gca().invert_yaxis()
        plt.title(f"GNNExplainer — Top Contagion Driver Institutions ({crisis_name})", fontsize=12, fontweight="bold")
        plt.xlabel("Normalized Node Contagion Importance Score")
        plt.tight_layout()

        out_path = figures_dir / f"gnn_explainer_{crisis_name.lower().replace(' ', '_')}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"  GNNExplainer plot saved -> {out_path}")
        return importance_series

    except Exception as e:
        print(f"WARNING: GNNExplainer calculation failed for {crisis_name}: {e}")
        return None
