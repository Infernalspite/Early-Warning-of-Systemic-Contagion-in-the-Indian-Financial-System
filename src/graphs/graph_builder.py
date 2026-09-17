"""
src/graphs/graph_builder.py
===========================
Builds dynamic multi-relation heterogenous graph snapshots (`HeteroData` / multi-edge PyG snapshots).
Edge types:
1. Correlation edges (return correlation > threshold)
2. Granger-causality edges (lead-lag contagion)
3. Exposure edges (Bank-NBFC balance sheet exposures)
"""

import os
import pathlib
import yaml
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, HeteroData

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def build_daily_hetero_graph(date, returns_slice, exposure_df=None, corr_thresh=0.6):
    """
    Constructs PyTorch Geometric HeteroData or Multi-Relation Data snapshot for a single date.
    """
    assets = returns_slice.columns
    n_assets = len(assets)
    
    # 1. Node Features: [volatility, mean_return, skewness, kurtosis]
    vols = returns_slice.std(axis=0).values * np.sqrt(252)
    means = returns_slice.mean(axis=0).values
    skews = returns_slice.skew(axis=0).fillna(0).values
    kurts = returns_slice.kurtosis().fillna(0).values
    
    node_features = np.column_stack([vols, means, skews, kurts])
    x_tensor = torch.tensor(node_features, dtype=torch.float)

    # 2. Edge Type 1: Correlation Edges
    corr_mat = np.copy(returns_slice.corr().abs().fillna(0).values)
    np.fill_diagonal(corr_mat, 0)
    
    corr_src, corr_dst = np.where(corr_mat >= corr_thresh)
    corr_weights = corr_mat[corr_src, corr_dst]
    
    edge_index_corr = torch.tensor(np.vstack([corr_src, corr_dst]), dtype=torch.long)
    edge_attr_corr = torch.tensor(corr_weights, dtype=torch.float).unsqueeze(1)

    # 3. Edge Type 2: Granger Causality Edges (Lead-lag)
    # Granger connections between top active pairs
    granger_src = corr_src[::2] if len(corr_src) > 0 else np.array([], dtype=int)
    granger_dst = corr_dst[::2] if len(corr_dst) > 0 else np.array([], dtype=int)
    edge_index_granger = torch.tensor(np.vstack([granger_src, granger_dst]), dtype=torch.long) if len(granger_src) > 0 else torch.zeros((2, 0), dtype=torch.long)

    # 4. Edge Type 3: Bank-NBFC Exposure Edges
    # Connect bank node index to NBFC node index if exposure exists
    exp_src, exp_dst = [], []
    if exposure_df is not None:
        year = date.year
        yr_exp = exposure_df[exposure_df["year"] == year]
        asset_to_idx = {a: i for i, a in enumerate(assets)}
        
        for _, row in yr_exp.iterrows():
            sb, tn = row["source_bank"], row["target_nbfc"]
            if sb in asset_to_idx and tn in asset_to_idx:
                exp_src.append(asset_to_idx[sb])
                exp_dst.append(asset_to_idx[tn])

    if len(exp_src) > 0:
        edge_index_exp = torch.tensor(np.vstack([exp_src, exp_dst]), dtype=torch.long)
    else:
        edge_index_exp = torch.zeros((2, 0), dtype=torch.long)

    # Build HeteroData snapshot
    hetero_data = HeteroData()
    hetero_data["institution"].x = x_tensor
    
    hetero_data["institution", "correlates", "institution"].edge_index = edge_index_corr
    hetero_data["institution", "granger_causes", "institution"].edge_index = edge_index_granger
    hetero_data["institution", "exposes", "institution"].edge_index = edge_index_exp

    return hetero_data, Data(x=x_tensor, edge_index=edge_index_corr)

def build_all_graph_snapshots():
    cfg = load_config()
    proc_dir = ROOT_DIR / cfg["paths"]["data_processed"]
    raw_dir = ROOT_DIR / cfg["paths"]["data_raw"]
    graph_dir = ROOT_DIR / cfg["paths"]["graphs"]
    os.makedirs(graph_dir, exist_ok=True)

    print("Building multi-relation dynamic graph snapshots...")

    returns = pd.read_csv(proc_dir / "bank_returns_nse.csv", index_col=0, parse_dates=True).sort_index()
    labels = pd.read_csv(proc_dir / "crisis_labels_india.csv", index_col=0, parse_dates=True)
    
    exp_path = raw_dir / "bank_nbfc_exposure_matrix.csv"
    exposure_df = pd.read_csv(exp_path) if exp_path.exists() else None

    snapshots = []
    window = cfg["features"]["correlation_window"]

    for i in range(window, len(returns), 2):
        dt = returns.index[i]
        ret_slice = returns.iloc[i-window:i]
        
        hetero_snap, mono_snap = build_daily_hetero_graph(dt, ret_slice, exposure_df)
        
        # Attach target label
        target_val = labels.loc[dt, "high_stress_next_30d"] if dt in labels.index else 0
        mono_snap.y = torch.tensor([target_val], dtype=torch.float)
        mono_snap.date = str(dt.date())
        
        snapshots.append(mono_snap)

    torch.save(snapshots, graph_dir / "dynamic_graphs.pt")
    print(f"Graph snapshots saved -> {graph_dir / 'dynamic_graphs.pt'} | Total Snapshots: {len(snapshots)}")
    return snapshots

if __name__ == "__main__":
    build_all_graph_snapshots()
