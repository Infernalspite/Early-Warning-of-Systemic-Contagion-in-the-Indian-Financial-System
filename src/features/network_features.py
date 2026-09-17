"""
src/features/network_features.py
================================
Computes graph centrality and network topology features over rolling windows:
1. Degree Centrality
2. Betweenness Centrality
3. Clustering Coefficient
4. PageRank Centrality (Chaturvedi & Singh 2022)
5. Network Density (thresholded correlation graph)
6. Absorption Ratio (PCA PC1 variance ratio)
"""

import os
import pathlib
import yaml
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.decomposition import PCA

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def compute_network_features(returns_df, window=30, corr_threshold=0.6):
    """
    Computes daily network centrality features across rolling correlation graphs.
    """
    cfg = load_config()
    dates = returns_df.index
    n_days = len(returns_df)

    mean_degree = []
    mean_betweenness = []
    clustering_coeff = []
    pagerank_mean = []
    network_density = []
    absorption_ratio = []

    for i in range(n_days):
        if i < window:
            mean_degree.append(0.0)
            mean_betweenness.append(0.0)
            clustering_coeff.append(0.0)
            pagerank_mean.append(0.0)
            network_density.append(0.0)
            absorption_ratio.append(0.0)
            continue

        sub_ret = returns_df.iloc[i-window:i].fillna(0)
        corr_matrix = sub_ret.corr().abs().fillna(0).values

        # 1. Absorption ratio (PCA)
        try:
            pca = PCA(n_components=1)
            pca.fit(sub_ret)
            abs_ratio = float(pca.explained_variance_ratio_[0])
        except Exception:
            abs_ratio = 0.0
        absorption_ratio.append(abs_ratio)

        # 2. Network Graph Construction
        adj = (corr_matrix > corr_threshold).astype(int)
        np.fill_diagonal(adj, 0)
        
        G = nx.from_numpy_array(adj)

        if G.number_of_nodes() > 0:
            deg = list(nx.degree_centrality(G).values())
            bet = list(nx.betweenness_centrality(G).values())
            clust = nx.average_clustering(G)
            try:
                pr = list(nx.pagerank(G, max_iter=100).values())
            except Exception:
                pr = [0.0]*len(G)
            dens = nx.density(G)
        else:
            deg, bet, clust, pr, dens = [0.0], [0.0], 0.0, [0.0], 0.0

        mean_degree.append(float(np.mean(deg)))
        mean_betweenness.append(float(np.mean(bet)))
        clustering_coeff.append(float(clust))
        pagerank_mean.append(float(np.mean(pr)))
        network_density.append(float(dens))

    net_df = pd.DataFrame({
        "mean_degree_centrality": mean_degree,
        "mean_betweenness_centrality": mean_betweenness,
        "clustering_coefficient": clustering_coeff,
        "pagerank_mean": pagerank_mean,
        "network_density_06": network_density,
        "absorption_ratio": absorption_ratio
    }, index=dates)

    net_df.index.name = "Date"
    return net_df

if __name__ == "__main__":
    cfg = load_config()
    proc_dir = ROOT_DIR / cfg["paths"]["data_processed"]
    ret_path = proc_dir / "bank_returns_nse.csv"
    if ret_path.exists():
        rets = pd.read_csv(ret_path, index_col=0, parse_dates=True)
        net_df = compute_network_features(rets.iloc[:100])
        print(f"Network features shape: {net_df.shape}")
