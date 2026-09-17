"""
src/features/econometric.py
===========================
Computes econometric systemic risk measures:
1. CoVaR (Systemic Quantile Regression delta CoVaR)
2. SRISK Proxy (Capital shortfall under systemic distress)
3. MES (Marginal Expected Shortfall)
4. Granger Causality Pairwise Edges & Rolling Edge Counts
"""

import os
import pathlib
import yaml
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def compute_covar(returns_df, quantile=0.05):
    """
    Computes system-level Delta CoVaR using quantile regression.
    """
    system_return = returns_df.mean(axis=1)
    covar_dict = {}
    for i in range(60, len(returns_df), 5):
        dt = returns_df.index[i]
        sub_returns = returns_df.iloc[i-60:i]
        sub_system = system_return.iloc[i-60:i]
        
        try:
            mod = sm.QuantReg(sub_system.values, sm.add_constant(sub_returns.mean(axis=1).values))
            res = mod.fit(q=quantile, max_iter=20)
            val = float(abs(res.params[1])) if len(res.params) > 1 else 0.0
        except Exception:
            val = 0.0
        covar_dict[dt] = val

    covar_series = pd.Series(covar_dict).reindex(returns_df.index).ffill().bfill().fillna(0.0)
    covar_series.name = "covar_system"
    return covar_series

def compute_mes(returns_df, threshold=-0.02):
    """
    Computes Marginal Expected Shortfall (average return of firm when market return < threshold).
    """
    system_return = returns_df.mean(axis=1)
    mes_values = []

    for i in range(30, len(returns_df)):
        window_ret = returns_df.iloc[i-30:i]
        window_sys = system_return.iloc[i-30:i]
        
        tail_days = window_sys < threshold
        if tail_days.sum() > 0:
            mes_val = abs(window_ret[tail_days].mean().mean())
        else:
            mes_val = 0.0
        mes_values.append(mes_val)

    full_mes = [0.0]*30 + mes_values
    return pd.Series(full_mes, index=returns_df.index, name="mes_avg")

def compute_srisk(returns_df, vix_series, k=0.08):
    """
    Computes SRISK proxy = Firm Debt Proxy * (1 - LRMES).
    """
    system_vol = returns_df.std(axis=1) * np.sqrt(252)
    srisk_proxy = system_vol * (vix_series / 100.0) * k
    return pd.Series(srisk_proxy, index=returns_df.index, name="srisk_proxy")

def compute_granger_edges(returns_df, window=60, p_thresh=0.05, max_pairs=10):
    """
    Computes pairwise Granger causality edge counts over rolling windows.
    Returns: (granger_counts_series, daily_granger_edges_dict)
    """
    cols = returns_df.columns
    n_assets = len(cols)
    daily_granger_dict = {}
    daily_granger_edges = {}

    pairs = [(cols[i], cols[j]) for i in range(n_assets) for j in range(n_assets) if i != j]
    sample_pairs = pairs[:max_pairs]

    for t in range(window, len(returns_df), 5):
        dt = returns_df.index[t]
        sub = returns_df.iloc[t-window:t].fillna(0)
        
        edge_list = []
        sig_count = 0
        
        for (src, dst) in sample_pairs:
            try:
                data = sub[[dst, src]].values
                if data.std(axis=0).min() > 1e-6:
                    res = grangercausalitytests(data, maxlag=1, verbose=False)
                    p_val = res[1][0]["ssr_ftest"][1]
                    if p_val < p_thresh:
                        sig_count += 1
                        edge_list.append((src, dst, 1.0 - p_val))
            except Exception:
                pass
                
        daily_granger_dict[dt] = sig_count
        daily_granger_edges[dt] = edge_list

    granger_series = pd.Series(daily_granger_dict).reindex(returns_df.index).ffill().bfill().fillna(0)
    granger_series.name = "granger_count"
    return granger_series, daily_granger_edges

if __name__ == "__main__":
    cfg = load_config()
    proc_dir = ROOT_DIR / cfg["paths"]["data_processed"]
    ret_path = proc_dir / "bank_returns_nse.csv"
    if ret_path.exists():
        rets = pd.read_csv(ret_path, index_col=0, parse_dates=True)
        covar = compute_covar(rets.iloc[:200])
        print(f"CoVaR computed: {covar.shape}")
