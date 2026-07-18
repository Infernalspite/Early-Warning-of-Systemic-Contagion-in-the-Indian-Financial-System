# pip install networkx statsmodels
"""
feature_engineering.py
======================
Builds all ML features for the Indian Risk Engine.

Original features (31):
  - Rolling correlations between banks
  - Volatility (rolling std of returns)
  - Momentum (rolling mean returns)
  - Macro indicators (VIX, INR/USD)
  - Network features (density, avg correlation)
  - Crisis labels merged in

New advanced features added:
  a) Network centrality (degree, betweenness, clustering) via NetworkX
  b) CoVaR proxy (system-level quantile regression) via statsmodels
  c) SRISK proxy (beta * VIX)
  d) MES - Marginal Expected Shortfall
  e) MIBOR-Repo spread proxy
  f) Granger causality count (rolling, sampled)

Run: python feature_engineering.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import random
from sklearn.decomposition import PCA

os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs/charts", exist_ok=True)

print("=" * 60)
print("FEATURE ENGINEERING -- INDIAN RISK ENGINE")
print("=" * 60)

# ============================================================
# LOAD ALL DATA
# ============================================================

print("\nLoading data...")

returns = pd.read_csv(
    "data/processed/bank_returns_nse.csv",
    index_col=0,
    parse_dates=True
)

prices = pd.read_csv(
    "data/raw/bank_prices_nse.csv",
    index_col=0,
    parse_dates=True
)

indices = pd.read_csv(
    "data/raw/nifty_indices.csv",
    index_col=0,
    parse_dates=True
)

macro = pd.read_csv(
    "data/raw/macro_indicators.csv",
    index_col=0,
    parse_dates=True
)

labels = pd.read_csv(
    "data/labels/crisis_labels_india.csv",
    index_col=0,
    parse_dates=True
)

# Fix date formats
returns.index = pd.to_datetime(returns.index, dayfirst=True)
prices.index  = pd.to_datetime(prices.index,  dayfirst=True)
indices.index = pd.to_datetime(indices.index, dayfirst=True)
macro.index   = pd.to_datetime(macro.index,   dayfirst=True)
labels.index  = pd.to_datetime(labels.index,  dayfirst=True)

# Sort all by date
returns = returns.sort_index()
prices  = prices.sort_index()
indices = indices.sort_index()
macro   = macro.sort_index()
labels  = labels.sort_index()

print(f"  Returns  : {returns.shape}")
print(f"  Prices   : {prices.shape}")
print(f"  Indices  : {indices.shape}")
print(f"  Macro    : {macro.shape}")
print(f"  Labels   : {labels.shape}")

bank_cols = returns.columns.tolist()
n_banks   = len(bank_cols)
print(f"  Banks    : {n_banks} ({', '.join(bank_cols[:4])} ...)")

# ============================================================
# FEATURE FAMILY 1 -- VOLATILITY FEATURES
# Rolling standard deviation of returns
# High volatility = stress signal
# Windows: 5 days, 10 days, 30 days
# ============================================================

print("\nBuilding Feature Family 1: Volatility...")

features = pd.DataFrame(index=returns.index)

for window in [5, 10, 30]:
    rolling_vol = returns.rolling(window=window).std()
    features[f"avg_volatility_{window}d"] = rolling_vol.mean(axis=1)
    features[f"max_volatility_{window}d"] = rolling_vol.max(axis=1)

print(f"  Volatility features: {[c for c in features.columns]}")

# ============================================================
# FEATURE FAMILY 2 -- MOMENTUM FEATURES
# ============================================================

print("\nBuilding Feature Family 2: Momentum...")

for window in [5, 10, 30]:
    rolling_ret = returns.rolling(window=window).mean()
    features[f"avg_return_{window}d"] = rolling_ret.mean(axis=1)
    features[f"min_return_{window}d"] = rolling_ret.min(axis=1)

print("  Momentum features added")

# ============================================================
# FEATURE FAMILY 3 -- NETWORK / CORRELATION FEATURES
# ============================================================

print("\nBuilding Feature Family 3: Network Correlation & Absorption Ratio...")
print("  (Computing rolling correlations and PCA...)")

window = 30
avg_corr_list    = []
network_density  = []
absorption_ratio = []
dates_list       = []

pca_model = PCA(n_components=1)

for i in range(window, len(returns)):
    window_data  = returns.iloc[i - window:i]
    window_clean = window_data.fillna(0)
    corr_mat     = window_clean.corr()

    upper  = corr_mat.where(
        np.triu(np.ones(corr_mat.shape), k=1).astype(bool)
    )
    values = upper.stack().values

    avg_corr = np.nanmean(values) if len(values) > 0 else 0
    density  = np.nanmean(values > 0.6) if len(values) > 0 else 0

    pca_model.fit(window_clean)
    abs_ratio = pca_model.explained_variance_ratio_[0]

    avg_corr_list.append(avg_corr)
    network_density.append(density)
    absorption_ratio.append(abs_ratio)
    dates_list.append(returns.index[i])

avg_corr_series    = pd.Series(avg_corr_list,   index=dates_list)
network_den_series = pd.Series(network_density,  index=dates_list)
abs_ratio_series   = pd.Series(absorption_ratio, index=dates_list)

features["avg_pairwise_correlation"] = avg_corr_series
features["network_density_06"]       = network_den_series
features["absorption_ratio"]         = abs_ratio_series

print("  Network & PCA features added")

# ============================================================
# FEATURE FAMILY 4 -- MACRO FEATURES
# ============================================================

print("\nBuilding Feature Family 4: Macro Indicators (Indian Subcontinent)...")

if "India VIX" in indices.columns:
    features["india_vix"]        = indices["India VIX"]
    features["india_vix_change"] = indices["India VIX"].pct_change()
    print("  India VIX added")

if "Nifty Bank" in indices.columns:
    features["nifty_bank_return_5d"] = (
        indices["Nifty Bank"].pct_change().rolling(5).mean()
    )
    nb      = indices["Nifty Bank"]
    nb_peak = nb.rolling(window=30, min_periods=1).max()
    features["nifty_bank_drawdown_30d"] = (nb - nb_peak) / nb_peak

    delta = nb.diff()
    gain  = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs    = gain / (loss + 1e-9)
    features["nifty_bank_rsi"] = 100 - (100 / (1 + rs))

    ema_12 = nb.ewm(span=12, adjust=False).mean()
    ema_26 = nb.ewm(span=26, adjust=False).mean()
    features["nifty_bank_ema_crossover"] = ema_12 - ema_26
    print("  Nifty Bank return, drawdown, RSI, and EMA crossover added")

if "Nifty Financial Services" in indices.columns:
    features["nifty_fin_return_5d"] = (
        indices["Nifty Financial Services"].pct_change().rolling(5).mean()
    )
    print("  Nifty Financial Services return added")

if "Nifty IT" in indices.columns:
    features["nifty_it_return_5d"] = (
        indices["Nifty IT"].pct_change().rolling(5).mean()
    )
    print("  Nifty IT return added")

if "INR_USD_Rate" in macro.columns:
    features["inr_usd"]        = macro["INR_USD_Rate"]
    features["inr_usd_change"] = macro["INR_USD_Rate"].pct_change()
    print("  INR/USD rate added")

if "RBI_Repo_Rate" in macro.columns:
    features["rbi_repo_rate"]            = macro["RBI_Repo_Rate"]
    features["rbi_repo_rate_change_30d"] = macro["RBI_Repo_Rate"].diff(30)
    print("  RBI Repo Rate features added")

# ============================================================
# FEATURE FAMILY 5 -- INDIVIDUAL BANK STRESS SIGNALS
# ============================================================

print("\nBuilding Feature Family 5: Bank Stress Signals...")

if "Yes Bank" in returns.columns:
    features["yes_bank_vol_10d"]   = returns["Yes Bank"].rolling(10).std()
    features["yes_bank_return_5d"] = returns["Yes Bank"].rolling(5).mean()
    print("  Yes Bank stress signals added")

if "State Bank of India" in returns.columns:
    features["sbi_vol_10d"] = returns["State Bank of India"].rolling(10).std()
    print("  SBI stress signal added")

if "HDFC Bank" in returns.columns:
    features["hdfc_vol_10d"] = returns["HDFC Bank"].rolling(10).std()
    print("  HDFC stress signal added")

# ============================================================
# MERGE BINARY LABEL high_stress_next_30d
# ============================================================

print("\nMerging high_stress_next_30d label...")

if "high_stress_next_30d" in labels.columns:
    features["high_stress_next_30d"] = labels["high_stress_next_30d"]
    features["high_stress_next_30d"] = (
        features["high_stress_next_30d"].fillna(0).astype(int)
    )
    print("  high_stress_next_30d merged")
else:
    print("  WARNING: high_stress_next_30d column not found in labels")

# ============================================================
# MERGE ORIGINAL CRISIS LABELS
# ============================================================

print("\nMerging original crisis labels...")

features = features.join(
    labels[["label", "label_name", "crisis_name"]], how="left"
)
features["label"]      = features["label"].fillna(0).astype(int)
features["label_name"] = features["label_name"].fillna("Normal")

# Drop rows where core features are all NaN (first ~30 rows due to rolling)
features = features.dropna(subset=["avg_volatility_30d"])

print(f"  After label merge & dropna: {features.shape}")

# ============================================================
# NEW FEATURE 6a -- NETWORK CENTRALITY (NetworkX)
# Rolling 30-day correlation graph, edge threshold = 0.4
# mean_degree_centrality, mean_betweenness_centrality,
# clustering_coefficient
# ============================================================

print("\nBuilding New Feature 6a: Network Centrality (NetworkX)...")

try:
    import networkx as nx
    HAS_NX = True
    print("  networkx available")
except ImportError:
    HAS_NX = False
    print("  WARNING: networkx not installed. "
          "Network centrality features will be NaN. "
          "Install with: pip install networkx")

degree_list      = []
betweenness_list = []
clustering_list  = []
nx_dates_list    = []

corr_window = 30
edge_thresh  = 0.4

for i in range(corr_window, len(returns)):
    window_data = returns.iloc[i - corr_window:i].fillna(0)
    date_i      = returns.index[i]

    if not HAS_NX:
        degree_list.append(np.nan)
        betweenness_list.append(np.nan)
        clustering_list.append(np.nan)
        nx_dates_list.append(date_i)
        continue

    try:
        corr_mat = window_data.corr()
        G = nx.Graph()
        G.add_nodes_from(range(n_banks))

        for r in range(n_banks):
            for c in range(r + 1, n_banks):
                val = corr_mat.iloc[r, c]
                if not np.isnan(val) and abs(val) >= edge_thresh:
                    G.add_edge(r, c, weight=val)

        deg_cent = nx.degree_centrality(G)
        btw_cent = nx.betweenness_centrality(G, normalized=True)
        clust    = nx.transitivity(G)

        degree_list.append(np.nanmean(list(deg_cent.values())))
        betweenness_list.append(np.nanmean(list(btw_cent.values())))
        clustering_list.append(clust)
    except Exception:
        degree_list.append(np.nan)
        betweenness_list.append(np.nan)
        clustering_list.append(np.nan)

    nx_dates_list.append(date_i)

features["mean_degree_centrality"] = pd.Series(
    degree_list, index=nx_dates_list
)
features["mean_betweenness_centrality"] = pd.Series(
    betweenness_list, index=nx_dates_list
)
features["clustering_coefficient"] = pd.Series(
    clustering_list, index=nx_dates_list
)

print(f"  mean_degree_centrality NaN      : "
      f"{features['mean_degree_centrality'].isna().sum()}")
print(f"  mean_betweenness_centrality NaN : "
      f"{features['mean_betweenness_centrality'].isna().sum()}")
print(f"  clustering_coefficient NaN      : "
      f"{features['clustering_coefficient'].isna().sum()}")

# ============================================================
# NEW FEATURE 6b -- CoVaR PROXY
# Quantile regression (q=0.05) of Nifty Bank 5-day return
# on each bank's 5-day return. Rolling 252-day window.
# CoVaR = predicted Nifty Bank return when bank is at VaR (5th pctile).
# Average across all banks => covar_system
# ============================================================

print("\nBuilding New Feature 6b: CoVaR Proxy (statsmodels QuantReg)...")

try:
    from statsmodels.regression.quantile_regression import QuantReg
    HAS_SM = True
    print("  statsmodels available")
except ImportError:
    HAS_SM = False
    print("  WARNING: statsmodels not installed. "
          "CoVaR/SRISK/MES/Granger features will be NaN. "
          "Install with: pip install statsmodels")

if "Nifty Bank" in indices.columns:
    nb_ret5 = indices["Nifty Bank"].pct_change(5)
else:
    nb_ret5 = pd.Series(np.nan, index=returns.index)

bank_ret5 = returns.rolling(5).mean()

covar_252_window = 252
covar_list  = []
covar_dates = []

covar_stride = 5  # recompute every 5 trading days, ffill between (QuantReg is expensive)

for i in range(covar_252_window, len(returns)):
    date_i = returns.index[i]
    covar_dates.append(date_i)

    if not HAS_SM:
        covar_list.append(np.nan)
        continue

    if (i - covar_252_window) % covar_stride != 0:
        covar_list.append(np.nan)  # filled via ffill after the loop
        continue

    try:
        nb_window   = nb_ret5.reindex(returns.index).iloc[i - covar_252_window:i]
        bank_window = bank_ret5.iloc[i - covar_252_window:i]

        bank_covar_vals = []
        for col in bank_cols:
            bk = bank_window[col]
            common = pd.concat([nb_window, bk], axis=1).dropna()
            if len(common) < 30:
                continue
            y = common.iloc[:, 0].values
            x = common.iloc[:, 1].values
            X = np.column_stack([np.ones(len(x)), x])
            try:
                model  = QuantReg(y, X)
                result = model.fit(q=0.05, max_iter=1000)
                bank_var  = np.nanpercentile(x, 5)
                covar_val = result.params[0] + result.params[1] * bank_var
                bank_covar_vals.append(covar_val)
            except Exception:
                pass

        covar_list.append(
            np.nanmean(bank_covar_vals) if bank_covar_vals else np.nan
        )
    except Exception:
        covar_list.append(np.nan)

features["covar_system"] = pd.Series(covar_list, index=covar_dates).ffill()
print(f"  covar_system NaN: {features['covar_system'].isna().sum()}")

# ============================================================
# NEW FEATURE 6c -- SRISK PROXY
# srisk_proxy = mean(bank_beta * india_vix_level * 0.01)
# beta from rolling 60-day OLS of bank return on Nifty Bank return
# ============================================================

print("\nBuilding New Feature 6c: SRISK Proxy...")

if "Nifty Bank" in indices.columns:
    nb_daily = indices["Nifty Bank"].pct_change()
else:
    nb_daily = pd.Series(np.nan, index=returns.index)

srisk_window = 60
srisk_list   = []
srisk_dates  = []

vix_series = (
    indices["India VIX"]
    if "India VIX" in indices.columns
    else pd.Series(np.nan, index=returns.index)
)

for i in range(srisk_window, len(returns)):
    date_i = returns.index[i]
    srisk_dates.append(date_i)

    try:
        nb_w  = nb_daily.reindex(returns.index).iloc[i - srisk_window:i]
        vix_i = vix_series.reindex(returns.index).iloc[i]

        betas = []
        for col in bank_cols:
            bk_w = returns[col].iloc[i - srisk_window:i]
            common = pd.concat([nb_w, bk_w], axis=1).dropna()
            if len(common) < 10:
                continue
            y = common.iloc[:, 1].values
            x = common.iloc[:, 0].values
            if np.var(x) > 0:
                beta = np.cov(x, y)[0, 1] / np.var(x)
                betas.append(beta)

        if betas and not np.isnan(vix_i):
            srisk_list.append(np.nanmean(betas) * vix_i * 0.01)
        else:
            srisk_list.append(np.nan)
    except Exception:
        srisk_list.append(np.nan)

features["srisk_proxy"] = pd.Series(srisk_list, index=srisk_dates)
print(f"  srisk_proxy NaN: {features['srisk_proxy'].isna().sum()}")

# ============================================================
# NEW FEATURE 6d -- MES (Marginal Expected Shortfall)
# On each date, worst 5% days of Nifty Bank in past 252 days;
# compute mean bank return on those same days.
# Average across all 20 banks => mes_avg
# ============================================================

print("\nBuilding New Feature 6d: MES (Marginal Expected Shortfall)...")

mes_window = 252
mes_list   = []
mes_dates  = []

nb_daily_idx = nb_daily.reindex(returns.index)

for i in range(mes_window, len(returns)):
    date_i = returns.index[i]
    mes_dates.append(date_i)

    try:
        nb_w      = nb_daily_idx.iloc[i - mes_window:i]
        threshold = np.nanpercentile(nb_w.dropna().values, 5)
        bad_days  = nb_w[nb_w <= threshold].index

        if len(bad_days) == 0:
            mes_list.append(np.nan)
            continue

        bank_mes_vals = []
        for col in bank_cols:
            bk_w    = returns[col].iloc[i - mes_window:i]
            bad_ret = bk_w.reindex(bad_days).dropna()
            if len(bad_ret) > 0:
                bank_mes_vals.append(np.nanmean(bad_ret.values))

        mes_list.append(
            np.nanmean(bank_mes_vals) if bank_mes_vals else np.nan
        )
    except Exception:
        mes_list.append(np.nan)

features["mes_avg"] = pd.Series(mes_list, index=mes_dates)
print(f"  mes_avg NaN: {features['mes_avg'].isna().sum()}")

# ============================================================
# NEW FEATURE 6e -- MIBOR-REPO SPREAD PROXY
# approx_mibor = india_vix / 10 + rbi_repo_rate
# spread = approx_mibor - rbi_repo_rate = india_vix / 10
# ============================================================

print("\nBuilding New Feature 6e: MIBOR-Repo Spread Proxy...")

if "India VIX" in indices.columns:
    features["mibor_repo_spread"] = indices["India VIX"] / 10.0
    print("  mibor_repo_spread = India VIX / 10 (spread proxy)")
else:
    features["mibor_repo_spread"] = np.nan
    print("  WARNING: India VIX not found; mibor_repo_spread set to NaN")

print(f"  mibor_repo_spread NaN: {features['mibor_repo_spread'].isna().sum()}")

# ============================================================
# NEW FEATURE 6f -- GRANGER CAUSALITY COUNT
# Rolling 60-day window, 10 random bank pairs, p<0.05, lag=1.
# Max 20 sampled dates per calendar year for speed.
# ============================================================

print("\nBuilding New Feature 6f: Granger Causality Count...")
print("  (Sampled dates only -- this may take a few minutes)")

try:
    from statsmodels.tsa.stattools import grangercausalitytests
    HAS_GRANGER = True
    print("  grangercausalitytests available")
except ImportError:
    HAS_GRANGER = False
    print("  WARNING: statsmodels not installed. granger_count will be NaN.")

granger_window   = 60
granger_n_pairs  = 10
granger_lag      = 1
granger_p_thresh = 0.05

# Build sampled date indices: max 20 per calendar year
all_dates    = returns.index[granger_window:]
years        = all_dates.year.unique()
sampled_idxs = set()

for yr in years:
    yr_mask  = all_dates.year == yr
    yr_pos   = np.where(yr_mask)[0]
    n_sample = min(20, len(yr_pos))
    chosen   = np.random.choice(yr_pos, size=n_sample, replace=False)
    for c in chosen:
        sampled_idxs.add(granger_window + c)

sampled_idxs = sorted(sampled_idxs)

all_bank_pairs = [
    (bank_cols[r], bank_cols[c])
    for r in range(n_banks)
    for c in range(r + 1, n_banks)
]

granger_vals = {}

for i in sampled_idxs:
    if i >= len(returns):
        continue
    date_i = returns.index[i]

    if not HAS_GRANGER:
        granger_vals[date_i] = np.nan
        continue

    try:
        window_data   = returns.iloc[i - granger_window:i].fillna(0)
        sampled_pairs = random.sample(
            all_bank_pairs,
            min(granger_n_pairs, len(all_bank_pairs))
        )
        sig_count = 0
        for (bk1, bk2) in sampled_pairs:
            try:
                ts_data = window_data[[bk1, bk2]].dropna()
                if len(ts_data) < granger_lag + 5:
                    continue
                res   = grangercausalitytests(
                    ts_data.values, maxlag=granger_lag, verbose=False
                )
                p_val = res[granger_lag][0]["ssr_ftest"][1]
                if p_val < granger_p_thresh:
                    sig_count += 1
            except Exception:
                pass
        granger_vals[date_i] = sig_count
    except Exception:
        granger_vals[date_i] = np.nan

granger_series_sparse = pd.Series(granger_vals)
granger_full          = granger_series_sparse.reindex(returns.index)
granger_full          = granger_full.ffill()

features["granger_count"] = granger_full
print(f"  granger_count NaN (after ffill): "
      f"{features['granger_count'].isna().sum()}")

# ============================================================
# SAVE FEATURE DATASET
# ============================================================

features.to_csv("data/processed/features_india.csv")
print(f"\nSaved to: data/processed/features_india.csv")

# ============================================================
# SAVE FEATURE LIST
# ============================================================

label_cols = {"label", "label_name", "crisis_name", "high_stress_next_30d"}
feat_cols  = [c for c in features.columns if c not in label_cols]

with open("data/processed/feature_list.txt", "w") as fh:
    for col in feat_cols:
        fh.write(col + "\n")

print(f"Saved feature list to: data/processed/feature_list.txt "
      f"({len(feat_cols)} features)")

# ============================================================
# CHART -- VISUALISE KEY FEATURES OVER TIME
# ============================================================

print("\nCreating feature visualization chart...")

fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)


def shade_crises(ax):
    for _, row in features[features["label"] > 0].iterrows():
        color = "red" if row["label"] == 2 else "yellow"
        ax.axvspan(
            row.name, row.name + pd.Timedelta(days=1),
            alpha=0.15, color=color
        )


ax = axes[0]
ax.plot(
    features.index,
    features["avg_volatility_30d"] * 100,
    color="purple", linewidth=1.2
)
shade_crises(ax)
ax.set_ylabel("Avg Volatility %")
ax.set_title("Average Bank Volatility (30-day rolling)",
             fontsize=11, fontweight="bold")

ax = axes[1]
ax.plot(
    features.index,
    features["network_density_06"],
    color="darkorange", linewidth=1.2
)
shade_crises(ax)
ax.set_ylabel("Network Density")
ax.set_title(
    "Network Density (Fraction of Bank Pairs with Correlation > 0.6)",
    fontsize=11, fontweight="bold"
)

ax = axes[2]
if "india_vix" in features.columns:
    ax.plot(
        features.index,
        features["india_vix"],
        color="red", linewidth=1.2
    )
    ax.axhline(y=25, color="black", linestyle="--",
               linewidth=1, label="Danger (>25)")
    ax.legend(fontsize=8)
shade_crises(ax)
ax.set_ylabel("India VIX")
ax.set_title("India VIX -- Fear Index",
             fontsize=11, fontweight="bold")

ax = axes[3]
if "absorption_ratio" in features.columns:
    ax.plot(
        features.index,
        features["absorption_ratio"] * 100,
        color="teal", linewidth=1.2
    )
shade_crises(ax)
ax.set_ylabel("Absorption Ratio %")
ax.set_title(
    "Absorption Ratio (Variance explained by PC1 of bank returns)",
    fontsize=11, fontweight="bold"
)
ax.set_xlabel("Date")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="red",    alpha=0.3, label="Crisis Period"),
    Patch(facecolor="yellow", alpha=0.5, label="Pre-Crisis Period"),
]
axes[0].legend(handles=legend_elements, loc="upper left", fontsize=8)

plt.suptitle(
    "Key Risk Features -- Indian Banking System (2014-2024)",
    fontsize=14, fontweight="bold", y=1.01
)
plt.tight_layout()
plt.savefig("outputs/charts/08_features_overview.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  Chart saved: outputs/charts/08_features_overview.png")

# ============================================================
# SUMMARY OF NEW FEATURES ADDED
# ============================================================

new_feature_cols = [
    "mean_degree_centrality",
    "mean_betweenness_centrality",
    "clustering_coefficient",
    "covar_system",
    "srisk_proxy",
    "mes_avg",
    "mibor_repo_spread",
    "granger_count",
    "high_stress_next_30d",
]

print("\n" + "=" * 60)
print("NEW FEATURES ADDED -- SUMMARY")
print("=" * 60)
print(f"  {'Feature':<35} {'NaN Count':>10}  {'Non-NaN':>10}")
print("  " + "-" * 58)
for col in new_feature_cols:
    if col in features.columns:
        nan_c   = int(features[col].isna().sum())
        non_nan = int(features[col].notna().sum())
        print(f"  {col:<35} {nan_c:>10}  {non_nan:>10}")
    else:
        print(f"  {col:<35} {'NOT ADDED':>10}")

print("\n" + "=" * 60)
print("FINAL FEATURE SUMMARY")
print("=" * 60)
print(f"\nTotal input features  : {len(feat_cols)}")
print(f"Total trading days    : {len(features)}")
print(f"Total columns (incl labels): {features.shape[1]}")

print("\nAll feature columns:")
for i, col in enumerate(feat_cols, 1):
    print(f"  {i:2}. {col}")

print("\n" + "=" * 60)
print("PHASE 4 COMPLETE!")
print("=" * 60)
print("\nFiles created/updated:")
print("  data/processed/features_india.csv")
print("  data/processed/feature_list.txt")
print("  outputs/charts/08_features_overview.png")
print("\nFeatures complete. Shape: {}".format(features.shape))