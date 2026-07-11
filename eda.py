"""
eda.py - Fully Fixed Version
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os

plt.style.use("seaborn-v0_8-darkgrid")
os.makedirs("outputs/charts", exist_ok=True)

print("="*60)
print("📊 INDIAN BANK RISK ENGINE — EDA")
print("="*60)

# ── LOAD DATA ────────────────────────────────────────────────
print("\n📂 Loading data...")

prices = pd.read_csv(
    "data/raw/bank_prices_nse.csv",
    index_col=0,
    parse_dates=True
)

returns = pd.read_csv(
    "data/processed/bank_returns_nse.csv",
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

# Fix date index format
prices.index  = pd.to_datetime(prices.index,  dayfirst=True)
returns.index = pd.to_datetime(returns.index, dayfirst=True)
indices.index = pd.to_datetime(indices.index, dayfirst=True)
macro.index   = pd.to_datetime(macro.index,   dayfirst=True)

print(f"✅ Bank prices  : {prices.shape}")
print(f"✅ Bank returns : {returns.shape}")
print(f"✅ Indices      : {indices.shape}")
print(f"✅ Macro        : {macro.shape}")

# ── CRISIS DATES ─────────────────────────────────────────────
CRISIS_EVENTS = {
    "IL&FS"    : "2018-09-21",
    "Yes Bank" : "2020-03-05",
    "COVID"    : "2020-03-23",
    "Adani"    : "2023-01-24",
}

def add_crisis_lines(ax, ymax):
    for event, date in CRISIS_EVENTS.items():
        ax.axvline(
            x=mdates.datestr2num(date),
            color="red",
            linestyle="--",
            linewidth=1.5,
            alpha=0.8
        )
        ax.text(
            mdates.datestr2num(date),
            ymax * 0.92,
            event,
            rotation=90,
            fontsize=8,
            color="red",
            va="top"
        )

# ════════════════════════════════════════════════════════════
# CHART 1 — ALL BANK PRICES NORMALIZED
# ════════════════════════════════════════════════════════════

print("\n📈 Chart 1: All Bank Prices Normalized...")

normalized = (prices / prices.iloc[0]) * 100

fig, ax = plt.subplots(figsize=(16, 7))

for col in normalized.columns:
    ax.plot(
        normalized.index,
        normalized[col],
        linewidth=0.8,
        alpha=0.7,
        label=col
    )

add_crisis_lines(ax, normalized.values.max())

ax.set_title(
    "Indian Bank Stock Prices — Normalized to 100 (2014–2024)",
    fontsize=14, fontweight="bold"
)
ax.set_xlabel("Date")
ax.set_ylabel("Normalized Price (Base = 100)")
ax.legend(loc="upper left", fontsize=7, ncol=2)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
plt.savefig("outputs/charts/01_bank_prices_normalized.png", dpi=150)
plt.show()
print("   ✅ Chart 1 saved!")


# ════════════════════════════════════════════════════════════
# CHART 2 — YES BANK COLLAPSE
# ════════════════════════════════════════════════════════════

print("\n📈 Chart 2: Yes Bank Collapse...")

fig, ax = plt.subplots(figsize=(14, 5))

yes_bank = prices["Yes Bank"]["2018-01-01":"2021-12-31"]

ax.plot(
    yes_bank.index,
    yes_bank.values,
    color="crimson",
    linewidth=2,
    label="Yes Bank"
)
ax.fill_between(
    yes_bank.index,
    yes_bank.values,
    alpha=0.15,
    color="crimson"
)

ax.axvline(
    x=mdates.datestr2num("2020-03-05"),
    color="black",
    linestyle="--",
    linewidth=2
)
ax.text(
    mdates.datestr2num("2020-03-05"),
    yes_bank.max() * 0.85,
    "RBI Moratorium\nMar 2020",
    fontsize=10,
    color="black",
    ha="right",
    fontweight="bold"
)

ax.set_title(
    "Yes Bank Stock Price — The Collapse (2018–2021)",
    fontsize=14, fontweight="bold"
)
ax.set_xlabel("Date")
ax.set_ylabel("Price (INR)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.legend()

plt.tight_layout()
plt.savefig("outputs/charts/02_yes_bank_collapse.png", dpi=150)
plt.show()
print("   ✅ Chart 2 saved!")


# ════════════════════════════════════════════════════════════
# CHART 3 — CORRELATION HEATMAP
# ════════════════════════════════════════════════════════════

print("\n📈 Chart 3: Correlation Heatmap...")

corr_matrix = returns.corr()

fig, ax = plt.subplots(figsize=(14, 11))

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    center=0,
    vmin=-1, vmax=1,
    square=True,
    linewidths=0.5,
    ax=ax,
    annot_kws={"size": 7}
)

ax.set_title(
    "Indian Bank Return Correlations\n"
    "(Higher = More Connected = Higher Contagion Risk)",
    fontsize=13, fontweight="bold"
)
ax.tick_params(axis="x", rotation=45)
ax.tick_params(axis="y", rotation=0)

plt.tight_layout()
plt.savefig("outputs/charts/03_correlation_heatmap.png", dpi=150)
plt.show()
print("   ✅ Chart 3 saved!")


# ════════════════════════════════════════════════════════════
# CHART 4 — INDIA VIX vs NIFTY BANK
# ════════════════════════════════════════════════════════════

print("\n📈 Chart 4: India VIX vs Nifty Bank...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8),
                                sharex=True)

if "India VIX" in indices.columns:
    vix = indices["India VIX"].dropna()
    ax1.plot(
        vix.index,
        vix.values,
        color="darkorange",
        linewidth=1.2
    )
    ax1.fill_between(
        vix.index,
        vix.values,
        alpha=0.3,
        color="orange"
    )
    ax1.axhline(
        y=25,
        color="red",
        linestyle="--",
        linewidth=1,
        label="Danger (VIX > 25)"
    )
    ax1.set_ylabel("India VIX")
    ax1.set_title(
        "India VIX (Fear Index) vs Nifty Bank",
        fontsize=13, fontweight="bold"
    )
    ax1.legend()

    for event, date in CRISIS_EVENTS.items():
        ax1.axvline(
            x=mdates.datestr2num(date),
            color="red",
            linestyle=":",
            linewidth=1.5,
            alpha=0.7
        )

if "Nifty Bank" in indices.columns:
    nifty = indices["Nifty Bank"].dropna()
    ax2.plot(
        nifty.index,
        nifty.values,
        color="royalblue",
        linewidth=1.2
    )
    ax2.set_ylabel("Nifty Bank Index")
    ax2.set_xlabel("Date")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    for event, date in CRISIS_EVENTS.items():
        ax2.axvline(
            x=mdates.datestr2num(date),
            color="red",
            linestyle=":",
            linewidth=1.5,
            alpha=0.7
        )

plt.tight_layout()
plt.savefig("outputs/charts/04_vix_vs_niftybank.png", dpi=150)
plt.show()
print("   ✅ Chart 4 saved!")


# ════════════════════════════════════════════════════════════
# CHART 5 — WORST SINGLE DAY RETURNS
# ════════════════════════════════════════════════════════════

print("\n📈 Chart 5: Worst Single-Day Returns...")

worst_days        = returns.min() * 100
worst_days_sorted = worst_days.sort_values()

fig, ax = plt.subplots(figsize=(14, 7))

colors = [
    "crimson"    if x < -15 else
    "darkorange" if x < -10 else
    "steelblue"
    for x in worst_days_sorted
]

bars = ax.barh(
    worst_days_sorted.index,
    worst_days_sorted.values,
    color=colors,
    edgecolor="white",
    linewidth=0.5
)

ax.set_title(
    "Worst Single-Day Return Per Bank (2014–2024)",
    fontsize=13, fontweight="bold"
)
ax.set_xlabel("Return (%)")
ax.axvline(x=0, color="black", linewidth=0.8)

for bar, val in zip(bars, worst_days_sorted.values):
    ax.text(
        val - 0.3,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.1f}%",
        va="center",
        ha="right",
        fontsize=8,
        color="white",
        fontweight="bold"
    )

plt.tight_layout()
plt.savefig("outputs/charts/05_worst_single_day_returns.png", dpi=150)
plt.show()
print("   ✅ Chart 5 saved!")


# ════════════════════════════════════════════════════════════
# CHART 6 — INR/USD RATE
# ════════════════════════════════════════════════════════════

print("\n📈 Chart 6: INR/USD Exchange Rate...")

fig, ax = plt.subplots(figsize=(14, 5))

if "INR_USD_Rate" in macro.columns:
    inr = macro["INR_USD_Rate"].dropna()
    ax.plot(
        inr.index,
        inr.values,
        color="green",
        linewidth=1.5
    )
    ax.fill_between(
        inr.index,
        inr.values,
        alpha=0.15,
        color="green"
    )

    for event, date in CRISIS_EVENTS.items():
        ax.axvline(
            x=mdates.datestr2num(date),
            color="red",
            linestyle="--",
            linewidth=1.5,
            alpha=0.8
        )
        ax.text(
            mdates.datestr2num(date),
            inr.max() * 0.98,
            event,
            rotation=90,
            fontsize=8,
            color="red",
            va="top"
        )

ax.set_title(
    "INR/USD Exchange Rate — Rupee Weakness = Banking Stress",
    fontsize=13, fontweight="bold"
)
ax.set_xlabel("Date")
ax.set_ylabel("INR per 1 USD")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
plt.savefig("outputs/charts/06_inr_usd_rate.png", dpi=150)
plt.show()
print("   ✅ Chart 6 saved!")


# ════════════════════════════════════════════════════════════
# SUMMARY STATISTICS
# ════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("📊 KEY STATISTICS")
print("="*60)

print("\n🏦 BANK RETURN STATISTICS (Daily %):")
print("-"*40)
stats = (returns * 100).describe().round(3)
print(stats.loc[["mean", "std", "min", "max"]].to_string())

print("\n🔗 TOP 5 MOST CORRELATED BANK PAIRS:")
print("-"*40)
corr_pairs = corr_matrix.unstack()
corr_pairs = corr_pairs[corr_pairs < 1.0]
corr_pairs = corr_pairs.sort_values(ascending=False)
for (b1, b2), corr in corr_pairs.head(5).items():
    print(f"   {b1[:15]:15s} ↔ {b2[:15]:15s} : {corr:.3f}")

print("\n" + "="*60)
print("🎉 ALL 6 CHARTS DONE!")
print("="*60)
print("\n📁 Saved to: outputs/charts/")
print("   01_bank_prices_normalized.png")
print("   02_yes_bank_collapse.png")
print("   03_correlation_heatmap.png")
print("   04_vix_vs_niftybank.png")
print("   05_worst_single_day_returns.png")
print("   06_inr_usd_rate.png")
print("\n✅ Phase 2 Complete! Ready for Phase 3.")