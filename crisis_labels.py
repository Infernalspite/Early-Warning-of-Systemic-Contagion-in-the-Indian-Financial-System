"""
crisis_labels.py
================
Labels every trading day as:
  3-class:  0=Normal, 1=Pre-Crisis, 2=Crisis
  Binary:   high_stress_next_30d = 1 if a crisis starts within 30 trading days

Run: python crisis_labels.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import os

os.makedirs("data/labels", exist_ok=True)
os.makedirs("outputs/charts", exist_ok=True)

print("="*60)
print("INDIAN CRISIS PERIOD LABELER")
print("="*60)

# ── LOAD BANK PRICES ─────────────────────────────────────────
prices = pd.read_csv("data/raw/bank_prices_nse.csv", index_col=0, parse_dates=True)
prices.index = pd.to_datetime(prices.index, dayfirst=True)
prices = prices.sort_index()

print(f"\nLoaded prices: {prices.shape}")
print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

# ================================================================
# CRISIS EVENT DEFINITIONS
# Each has:
#   peak_date : the actual trigger date (used for binary label anchor)
#   start     : when the stressed window begins
#   end       : when market stabilised
# ================================================================

CRISIS_PERIODS = [
    {
        "name"       : "2008 GFC Spillover",
        "peak_date"  : "2008-10-24",
        "start"      : "2008-09-15",
        "end"        : "2009-03-31",
        "description": "Lehman collapse spillover — Sensex halved, INR crisis"
    },
    {
        "name"       : "2013 Taper Tantrum",
        "peak_date"  : "2013-08-28",
        "start"      : "2013-06-01",
        "end"        : "2013-09-30",
        "description": "Fed taper signal — INR crashed to 68, RBI emergency rate hike"
    },
    {
        "name"       : "IL&FS Crisis",
        "peak_date"  : "2018-09-21",
        "start"      : "2018-08-01",
        "end"        : "2018-12-31",
        "description": "IL&FS defaulted on debt payments — shadow banking collapse"
    },
    {
        "name"       : "Yes Bank Crisis",
        "peak_date"  : "2020-03-05",
        "start"      : "2020-01-01",
        "end"        : "2020-04-30",
        "description": "RBI imposed moratorium — near collapse of Yes Bank"
    },
    {
        "name"       : "COVID Crash",
        "peak_date"  : "2020-03-23",
        "start"      : "2020-02-01",
        "end"        : "2020-05-31",
        "description": "Global pandemic crash — Nifty fell 40% in weeks"
    },
    {
        "name"       : "Lakshmi Vilas Bank",
        "peak_date"  : "2020-11-17",
        "start"      : "2020-10-01",
        "end"        : "2020-12-31",
        "description": "RBI forced merger with DBS Bank India"
    },
    {
        "name"       : "Adani Crisis",
        "peak_date"  : "2023-01-24",
        "start"      : "2023-01-24",
        "end"        : "2023-03-31",
        "description": "Hindenburg report — PSU banks had heavy Adani exposure"
    },
]

# ================================================================
# 3-CLASS LABELS  (label = 0 / 1 / 2)
# ================================================================

print("\n[1] Creating 3-class crisis labels...")

labels = pd.DataFrame(index=prices.index)
labels["label"]       = 0
labels["label_name"]  = "Normal"
labels["crisis_name"] = ""

for crisis in CRISIS_PERIODS:
    crisis_start = pd.Timestamp(crisis["start"])
    crisis_end   = pd.Timestamp(crisis["end"])
    pre_crisis   = crisis_start - pd.Timedelta(days=30)

    pre_mask = (labels.index >= pre_crisis) & (labels.index < crisis_start)
    labels.loc[pre_mask, "label"]       = 1
    labels.loc[pre_mask, "label_name"]  = "Pre-Crisis"
    labels.loc[pre_mask, "crisis_name"] = crisis["name"]

    crisis_mask = (labels.index >= crisis_start) & (labels.index <= crisis_end)
    labels.loc[crisis_mask, "label"]       = 2
    labels.loc[crisis_mask, "label_name"]  = "Crisis"
    labels.loc[crisis_mask, "crisis_name"] = crisis["name"]

    print(f"   {crisis['name']}: pre={pre_crisis.date()} | crisis={crisis_start.date()} to {crisis_end.date()}")

# ================================================================
# BINARY LABEL:  high_stress_next_30d
#
# Forward-looking: on date t, the label = 1 if a CRISIS PERIOD
# STARTS within the next 30 TRADING DAYS from t.
#
# This is what every model will predict.  It makes the comparison
# valid because the same label is used across all 5 models.
# ================================================================

print("\n[2] Creating binary label: high_stress_next_30d ...")

labels["high_stress_next_30d"] = 0

trading_days = labels.index  # only trading days in the index

for crisis in CRISIS_PERIODS:
    crisis_start = pd.Timestamp(crisis["start"])

    # Find all trading days that are within 30 trading days BEFORE crisis_start
    # i.e., t such that 0 < (crisis_start - t) in trading days <= 30
    for i, dt in enumerate(trading_days):
        if dt >= crisis_start:
            continue
        # count trading days between dt and crisis_start
        future_days = trading_days[(trading_days > dt) & (trading_days <= crisis_start)]
        if len(future_days) <= 30:
            labels.loc[dt, "high_stress_next_30d"] = 1

print(f"   Binary label created.")

# ================================================================
# SUMMARY
# ================================================================

print("\n" + "="*60)
print("LABEL DISTRIBUTION SUMMARY")
print("="*60)

total = len(labels)
for val, name in [(0,"Normal"),(1,"Pre-Crisis"),(2,"Crisis")]:
    n = (labels["label"] == val).sum()
    print(f"   3-class {val} ({name:10s}): {n:4d} days ({n/total*100:.1f}%)")

pos = labels["high_stress_next_30d"].sum()
print(f"\n   Binary  1 (high_stress_next_30d=1): {pos:4d} days ({pos/total*100:.1f}%)")
print(f"   Binary  0 (normal):                 {total-pos:4d} days ({(total-pos)/total*100:.1f}%)")

# ================================================================
# SAVE
# ================================================================

labels.to_csv("data/labels/crisis_labels_india.csv")
print(f"\nLabels saved to: data/labels/crisis_labels_india.csv")

# ================================================================
# CHART
# ================================================================

print("\n[3] Creating chart...")

try:
    indices = pd.read_csv("data/raw/nifty_indices.csv", index_col=0, parse_dates=True)
    indices.index = pd.to_datetime(indices.index, dayfirst=True)
    indices = indices.sort_index()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    if "Nifty Bank" in indices.columns:
        nifty = indices["Nifty Bank"].dropna()
        ax1.plot(nifty.index, nifty.values, color="royalblue", linewidth=1.2, label="Nifty Bank")

    for crisis in CRISIS_PERIODS:
        cs = pd.Timestamp(crisis["start"])
        ce = pd.Timestamp(crisis["end"])
        pre = cs - pd.Timedelta(days=30)
        ax1.axvspan(pre, cs, alpha=0.22, color="yellow")
        ax1.axvspan(cs, ce, alpha=0.28, color="red")
        ax1.text(cs, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 50000,
                 crisis["name"], rotation=90, fontsize=7, color="darkred", va="top", ha="right")

    ax1.set_title("Nifty Bank with 3-Class Crisis Labels (Yellow=Pre, Red=Crisis)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Nifty Bank")
    ax1.legend(handles=[
        plt.Line2D([0],[0], color="royalblue", linewidth=2, label="Nifty Bank"),
        Patch(facecolor="yellow", alpha=0.5, label="Pre-Crisis"),
        Patch(facecolor="red",    alpha=0.4, label="Crisis"),
    ], loc="upper left", fontsize=9)

    # Bottom panel: binary label
    ax2.fill_between(labels.index, labels["high_stress_next_30d"], alpha=0.6, color="darkorange", step="mid")
    ax2.set_ylabel("high_stress_next_30d\n(Binary)", fontsize=9)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Normal", "Stress Incoming"])
    ax2.set_title("Binary Label: high_stress_next_30d = 1 within 30 trading days", fontsize=10)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig("outputs/charts/07_crisis_labels.png", dpi=150)
    plt.close()
    print("   Chart saved: outputs/charts/07_crisis_labels.png")
except Exception as e:
    print(f"   Warning: chart skipped ({e})")

print("\n" + "="*60)
print("LABELING COMPLETE")
print("="*60)
print("   data/labels/crisis_labels_india.csv")
print("   Columns: label, label_name, crisis_name, high_stress_next_30d")