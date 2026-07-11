"""
download_data.py
================
Downloads all datasets needed for the
Indian Systemic Risk Contagion Engine project.

Run this file once to get all your data.
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime

# ── CREATE FOLDERS ───────────────────────────────────────────
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

print("="*60)
print("🇮🇳  INDIAN RISK ENGINE — DATA DOWNLOADER")
print("="*60)

# ════════════════════════════════════════════════════════════
# DATASET 1 — INDIAN BANK STOCK PRICES (NSE)
# 20 banks × 10 years of daily closing prices
# ════════════════════════════════════════════════════════════

print("\n📥 DATASET 1: Indian Bank Stock Prices (NSE)")
print("-"*50)

BANKS = {
    # PSU Banks
    "SBIN.NS"       : "State Bank of India",
    "PNB.NS"        : "Punjab National Bank",
    "BANKBARODA.NS" : "Bank of Baroda",
    "CANBK.NS"      : "Canara Bank",
    "UNIONBANK.NS"  : "Union Bank of India",
    "INDIANB.NS"    : "Indian Bank",
    "IOB.NS"        : "Indian Overseas Bank",
    "MAHABANK.NS"   : "Bank of Maharashtra",
    # Private Banks
    "HDFCBANK.NS"   : "HDFC Bank",
    "ICICIBANK.NS"  : "ICICI Bank",
    "KOTAKBANK.NS"  : "Kotak Mahindra Bank",
    "AXISBANK.NS"   : "Axis Bank",
    "INDUSINDBK.NS" : "IndusInd Bank",
    "YESBANK.NS"    : "Yes Bank",
    "FEDERALBNK.NS" : "Federal Bank",
    "BANDHANBNK.NS" : "Bandhan Bank",
    "IDFCFIRSTB.NS" : "IDFC First Bank",
    "RBLBANK.NS"    : "RBL Bank",
    "AUBANK.NS"     : "AU Small Finance Bank",
    "DCBBANK.NS"    : "DCB Bank",
}

tickers = list(BANKS.keys())

raw = yf.download(
    tickers     = tickers,
    start       = "2014-01-01",
    end         = datetime.today().strftime("%Y-%m-%d"),
    auto_adjust = True,
    progress    = True
)

# Keep only closing prices
bank_prices = raw["Close"].copy()
bank_prices.rename(columns=BANKS, inplace=True)
bank_prices.dropna(how="all", inplace=True)
bank_prices.ffill(inplace=True)

# Save
bank_prices.to_csv("data/raw/bank_prices_nse.csv")
print(f"✅ Bank Prices saved!")
print(f"   Shape : {bank_prices.shape[0]} days × {bank_prices.shape[1]} banks")
print(f"   File  : data/raw/bank_prices_nse.csv")


# ════════════════════════════════════════════════════════════
# DATASET 2 — DAILY RETURNS
# % change in price from previous day
# This is what ML models actually use
# ════════════════════════════════════════════════════════════

print("\n📥 DATASET 2: Daily Bank Returns")
print("-"*50)

returns = bank_prices.pct_change()
returns.dropna(how="all", inplace=True)
returns.to_csv("data/processed/bank_returns_nse.csv")

print(f"✅ Daily Returns saved!")
print(f"   Shape : {returns.shape[0]} days × {returns.shape[1]} banks")
print(f"   File  : data/processed/bank_returns_nse.csv")


# ════════════════════════════════════════════════════════════
# DATASET 3 — NIFTY INDICES
# Nifty 50, Bank Nifty, India VIX, Nifty Financial Services, Nifty IT
# These are the benchmark/fear indicators
# ════════════════════════════════════════════════════════════

print("\n📥 DATASET 3: Nifty Indices + India VIX")
print("-"*50)

INDICES = {
    "^NSEI"      : "Nifty 50",
    "^NSEBANK"   : "Nifty Bank",
    "^INDIAVIX"  : "India VIX",
    "NIFTY_FIN_SERVICE.NS": "Nifty Financial Services",
    "^CNXIT"     : "Nifty IT",
}

idx_raw = yf.download(
    tickers     = list(INDICES.keys()),
    start       = "2014-01-01",
    end         = datetime.today().strftime("%Y-%m-%d"),
    auto_adjust = True,
    progress    = True
)

# Handle multi-index columns if returned by yfinance
if isinstance(idx_raw.columns, pd.MultiIndex):
    indices = idx_raw["Close"].copy()
else:
    indices = idx_raw["Close"].copy() if "Close" in idx_raw else idx_raw

indices.rename(columns=INDICES, inplace=True)
indices.dropna(how="all", inplace=True)
indices.ffill(inplace=True)
indices.to_csv("data/raw/nifty_indices.csv")

print(f"✅ Nifty Indices saved!")
print(f"   Shape : {indices.shape[0]} days × {indices.shape[1]} indices")
print(f"   File  : data/raw/nifty_indices.csv")


# ════════════════════════════════════════════════════════════
# DATASET 4 — MACRO INDICATORS (INDIAN SUBCONTINENT ONLY)
# INR/USD exchange rate & RBI Repo Rate
# These show domestic monetary policy and currency stress
# ════════════════════════════════════════════════════════════

print("\n📥 DATASET 4: Macro Indicators (Indian Subcontinent)")
print("-"*50)

MACRO = {
    "INR=X"  : "INR_USD_Rate",
}

macro_raw = yf.download(
    tickers     = list(MACRO.keys()),
    start       = "2014-01-01",
    end         = datetime.today().strftime("%Y-%m-%d"),
    auto_adjust = True,
    progress    = True
)

if isinstance(macro_raw.columns, pd.MultiIndex):
    macro = macro_raw["Close"].copy()
else:
    macro = macro_raw["Close"].copy() if "Close" in macro_raw else macro_raw

macro.rename(columns=MACRO, inplace=True)
macro.dropna(how="all", inplace=True)
macro.ffill(inplace=True)

# Generate RBI Repo Rate
def get_rbi_repo_rate(index):
    changes = [
        ("2014-01-01", 7.75),
        ("2014-01-28", 8.00),
        ("2015-01-15", 7.75),
        ("2015-03-04", 7.50),
        ("2015-06-02", 7.25),
        ("2015-09-29", 6.75),
        ("2016-04-05", 6.50),
        ("2016-10-04", 6.25),
        ("2017-08-02", 6.00),
        ("2018-06-06", 6.25),
        ("2018-08-01", 6.50),
        ("2019-02-07", 6.25),
        ("2019-04-04", 6.00),
        ("2019-06-06", 5.75),
        ("2019-08-07", 5.40),
        ("2019-10-04", 5.15),
        ("2020-03-27", 4.40),
        ("2020-05-22", 4.00),
        ("2022-05-04", 4.40),
        ("2022-06-08", 4.90),
        ("2022-08-05", 5.40),
        ("2022-09-30", 5.90),
        ("2022-12-07", 6.25),
        ("2023-02-08", 6.50),
        ("2025-02-07", 6.25),
        ("2025-04-09", 6.00),
        ("2025-06-06", 5.50),
        ("2025-12-05", 5.25),
    ]
    s = pd.Series(index=index, dtype=float)
    dt_index = pd.to_datetime(index)
    sorted_changes = sorted(changes, key=lambda x: x[0])
    
    # Fill based on chronological updates
    for date_str, rate in sorted_changes:
        change_dt = pd.to_datetime(date_str)
        s.loc[dt_index >= change_dt] = rate
        
    s.bfill(inplace=True)
    s.ffill(inplace=True)
    return s

macro["RBI_Repo_Rate"] = get_rbi_repo_rate(macro.index)
macro.to_csv("data/raw/macro_indicators.csv")

print(f"✅ Macro Indicators saved!")
print(f"   Shape : {macro.shape[0]} days × {macro.shape[1]} indicators")
print(f"   File  : data/raw/macro_indicators.csv")


# ════════════════════════════════════════════════════════════
# DATASET 5 — MASTER DATASET
# Merge everything into one single file
# This is what we feed into the ML models
# ════════════════════════════════════════════════════════════

print("\n📥 DATASET 5: Master Dataset (All Combined)")
print("-"*50)

master = bank_prices.join(indices, how="left")
master = master.join(macro, how="left")
master.ffill(inplace=True)
master.to_csv("data/processed/master_dataset.csv")

print(f"✅ Master Dataset saved!")
print(f"   Shape : {master.shape[0]} days × {master.shape[1]} columns")
print(f"   File  : data/processed/master_dataset.csv")


# ════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("🎉 ALL DATASETS DOWNLOADED SUCCESSFULLY!")
print("="*60)
print("\n📁 FILES CREATED:")
print("   data/raw/")
print("   ├── bank_prices_nse.csv      ← 20 bank prices (10 years)")
print("   ├── nifty_indices.csv        ← Nifty50, BankNifty, VIX")
print("   └── macro_indicators.csv     ← INR/USD, Gold, Oil, VIX")
print("\n   data/processed/")
print("   ├── bank_returns_nse.csv     ← Daily % returns")
print("   └── master_dataset.csv       ← Everything merged")

print("\n📊 QUICK STATS:")
print(f"   Date range    : 2014-01-01 → {datetime.today().strftime('%Y-%m-%d')}")
print(f"   Trading days  : ~{len(bank_prices)} days")
print(f"   Banks tracked : {len(BANKS)}")
print(f"   Total columns : {master.shape[1]}")

print("\n✅ You are ready to show this tomorrow!")
print("   Open any CSV in data/raw/ to see the data.")