"""
src/data/fetch_prices.py
========================
Downloads stock price time-series for Indian Banks (15) and NBFCs (12),
plus key benchmark indices (Nifty 50, Nifty Bank, Nifty Fin Service, India VIX).

Primary: yfinance
Fallback: jugaad-data (NSE history)
"""

import os
import pathlib
import yaml
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# Load Config
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def fetch_yfinance_prices(tickers_dict, start_date="2005-01-01", end_date="2026-09-15"):
    """
    Downloads historical close prices via yfinance.
    """
    tickers = list(tickers_dict.keys())
    print(f"Fetching {len(tickers)} tickers via yfinance ({start_date} to {end_date})...")
    
    raw = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )
    
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy() if "Close" in raw.columns else raw.copy()
        
    prices.rename(columns=tickers_dict, inplace=True)
    prices.dropna(how="all", inplace=True)
    prices.ffill(inplace=True)
    prices.bfill(inplace=True)
    return prices

def fetch_all_prices():
    cfg = load_config()
    raw_dir = ROOT_DIR / cfg["paths"]["data_raw"]
    proc_dir = ROOT_DIR / cfg["paths"]["data_processed"]
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)

    start_date = cfg["dates"]["start"]
    end_date = cfg["dates"]["end"]

    # 1. Fetch Banks
    bank_dict = cfg["tickers"]["banks"]
    bank_prices = fetch_yfinance_prices(bank_dict, start_date, end_date)
    bank_prices.to_csv(raw_dir / "bank_prices_nse.csv")
    print(f"Bank Prices saved -> {raw_dir / 'bank_prices_nse.csv'} | Shape: {bank_prices.shape}")

    # 2. Fetch NBFCs
    nbfc_dict = cfg["tickers"]["nbfcs"]
    nbfc_prices = fetch_yfinance_prices(nbfc_dict, start_date, end_date)
    nbfc_prices.to_csv(raw_dir / "nbfc_prices_nse.csv")
    print(f"NBFC Prices saved -> {raw_dir / 'nbfc_prices_nse.csv'} | Shape: {nbfc_prices.shape}")

    # 3. Combine Bank + NBFC panel
    combined_prices = pd.concat([bank_prices, nbfc_prices], axis=1).ffill().bfill()
    combined_prices.to_csv(raw_dir / "all_financial_prices_nse.csv")

    # Compute daily log returns
    combined_returns = np.log(combined_prices / combined_prices.shift(1)).dropna(how="all")
    combined_returns.to_csv(proc_dir / "bank_returns_nse.csv")
    print(f"Combined Returns saved -> {proc_dir / 'bank_returns_nse.csv'} | Shape: {combined_returns.shape}")

    # 4. Fetch Benchmark Indices
    indices_dict = cfg["tickers"]["indices"]
    indices_prices = fetch_yfinance_prices(indices_dict, start_date, end_date)
    indices_prices.to_csv(raw_dir / "nifty_indices.csv")
    print(f"Indices saved -> {raw_dir / 'nifty_indices.csv'} | Shape: {indices_prices.shape}")

    print("\nFetch Prices Complete!")

if __name__ == "__main__":
    fetch_all_prices()
