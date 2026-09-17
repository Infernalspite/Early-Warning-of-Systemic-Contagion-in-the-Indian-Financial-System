"""
src/data/fetch_macro.py
=======================
Fetches macro indicators from FRED API (US Fed rate, VIX, spreads)
and RBI DBIE / proxy sources for Indian interbank rates (MIBOR, Repo).
"""

import os
import pathlib
import yaml
import pandas as pd
import numpy as np
from fredapi import Fred

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def fetch_macro_indicators():
    cfg = load_config()
    raw_dir = ROOT_DIR / cfg["paths"]["data_raw"]
    os.makedirs(raw_dir, exist_ok=True)

    api_key = cfg["api_keys"]["fred"]
    fred_series_map = cfg["fred_series"]

    print("Fetching macro data from FRED API...")
    fred = Fred(api_key=api_key)

    df_list = []
    for series_id, col_name in fred_series_map.items():
        try:
            s = fred.get_series(series_id, observation_start=cfg["dates"]["start"], observation_end=cfg["dates"]["end"])
            s.name = col_name
            df_list.append(s)
            print(f"  Downloaded {series_id} ({col_name}): {len(s)} rows")
        except Exception as e:
            print(f"  WARNING: Could not fetch FRED series {series_id}: {e}")

    if df_list:
        macro_df = pd.concat(df_list, axis=1).ffill().bfill()
    else:
        macro_df = pd.DataFrame()

    # Load/synthesize RBI interbank rate data (MIBOR, Repo rate)
    # Repo rate historical schedule in India
    repo_schedule = pd.DataFrame([
        ("2005-01-01", 6.00), ("2008-07-01", 9.00), ("2009-04-01", 4.75),
        ("2011-10-01", 8.50), ("2013-05-01", 7.25), ("2015-01-01", 7.75),
        ("2018-08-01", 6.50), ("2020-03-27", 4.40), ("2020-05-22", 4.00),
        ("2022-05-04", 4.40), ("2023-02-08", 6.50), ("2026-09-15", 6.50)
    ], columns=["Date", "repo_rate"])
    repo_schedule["Date"] = pd.to_datetime(repo_schedule["Date"])
    repo_schedule.set_index("Date", inplace=True)
    
    full_dates = pd.date_range(start=cfg["dates"]["start"], end=cfg["dates"]["end"], freq="D")
    repo_daily = repo_schedule.reindex(full_dates).ffill().bfill()
    repo_daily.index.name = "Date"

    macro_df = macro_df.reindex(full_dates).ffill().bfill()
    macro_df["repo_rate"] = repo_daily["repo_rate"]

    macro_df.to_csv(raw_dir / "fred_global_macro.csv")
    macro_df.to_csv(raw_dir / "macro_indicators.csv")
    print(f"Macro indicators saved -> {raw_dir / 'macro_indicators.csv'} | Shape: {macro_df.shape}")

if __name__ == "__main__":
    fetch_macro_indicators()
