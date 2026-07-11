"""
download_fred.py
================
Downloads US macro data from FRED (Federal Reserve Economic Data)
Used only as INPUT for CoVaR spread computation — not as a
standalone predictor.  The derived feature (MIBOR - US T-bill spread)
captures the global credit environment's influence on India.

Requires: pip install fredapi pandas
Free API key from: https://fred.stlouisfed.org/docs/api/api_key.html

Fallback: if no API key, uses yfinance proxies (^IRX, ^TYX).

Run: python download_fred.py
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

os.makedirs("data/raw", exist_ok=True)

START_DATE = "2008-01-01"
END_DATE   = "2026-07-10"

print("="*60)
print("FRED / GLOBAL MACRO DOWNLOADER")
print("="*60)

# ================================================================
# Try FRED API first, fall back to yfinance proxies
# ================================================================

fred_data = {}
use_fallback = True

try:
    from fredapi import Fred
    # Try with a key from environment variable
    import os as _os
    api_key = _os.environ.get("FRED_API_KEY", "")
    if api_key:
        fred = Fred(api_key=api_key)
        print("\n[1] Fetching from FRED API...")
        fred_data["US_10Y_Yield"]     = fred.get_series("DGS10",    observation_start=START_DATE, observation_end=END_DATE)
        fred_data["US_3M_TBill"]      = fred.get_series("TB3MS",    observation_start=START_DATE, observation_end=END_DATE)
        fred_data["US_BBB_Spread"]    = fred.get_series("BAMLC0A4CBBB", observation_start=START_DATE, observation_end=END_DATE)
        fred_data["US_HY_Spread"]     = fred.get_series("BAMLH0A0HYM2", observation_start=START_DATE, observation_end=END_DATE)
        use_fallback = False
        print("   FRED API data fetched successfully.")
    else:
        print("   No FRED_API_KEY environment variable found — using yfinance fallback.")
except Exception as e:
    print(f"   FRED API unavailable ({e}) — using yfinance fallback.")

if use_fallback:
    print("\n[1] Fetching global macro proxies via yfinance...")
    import yfinance as yf

    # ^IRX = 13-week T-Bill yield (%)
    # ^TNX = 10-year Treasury yield (%)
    # ^TYX = 30-year Treasury yield (%)
    tickers = {
        "US_3M_TBill"  : "^IRX",
        "US_10Y_Yield" : "^TNX",
    }

    for col, ticker in tickers.items():
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE,
                             auto_adjust=True, progress=False)
            if df.empty:
                raise ValueError("empty")
            series = df["Close"].squeeze()
            series.name = col
            fred_data[col] = series
            print(f"   {col} ({ticker}): {len(series)} rows")
        except Exception as e:
            print(f"   {col}: FAILED ({e}) — filling with NaN")
            fred_data[col] = pd.Series(dtype=float, name=col)

    # Approximate BBB spread from iShares IG ETF price proxy (LQD)
    # A crude but available proxy for credit conditions
    try:
        lqd = yf.download("LQD", start=START_DATE, end=END_DATE,
                           auto_adjust=True, progress=False)["Close"].squeeze()
        # Normalise to a spread proxy: deviation from 200-day SMA
        lqd_spread = ((lqd.rolling(200).mean() - lqd) / lqd * 100).rename("US_Credit_Spread_Proxy")
        fred_data["US_Credit_Spread_Proxy"] = lqd_spread
        print(f"   US_Credit_Spread_Proxy (LQD): {len(lqd_spread)} rows")
    except Exception as e:
        print(f"   US_Credit_Spread_Proxy: FAILED ({e})")

# ================================================================
# Build a combined daily DataFrame
# ================================================================

print("\n[2] Building combined FRED/macro DataFrame...")

# Create a common daily index
all_dates = pd.date_range(start=START_DATE, end=END_DATE, freq="B")
combined  = pd.DataFrame(index=all_dates)

for col, series in fred_data.items():
    if isinstance(series, pd.Series) and not series.empty:
        s = series.copy()
        s.index = pd.to_datetime(s.index)
        combined[col] = s.reindex(all_dates).ffill().bfill()
    else:
        combined[col] = np.nan

# ================================================================
# Derived Features
# ================================================================

print("[3] Computing derived spread features...")

if "US_10Y_Yield" in combined.columns and "US_3M_TBill" in combined.columns:
    # US yield curve slope (recession indicator)
    combined["US_Yield_Curve_Slope"] = combined["US_10Y_Yield"] - combined["US_3M_TBill"]
    print("   US_Yield_Curve_Slope computed")

# ================================================================
# Save
# ================================================================

out_path = "data/raw/fred_global_macro.csv"
combined.to_csv(out_path)

print(f"\nSaved: {out_path}")
print(f"Shape: {combined.shape}")
print(f"Columns: {list(combined.columns)}")
print("\nGlobal macro data ready for CoVaR feature computation.")
