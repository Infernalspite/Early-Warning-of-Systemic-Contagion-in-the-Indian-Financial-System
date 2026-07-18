"""
api/live_score.py
==================
Vercel Python serverless function — the Tier 2 "full backend" the
dashboard tries first (GET /api/live_score). Not deployed by default;
the dashboard falls back to a client-side partial score if this
returns 404, which it will until you set it up (see README.md in
this folder).

Data source: yfinance (scrapes Yahoo Finance's public chart
endpoints — the same source feature_engineering.py used to build the
original historical dataset). No API key needed. This has to run
server-side: Yahoo's endpoints don't send CORS headers, so a browser
can't call them directly, which is exactly why this is a backend
function and not more client-side JS.

What it does when deployed:
  1. Pulls recent daily closes for a reduced bank basket + Nifty Bank
     + India VIX + USD/INR straight from Yahoo Finance via yfinance.
  2. Rebuilds the subset of features.csv columns that are computable
     from that pull in the time budget of a serverless request.
  3. Re-runs Logistic Regression, Random Forest and XGBoost (the
     three models cheap enough to bundle without exceeding a
     serverless free-tier size/time limit) and returns the results.

What it deliberately does NOT do:
  - Re-run LSTM or GNN. Both need a full local PyTorch runtime;
    bundling torch (~700MB+) blows past Vercel's free-tier function
    size limit, and the GNN additionally needs same-day graphs for
    all 20 banks, not a reduced basket. Both stay benchmark-only.
  - Recompute Granger causality / CoVaR / SRISK / MES from scratch on
    every request — those are expensive (this is why
    feature_engineering.py takes ~15 minutes locally). This endpoint
    reuses the LAST COMPUTED values for those columns from
    data/processed/features_india.csv and only refreshes the
    columns that are cheap to recompute per-request (returns,
    volatility, VIX, FX, Nifty Bank technicals).
  - Guarantee uptime. yfinance scrapes an unofficial endpoint that
    Yahoo can rate-limit or change without notice — that's the
    tradeoff for not needing a key. If a pull fails, that column
    silently falls back to its last known static value and
    `live_feature_keys` in the response will just be shorter — the
    endpoint still returns 200 with whatever it did get.

Setup:
  1. pip install -r api/requirements.txt
  2. Deploy to Vercel — no environment variables, no signup, no key.
  3. Open the dashboard. It probes /api/live_score automatically and
     switches from "Tier 1 · client-side" to "Tier 2 · full backend"
     as soon as this responds.

Cold-start note: yfinance's first call in a fresh serverless instance
is often the slowest (session setup + Yahoo's own latency). If you
hit Vercel's execution time limit on the free plan, trim
LIVE_BASKET below to fewer tickers.
"""
import os
import json
import pickle
import datetime

import numpy as np
import pandas as pd
import yfinance as yf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# Reduced live basket — large, liquid banks only. The full benchmark
# uses 20 banks; pulling all 20 fresh on every request is unnecessary
# load on Yahoo's endpoint and slows the response for no real gain
# in signal (these five are highly correlated with the other 15
# during actual stress events).
LIVE_BASKET = ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS"]
NIFTY_BANK_TICKER = "^NSEBANK"
INDIA_VIX_TICKER = "^INDIAVIX"
USD_INR_TICKER = "INR=X"


def _closes(df, ticker=None):
    """yfinance's return shape differs between a single download() call
    and a multi-ticker one — this normalizes both to a plain Series."""
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        try:
            return df["Close"][ticker].dropna()
        except Exception:
            return None
    return df["Close"].dropna()


def fetch_live_market_data():
    """One batched yfinance pull covering the whole live basket plus
    Nifty Bank, India VIX and USD/INR. Batching keeps this to a
    single round-trip to Yahoo instead of N separate ones."""
    all_tickers = LIVE_BASKET + [NIFTY_BANK_TICKER, INDIA_VIX_TICKER, USD_INR_TICKER]
    raw = yf.download(all_tickers, period="2mo", interval="1d",
                       progress=False, threads=True, timeout=8)
    out = {}
    for t in all_tickers:
        s = _closes(raw, t)
        if s is not None and len(s) > 1:
            out[t] = s
    return out


def build_live_feature_row(static_defaults):
    """Builds one row matching feature_list.txt, mixing live-pulled
    values with the last known static values for anything not
    cheaply recomputable per-request. Returns (row, live_keys)."""
    row = dict(static_defaults)
    live_keys = []

    try:
        series = fetch_live_market_data()
    except Exception:
        series = {}

    # --- live: bank basket returns / volatility / correlation ---
    bank_series = {t: series[t] for t in LIVE_BASKET if t in series}
    if bank_series:
        rets = {t: np.diff(np.log(s.values)) for t, s in bank_series.items() if len(s) > 6}
        if rets:
            row["avg_return_5d"] = float(np.mean([r[-5:].mean() for r in rets.values()]))
            row["avg_volatility_5d"] = float(np.mean([r[-5:].std() for r in rets.values()]))
            row["avg_volatility_10d"] = float(np.mean([r[-10:].std() if len(r) >= 10 else r.std() for r in rets.values()]))
            row["avg_volatility_30d"] = float(np.mean([r[-30:].std() if len(r) >= 30 else r.std() for r in rets.values()]))
            live_keys += ["avg_return_5d", "avg_volatility_5d", "avg_volatility_10d", "avg_volatility_30d"]
            if len(rets) > 1:
                min_len = min(len(r) for r in rets.values())
                mat = np.array([r[-min_len:] for r in rets.values()])
                corr = np.corrcoef(mat)
                iu = np.triu_indices_from(corr, k=1)
                row["avg_pairwise_correlation"] = float(np.mean(corr[iu]))
                live_keys.append("avg_pairwise_correlation")
            # single-name vol proxies the benchmark tracks individually
            name_map = {"SBIN.NS": "sbi_vol_10d", "HDFCBANK.NS": "hdfc_vol_10d"}
            for t, col in name_map.items():
                if t in rets and len(rets[t]) >= 10:
                    row[col] = float(rets[t][-10:].std())
                    live_keys.append(col)

    # --- live: Nifty Bank index ---
    if NIFTY_BANK_TICKER in series:
        s = series[NIFTY_BANK_TICKER].values
        if len(s) > 5:
            row["nifty_bank_return_5d"] = float((s[-1] - s[-6]) / s[-6])
            live_keys.append("nifty_bank_return_5d")
        if len(s) > 30:
            roll_max = np.max(s[-30:])
            row["nifty_bank_drawdown_30d"] = float((s[-1] - roll_max) / roll_max)
            live_keys.append("nifty_bank_drawdown_30d")
        if len(s) > 15:
            deltas = np.diff(s[-15:])
            gains = deltas[deltas > 0].sum()
            losses = -deltas[deltas < 0].sum()
            rs = gains / losses if losses > 0 else 0
            row["nifty_bank_rsi"] = float(100 - (100 / (1 + rs))) if losses > 0 else 100.0
            live_keys.append("nifty_bank_rsi")

    # --- live: India VIX ---
    if INDIA_VIX_TICKER in series:
        s = series[INDIA_VIX_TICKER].values
        row["india_vix"] = float(s[-1])
        if len(s) > 1:
            row["india_vix_change"] = float((s[-1] - s[-2]) / s[-2])
        live_keys += ["india_vix", "india_vix_change"]

    # --- live: USD/INR ---
    if USD_INR_TICKER in series:
        s = series[USD_INR_TICKER].values
        row["inr_usd"] = float(s[-1])
        if len(s) > 1:
            row["inr_usd_change"] = float((s[-1] - s[-2]) / s[-2])
        live_keys += ["inr_usd", "inr_usd_change"]

    return row, sorted(set(live_keys))


def load_static_defaults():
    """Last known value for every feature column, used as the
    fallback for anything not refreshed live this request."""
    feat_path = os.path.join(DATA_DIR, "features_india.csv")
    list_path = os.path.join(DATA_DIR, "feature_list.txt")
    with open(list_path) as f:
        feature_cols = [l.strip() for l in f if l.strip()]
    df = pd.read_csv(feat_path, index_col=0, parse_dates=True).sort_index()
    last_row = df[feature_cols].ffill().iloc[-1]
    return feature_cols, last_row.to_dict()


def score_models(feature_cols, row):
    X = pd.DataFrame([row])[feature_cols]
    results = {}

    try:
        with open(os.path.join(MODELS_DIR, "logistic_regression.pkl"), "rb") as f:
            lr = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "scaler_lr.pkl"), "rb") as f:
            sc = pickle.load(f)
        results["Logistic Regression"] = float(lr.predict_proba(sc.transform(X))[0, 1])
    except Exception:
        results["Logistic Regression"] = None

    try:
        with open(os.path.join(MODELS_DIR, "random_forest_binary.pkl"), "rb") as f:
            rf = pickle.load(f)
        results["Random Forest"] = float(rf.predict_proba(X)[0, 1])
    except Exception:
        results["Random Forest"] = None

    try:
        with open(os.path.join(MODELS_DIR, "xgboost.pkl"), "rb") as f:
            xgb = pickle.load(f)
        results["XGBoost"] = float(xgb.predict_proba(X)[0, 1])
    except Exception:
        results["XGBoost"] = None

    return results


def handler(request):
    """Vercel Python runtime entrypoint. No API key required — data
    comes from a live yfinance/Yahoo Finance pull."""
    try:
        feature_cols, static_defaults = load_static_defaults()
        row, live_keys = build_live_feature_row(static_defaults)
        scores = score_models(feature_cols, row)
        payload = {
            "probability": scores.get("Logistic Regression"),
            "model_scores": scores,
            "live_feature_count": max(len(live_keys), 0),
            "live_feature_keys": live_keys,
            "feature_row": {k: row.get(k) for k in feature_cols},
            "source": "yfinance (Yahoo Finance), live pull",
            "basket": LIVE_BASKET,
            "computed_at": datetime.datetime.utcnow().isoformat() + "Z",
            "note": "Random Forest and XGBoost use the same live-basket features as Logistic "
                    "Regression. LSTM and GNN are not re-scored live — see module docstring. "
                    "Columns not listed in live_feature_keys held at their last known dataset "
                    "value because this request's Yahoo Finance pull didn't return them.",
        }
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps(payload),
        }
    except Exception as e:
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": str(e)}),
        }
