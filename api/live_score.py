"""
api/live_score.py
==================
Vercel Python serverless function — the Tier 2 "full backend" the
dashboard tries first (GET /api/live_score). Not deployed by default;
the dashboard falls back to a client-side partial score if this
isn't reachable, which it won't be until you deploy it (see
README.md in this folder).

IMPORTANT — Vercel's Python runtime interface:
  Vercel does NOT use the AWS-Lambda-style "def handler(request):
  return {'statusCode': ...}" pattern. It requires a class named
  `handler` inheriting from http.server.BaseHTTPRequestHandler, with
  a do_GET method that writes the response itself. Using the wrong
  interface is a silent failure mode — Vercel just won't invoke the
  function correctly, the dashboard's fetch to /api/live_score will
  fail, and it'll permanently sit on Tier 1 (Logistic Regression
  only) with no visible error. If you're reading this because "the
  other models aren't loading," this class signature is the first
  thing to check.

Data source: yfinance (scrapes Yahoo Finance's public chart
endpoints — the same source feature_engineering.py used to build the
original historical dataset). No API key needed. This has to run
server-side: Yahoo's endpoints don't send CORS headers, so a browser
can't call them directly — that's why this is a backend function and
not more client-side JavaScript.

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
    reuses their LAST COMPUTED values from
    data/processed/features_india.csv and only refreshes the columns
    that are cheap per-request.
  - Guarantee uptime. yfinance scrapes an unofficial endpoint that
    Yahoo can rate-limit, geo-restrict for datacenter IPs, or change
    without notice. If the pull fails, this still returns 200 with
    whatever it got (possibly zero live columns) rather than a raw
    500 — the frontend shows exactly how many of the 39 features came
    through live via `live_feature_keys`, rather than silently
    pretending everything is live.

Setup:
  1. pip install -r api/requirements.txt
  2. Deploy to Vercel — no environment variables, no signup, no key.
  3. Open the dashboard. It probes /api/live_score automatically and
     switches from "Tier 1 · client-side" to "Tier 2 · full backend"
     as soon as this responds correctly.

If it still doesn't switch after deploying, check (in order):
  a. Open /api/live_score directly in your browser — do you get JSON
     back, or a Vercel error page? A blank/error page means the
     function itself isn't deploying (check the Vercel build log for
     a missing dependency or a Python version mismatch).
  b. Check the "source" and "live_feature_keys" fields in the JSON
     response — if live_feature_keys is empty, the function IS
     running but yfinance's pull failed (likely Yahoo rate-limiting
     Vercel's shared IP ranges, which does happen). The dashboard
     will still show "Tier 2" with 0 live features in that case,
     not fall back to Tier 1 — if you're seeing Tier 1 instead,
     that means (a), not this.
  c. Check Vercel's function logs for the actual exception — this
     endpoint deliberately swallows errors into the JSON response's
     "error" field instead of raising, so check that field too.
"""
import os
import sys
import json
import pickle
import datetime
from http.server import BaseHTTPRequestHandler

# yfinance needs a writable cache directory; only /tmp is writable in
# a Vercel serverless function, so this must be set before import.
os.environ.setdefault("YF_CACHE_DIR", "/tmp/yfinance_cache")

import numpy as np
import pandas as pd
import yfinance as yf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

LIVE_BASKET = ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS"]
NIFTY_BANK_TICKER = "^NSEBANK"
INDIA_VIX_TICKER = "^INDIAVIX"
USD_INR_TICKER = "INR=X"


def _closes(df, ticker=None):
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        try:
            return df["Close"][ticker].dropna()
        except Exception:
            return None
    return df["Close"].dropna()


def fetch_live_market_data():
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
    row = dict(static_defaults)
    live_keys = []
    fetch_error = None

    try:
        series = fetch_live_market_data()
    except Exception as e:
        series = {}
        fetch_error = str(e)

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
            name_map = {"SBIN.NS": "sbi_vol_10d", "HDFCBANK.NS": "hdfc_vol_10d"}
            for t, col in name_map.items():
                if t in rets and len(rets[t]) >= 10:
                    row[col] = float(rets[t][-10:].std())
                    live_keys.append(col)

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

    if INDIA_VIX_TICKER in series:
        s = series[INDIA_VIX_TICKER].values
        row["india_vix"] = float(s[-1])
        if len(s) > 1:
            row["india_vix_change"] = float((s[-1] - s[-2]) / s[-2])
        live_keys += ["india_vix", "india_vix_change"]

    if USD_INR_TICKER in series:
        s = series[USD_INR_TICKER].values
        row["inr_usd"] = float(s[-1])
        if len(s) > 1:
            row["inr_usd_change"] = float((s[-1] - s[-2]) / s[-2])
        live_keys += ["inr_usd", "inr_usd_change"]

    return row, sorted(set(live_keys)), fetch_error


def load_static_defaults():
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


def compute_payload():
    """All the actual work, isolated from the HTTP plumbing so it's
    testable directly (see README.md) without spinning up a server."""
    try:
        feature_cols, static_defaults = load_static_defaults()
        row, live_keys, fetch_error = build_live_feature_row(static_defaults)
        scores = score_models(feature_cols, row)
        payload = {
            "probability": scores.get("Logistic Regression"),
            "model_scores": scores,
            "live_feature_count": len(live_keys),
            "live_feature_keys": live_keys,
            "feature_row": {k: row.get(k) for k in feature_cols},
            "source": "yfinance (Yahoo Finance), live pull",
            "basket": LIVE_BASKET,
            "fetch_error": fetch_error,
            "computed_at": datetime.datetime.utcnow().isoformat() + "Z",
            "note": "Random Forest and XGBoost use the same live-basket features as Logistic "
                    "Regression. LSTM and GNN are not re-scored live — see module docstring. "
                    "Columns not listed in live_feature_keys held at their last known dataset "
                    "value because this request's Yahoo Finance pull didn't return them.",
        }
        return payload
    except Exception as e:
        return {"error": str(e), "error_type": type(e).__name__}


class handler(BaseHTTPRequestHandler):
    """Vercel's required Python entrypoint shape: a BaseHTTPRequestHandler
    subclass literally named `handler`. See the module docstring —
    using a plain function here is the #1 reason this silently never
    gets invoked."""

    def _write_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            payload = compute_payload()
            self._write_json(payload, 200)
        except Exception as e:
            # last-resort catch so the function never hard-crashes into
            # a raw 500 with no JSON body — the frontend expects JSON.
            self._write_json({"error": str(e), "error_type": type(e).__name__}, 200)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
