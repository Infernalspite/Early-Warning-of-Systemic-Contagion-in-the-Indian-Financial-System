"""
api/live_score.py
==================
Vercel Python serverless function — deployed at GET /api/live_score.

Pulls live market data from Yahoo Finance via yfinance, re-runs
Logistic Regression, Random Forest and XGBoost on the fresh features,
and returns a JSON payload the dashboard displays.

LSTM and GNN are excluded — bundling PyTorch (~700 MB) exceeds
Vercel's free-tier function size limit.

Vercel's Python runtime interface:
  Requires a class named `handler` inheriting from
  http.server.BaseHTTPRequestHandler with a do_GET method.
  Using a plain function is a silent failure mode.
"""
import os
import sys
import json
import pickle
import datetime
from http.server import BaseHTTPRequestHandler

# yfinance needs a writable cache dir; /tmp is the only writable
# location in a Vercel serverless function.
os.environ.setdefault("YF_CACHE_DIR", "/tmp/yfinance_cache")

import numpy as np
import pandas as pd
import yfinance as yf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

LIVE_BASKET = [
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS",
    "AXISBANK.NS", "KOTAKBANK.NS",
]
NIFTY_BANK_TICKER = "^NSEBANK"
INDIA_VIX_TICKER  = "^INDIAVIX"
USD_INR_TICKER    = "INR=X"


# ------------------------------------------------------------------ #
#  yfinance helpers — handle both old (flat) and new (MultiIndex)     #
#  column layouts transparently.                                      #
# ------------------------------------------------------------------ #

def _get_close(df: pd.DataFrame, ticker: str) -> pd.Series | None:
    """Return the Close series for `ticker` regardless of column layout."""
    if df is None or df.empty:
        return None
    # New yfinance (>=0.2.x) returns MultiIndex columns: (field, ticker)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            s = df["Close"][ticker].dropna()
            return s if not s.empty else None
        except (KeyError, TypeError):
            return None
    # Old layout: flat columns, single ticker download
    if "Close" in df.columns:
        s = df["Close"].dropna()
        return s if not s.empty else None
    return None


def fetch_live_market_data() -> dict:
    """Download recent daily OHLCV for all tickers. Returns {ticker: Series}."""
    all_tickers = LIVE_BASKET + [NIFTY_BANK_TICKER, INDIA_VIX_TICKER, USD_INR_TICKER]
    # group=True keeps MultiIndex; auto_adjust avoids split/dividend noise
    raw = yf.download(
        all_tickers,
        period="3mo",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
        timeout=12,
    )
    out = {}
    for ticker in all_tickers:
        s = _get_close(raw, ticker)
        if s is not None and len(s) > 1:
            out[ticker] = s
    return out


# ------------------------------------------------------------------ #
#  Feature engineering on live data                                   #
# ------------------------------------------------------------------ #

def build_live_feature_row(static_defaults: dict) -> tuple:
    """
    Returns (row_dict, live_keys_list, fetch_error_or_None).
    Columns not computable from the live pull are left at their
    static_defaults value (last known value from features_india.csv).
    """
    row = dict(static_defaults)
    live_keys: list[str] = []
    fetch_error = None

    try:
        series = fetch_live_market_data()
    except Exception as exc:
        series = {}
        fetch_error = str(exc)

    # ---- bank basket features ----------------------------------------
    bank_series = {t: series[t] for t in LIVE_BASKET if t in series}
    if bank_series:
        rets = {}
        for t, s in bank_series.items():
            vals = s.values.astype(float)
            if len(vals) > 6:
                rets[t] = np.diff(np.log(vals))

        if rets:
            r5  = [r[-5:].mean()  for r in rets.values()]
            rv5 = [r[-5:].std()   for r in rets.values()]
            rv10= [r[-10:].std() if len(r) >= 10 else r.std() for r in rets.values()]
            rv30= [r[-30:].std() if len(r) >= 30 else r.std() for r in rets.values()]

            row["avg_return_5d"]       = float(np.mean(r5))
            row["avg_volatility_5d"]   = float(np.mean(rv5))
            row["avg_volatility_10d"]  = float(np.mean(rv10))
            row["avg_volatility_30d"]  = float(np.mean(rv30))
            live_keys += ["avg_return_5d", "avg_volatility_5d",
                          "avg_volatility_10d", "avg_volatility_30d"]

            if len(rets) > 1:
                min_len = min(len(r) for r in rets.values())
                mat  = np.array([r[-min_len:] for r in rets.values()])
                corr = np.corrcoef(mat)
                iu   = np.triu_indices_from(corr, k=1)
                row["avg_pairwise_correlation"] = float(np.mean(corr[iu]))
                live_keys.append("avg_pairwise_correlation")

            # idiosyncratic vol for named banks
            name_map = {"SBIN.NS": "sbi_vol_10d", "HDFCBANK.NS": "hdfc_vol_10d"}
            for t, col in name_map.items():
                if t in rets and len(rets[t]) >= 10:
                    row[col] = float(rets[t][-10:].std())
                    live_keys.append(col)

    # ---- Nifty Bank ---------------------------------------------------
    if NIFTY_BANK_TICKER in series:
        s = series[NIFTY_BANK_TICKER].values.astype(float)
        if len(s) > 5:
            row["nifty_bank_return_5d"] = float((s[-1] - s[-6]) / s[-6])
            live_keys.append("nifty_bank_return_5d")
        if len(s) > 30:
            roll_max = float(np.max(s[-30:]))
            row["nifty_bank_drawdown_30d"] = float((s[-1] - roll_max) / roll_max)
            live_keys.append("nifty_bank_drawdown_30d")
        if len(s) > 14:
            deltas = np.diff(s[-15:])
            gains  = deltas[deltas > 0].sum()
            losses = -deltas[deltas < 0].sum()
            if losses > 0:
                row["nifty_bank_rsi"] = float(100 - (100 / (1 + gains / losses)))
            else:
                row["nifty_bank_rsi"] = 100.0
            live_keys.append("nifty_bank_rsi")

    # ---- India VIX ----------------------------------------------------
    if INDIA_VIX_TICKER in series:
        s = series[INDIA_VIX_TICKER].values.astype(float)
        row["india_vix"] = float(s[-1])
        if len(s) > 1:
            row["india_vix_change"] = float((s[-1] - s[-2]) / s[-2])
        live_keys += ["india_vix", "india_vix_change"]

    # ---- USD/INR ------------------------------------------------------
    if USD_INR_TICKER in series:
        s = series[USD_INR_TICKER].values.astype(float)
        row["inr_usd"] = float(s[-1])
        if len(s) > 1:
            row["inr_usd_change"] = float((s[-1] - s[-2]) / s[-2])
        live_keys += ["inr_usd", "inr_usd_change"]

    return row, sorted(set(live_keys)), fetch_error


# ------------------------------------------------------------------ #
#  Static feature defaults                                            #
# ------------------------------------------------------------------ #

def load_static_defaults() -> tuple:
    """
    Returns (feature_cols_list, last_row_dict) from the frozen
    features_india.csv — used as fallbacks for columns the live
    pull can't cheaply recompute.
    """
    list_path = os.path.join(DATA_DIR, "feature_list.txt")
    feat_path = os.path.join(DATA_DIR, "features_india.csv")

    with open(list_path) as fh:
        feature_cols = [line.strip() for line in fh if line.strip()]

    df = pd.read_csv(feat_path, index_col=0, parse_dates=True).sort_index()
    last_row = df[feature_cols].ffill().iloc[-1]
    return feature_cols, last_row.to_dict()


# ------------------------------------------------------------------ #
#  Model scoring                                                      #
# ------------------------------------------------------------------ #

def score_models(feature_cols: list, row: dict) -> dict:
    """Run LR, RF, XGBoost on the feature row. Returns {model_name: prob}."""
    X = pd.DataFrame([row])[feature_cols]
    results = {}

    # Logistic Regression
    try:
        with open(os.path.join(MODELS_DIR, "logistic_regression.pkl"), "rb") as fh:
            lr = pickle.load(fh)
        with open(os.path.join(MODELS_DIR, "scaler_lr.pkl"), "rb") as fh:
            sc = pickle.load(fh)
        results["Logistic Regression"] = float(lr.predict_proba(sc.transform(X))[0, 1])
    except Exception as exc:
        results["Logistic Regression"] = None

    # Random Forest
    try:
        with open(os.path.join(MODELS_DIR, "random_forest_binary.pkl"), "rb") as fh:
            rf = pickle.load(fh)
        results["Random Forest"] = float(rf.predict_proba(X)[0, 1])
    except Exception as exc:
        results["Random Forest"] = None

    # XGBoost
    try:
        with open(os.path.join(MODELS_DIR, "xgboost.pkl"), "rb") as fh:
            xgb_model = pickle.load(fh)
        results["XGBoost"] = float(xgb_model.predict_proba(X)[0, 1])
    except Exception as exc:
        results["XGBoost"] = None

    return results


# ------------------------------------------------------------------ #
#  Main compute payload (isolated from HTTP so it's testable)        #
# ------------------------------------------------------------------ #

def compute_payload() -> dict:
    try:
        feature_cols, static_defaults = load_static_defaults()
        row, live_keys, fetch_error = build_live_feature_row(static_defaults)
        scores = score_models(feature_cols, row)
        return {
            "probability":          scores.get("Logistic Regression"),
            "model_scores":         scores,
            "live_feature_count":   len(live_keys),
            "live_feature_keys":    live_keys,
            "feature_row":          {k: row.get(k) for k in feature_cols},
            "source":               "yfinance (Yahoo Finance), live pull",
            "basket":               LIVE_BASKET,
            "fetch_error":          fetch_error,
            "computed_at":          datetime.datetime.utcnow().isoformat() + "Z",
            "note": (
                "Random Forest and XGBoost use the same live-basket features as "
                "Logistic Regression. LSTM and GNN are not re-scored live (PyTorch "
                "bundle size exceeds Vercel free-tier limit). Columns not in "
                "live_feature_keys are held at their last known dataset value."
            ),
        }
    except Exception as exc:
        return {"error": str(exc), "error_type": type(exc).__name__}


# ------------------------------------------------------------------ #
#  Vercel HTTP handler                                                #
# ------------------------------------------------------------------ #

class handler(BaseHTTPRequestHandler):
    """
    Vercel's required Python entrypoint: a BaseHTTPRequestHandler
    subclass named exactly `handler`.  A plain function won't work —
    Vercel will silently ignore the file.
    """

    def log_message(self, fmt, *args):
        # suppress the default access-log noise in Vercel function logs
        pass

    def _write_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type",  "application/json")
        self.send_header("Content-Length", str(len(body)))
        # Never serve cached data — every fetch must be fresh
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            payload = compute_payload()
            self._write_json(payload, 200)
        except Exception as exc:
            self._write_json(
                {"error": str(exc), "error_type": type(exc).__name__}, 200
            )

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

if __name__ == "__main__":
    import http.server
    port = int(os.environ.get("PORT", 8080))
    server = http.server.HTTPServer(("0.0.0.0", port), handler)
    print(f"Starting contagion live scoring API server on port {port}...")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping api server...")
        server.server_close()
