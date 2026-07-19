"""
api/live_score.py
=================
Vercel Python serverless function — GET /api/live_score

Architecture
------------
No scikit-learn or xgboost at runtime. All three models (LR, RF, XGBoost)
are stored as JSON and evaluated with ~50 lines of pure Python math.
The pkl files remain in the repo for reference/retraining — we just don't
load them here because they require the full 883 MB sklearn+xgboost bundle
which exceeds Vercel's 500 MB function size limit.

Predictions are mathematically identical to sklearn/xgboost — verified to
< 1e-8 error on the test set (see convert_models.py).

Bundle size with this approach:
  numpy>=1.24   ~30 MB
  pandas>=2.0   ~50 MB
  yfinance      ~ 5 MB
  Total         ~85 MB  (well under the 500 MB limit)

Live data
---------
Pulls daily OHLCV from Yahoo Finance (yfinance, no API key required).
Recomputes: avg_return_5d, avg_volatility_{5,10,30}d, avg_pairwise_correlation,
            india_vix, india_vix_change, inr_usd, inr_usd_change,
            nifty_bank_{return_5d,drawdown_30d,rsi}, sbi_vol_10d, hdfc_vol_10d.
All other columns default to the last row of features_india.csv.

Vercel interface
----------------
Must be a class named `handler` inheriting from BaseHTTPRequestHandler.
A plain function is a silent failure on Vercel.
"""

import os
import json
import math
import datetime
from http.server import BaseHTTPRequestHandler

import numpy as np
import pandas as pd
import yfinance as yf

os.environ.setdefault("YF_CACHE_DIR", "/tmp/yfinance_cache")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data", "processed")

LIVE_BASKET      = ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS"]
NIFTY_BANK       = "^NSEBANK"
INDIA_VIX        = "^INDIAVIX"
USD_INR          = "INR=X"


# ─────────────────────────────────────────────────────────────────────────────
#  Pure-Python model predictors (no sklearn / xgboost at runtime)
# ─────────────────────────────────────────────────────────────────────────────

def _lr_predict(lr_data: dict, row: dict, feature_cols: list) -> float:
    z = lr_data["intercept"]
    for i, col in enumerate(lr_data["features"]):
        val   = row.get(col, 0.0) or 0.0
        sc    = lr_data["scaler_scale"][i]
        scaled = (val - lr_data["scaler_mean"][i]) / sc if sc else 0.0
        z     += scaled * lr_data["coef"][i]
    return 1.0 / (1.0 + math.exp(-z))


def _rf_predict(rf_trees: list, row: dict, feature_cols: list) -> float:
    x = [row.get(c, 0.0) or 0.0 for c in feature_cols]
    total = None
    for nodes in rf_trees:
        nid = 0
        while nodes[nid]["left"] != -1:
            fi  = nodes[nid]["feature"]
            nid = nodes[nid]["left"] if x[fi] <= nodes[nid]["threshold"] else nodes[nid]["right"]
        v = nodes[nid]["value"]
        total = v if total is None else [total[i] + v[i] for i in range(len(v))]
    s = sum(total)
    probs = [v / s for v in total] if s else total
    return float(probs[1]) if len(probs) > 1 else float(probs[0])


def _xgb_walk(node: dict, x: list, fn: list) -> float:
    if "leaf" in node:
        return node["leaf"]
    sf = node["split"]
    fi = fn.index(sf) if sf in fn else (int(sf[1:]) if sf[1:].isdigit() else 0)
    val = x[fi]
    tid = node["yes"] if (val is None or math.isnan(val) or val < node["split_condition"]) else node["no"]
    for c in node["children"]:
        if c["nodeid"] == tid:
            return _xgb_walk(c, x, fn)
    return 0.0


def _xgb_predict(xgb_data: dict, row: dict, feature_cols: list) -> float:
    fn = xgb_data["feature_names"]
    x  = [row.get(c, 0.0) or 0.0 for c in feature_cols]
    bs = xgb_data["base_score"]
    leaf_sum   = sum(_xgb_walk(t, x, fn) for t in xgb_data["trees"])
    margin_off = math.log(bs / (1.0 - bs)) if 0.0 < bs < 1.0 else 0.0
    return 1.0 / (1.0 + math.exp(-(margin_off + leaf_sum)))


# ─────────────────────────────────────────────────────────────────────────────
#  yfinance helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_close(df, ticker):
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        try:
            s = df["Close"][ticker].dropna()
            return s if not s.empty else None
        except Exception:
            return None
    if "Close" in df.columns:
        s = df["Close"].dropna()
        return s if not s.empty else None
    return None


def fetch_live_data():
    all_tickers = LIVE_BASKET + [NIFTY_BANK, INDIA_VIX, USD_INR]
    try:
        raw = yf.download(
            all_tickers, period="3mo", interval="1d",
            auto_adjust=True, progress=False, threads=True, timeout=12,
        )
        return {t: s for t in all_tickers if (s := _get_close(raw, t)) is not None and len(s) > 1}
    except Exception as e:
        return {}, str(e)


# ─────────────────────────────────────────────────────────────────────────────
#  Feature engineering on live data
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_row(static_defaults: dict):
    try:
        result = fetch_live_data()
        if isinstance(result, tuple):
            series, fetch_err = result
        else:
            series, fetch_err = result, None
    except Exception as e:
        series, fetch_err = {}, str(e)

    row       = dict(static_defaults)
    live_keys = []

    # Bank basket
    bank_s = {t: series[t].values.astype(float) for t in LIVE_BASKET if t in series and len(series[t]) > 6}
    if bank_s:
        rets = {t: np.diff(np.log(v)) for t, v in bank_s.items()}
        r5  = [r[-5:].mean()  for r in rets.values()]
        rv5 = [r[-5:].std()   for r in rets.values()]
        rv10= [r[-10:].std() if len(r)>=10 else r.std() for r in rets.values()]
        rv30= [r[-30:].std() if len(r)>=30 else r.std() for r in rets.values()]
        row["avg_return_5d"]      = float(np.mean(r5))
        row["avg_volatility_5d"]  = float(np.mean(rv5))
        row["avg_volatility_10d"] = float(np.mean(rv10))
        row["avg_volatility_30d"] = float(np.mean(rv30))
        live_keys += ["avg_return_5d","avg_volatility_5d","avg_volatility_10d","avg_volatility_30d"]
        if len(rets) > 1:
            min_len = min(len(r) for r in rets.values())
            mat  = np.array([r[-min_len:] for r in rets.values()])
            corr = np.corrcoef(mat)
            iu   = np.triu_indices_from(corr, k=1)
            row["avg_pairwise_correlation"] = float(np.mean(corr[iu]))
            live_keys.append("avg_pairwise_correlation")
        for ticker, col in [("SBIN.NS","sbi_vol_10d"),("HDFCBANK.NS","hdfc_vol_10d")]:
            if ticker in rets and len(rets[ticker]) >= 10:
                row[col] = float(rets[ticker][-10:].std())
                live_keys.append(col)

    # Nifty Bank
    if NIFTY_BANK in series:
        s = series[NIFTY_BANK].values.astype(float)
        if len(s) > 5:
            row["nifty_bank_return_5d"] = float((s[-1]-s[-6])/s[-6]); live_keys.append("nifty_bank_return_5d")
        if len(s) > 30:
            row["nifty_bank_drawdown_30d"] = float((s[-1]-np.max(s[-30:]))/np.max(s[-30:])); live_keys.append("nifty_bank_drawdown_30d")
        if len(s) > 14:
            d = np.diff(s[-15:]); g = d[d>0].sum(); l = -d[d<0].sum()
            row["nifty_bank_rsi"] = float(100 - 100/(1+g/l)) if l > 0 else 100.0
            live_keys.append("nifty_bank_rsi")

    # India VIX
    if INDIA_VIX in series:
        s = series[INDIA_VIX].values.astype(float)
        row["india_vix"] = float(s[-1]); live_keys.append("india_vix")
        if len(s) > 1:
            row["india_vix_change"] = float((s[-1]-s[-2])/s[-2]); live_keys.append("india_vix_change")

    # USD/INR
    if USD_INR in series:
        s = series[USD_INR].values.astype(float)
        row["inr_usd"] = float(s[-1]); live_keys.append("inr_usd")
        if len(s) > 1:
            row["inr_usd_change"] = float((s[-1]-s[-2])/s[-2]); live_keys.append("inr_usd_change")

    return row, sorted(set(live_keys)), fetch_err


# ─────────────────────────────────────────────────────────────────────────────
#  Static defaults from features_india.csv
# ─────────────────────────────────────────────────────────────────────────────

def load_static_defaults():
    with open(os.path.join(DATA_DIR, "feature_list.txt")) as f:
        feature_cols = [l.strip() for l in f if l.strip()]
    df = pd.read_csv(os.path.join(DATA_DIR, "features_india.csv"), index_col=0, parse_dates=True).sort_index()
    last_row = df[feature_cols].ffill().iloc[-1]
    return feature_cols, last_row.to_dict()


# ─────────────────────────────────────────────────────────────────────────────
#  Score all models
# ─────────────────────────────────────────────────────────────────────────────

def score_models(feature_cols: list, row: dict) -> dict:
    results = {}

    # Logistic Regression
    try:
        with open(os.path.join(MODELS_DIR, "logistic_regression.json")) as f:
            results["Logistic Regression"] = _lr_predict(json.load(f), row, feature_cols)
    except Exception:
        results["Logistic Regression"] = None

    # Random Forest
    try:
        with open(os.path.join(MODELS_DIR, "random_forest_binary.json")) as f:
            results["Random Forest"] = _rf_predict(json.load(f), row, feature_cols)
    except Exception:
        results["Random Forest"] = None

    # XGBoost
    try:
        with open(os.path.join(MODELS_DIR, "xgboost.json")) as f:
            results["XGBoost"] = _xgb_predict(json.load(f), row, feature_cols)
    except Exception:
        results["XGBoost"] = None

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Main payload
# ─────────────────────────────────────────────────────────────────────────────

def compute_payload() -> dict:
    try:
        feature_cols, static_defaults = load_static_defaults()
        row, live_keys, fetch_err     = build_feature_row(static_defaults)
        scores                        = score_models(feature_cols, row)
        return {
            "probability":        scores.get("Logistic Regression"),
            "model_scores":       scores,
            "live_feature_count": len(live_keys),
            "live_feature_keys":  live_keys,
            "feature_row":        {k: row.get(k) for k in feature_cols},
            "source":             "yfinance (Yahoo Finance), live pull",
            "basket":             LIVE_BASKET,
            "fetch_error":        fetch_err,
            "computed_at":        datetime.datetime.utcnow().isoformat() + "Z",
            "note": (
                "LR, RF and XGBoost use pure-Python JSON inference "
                "(predictions identical to sklearn/xgboost to <1e-8). "
                "LSTM and GNN remain benchmark-only."
            ),
        }
    except Exception as exc:
        return {"error": str(exc), "error_type": type(exc).__name__}


# ─────────────────────────────────────────────────────────────────────────────
#  Vercel HTTP handler (class-based, required by Vercel Python runtime)
# ─────────────────────────────────────────────────────────────────────────────

class handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress access log noise

    def _write_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type",   "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control",  "no-store, no-cache, must-revalidate")
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            self._write_json(compute_payload())
        except Exception as exc:
            self._write_json({"error": str(exc), "error_type": type(exc).__name__})

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    server = __import__("http.server", fromlist=["HTTPServer"]).HTTPServer(("0.0.0.0", port), handler)
    print(f"Contagion API running on port {port}...")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
