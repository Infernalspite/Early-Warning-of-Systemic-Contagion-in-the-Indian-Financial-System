"""
run_local.py
============
Runs the whole dashboard locally — no Vercel, no GitHub, no
deployment config to get wrong. Just:

    pip install -r requirements_local.txt
    python run_local.py

...then it opens http://localhost:5000 in your browser automatically.

What this gives you that the Vercel path didn't:
  Running locally means no serverless size limit (Vercel's Python
  functions cap out around 250MB unzipped on the free tier, which is
  why the deployed version could only afford Logistic Regression,
  Random Forest and XGBoost). Locally, torch is fine to load, so
  this adds a 4th model — LSTM — to the live-updating set.

  GNN still isn't live-scored. Not a size limit this time — it's
  architectural: the GNN needs a fresh correlation graph across all
  20 banks rebuilt from live data every request, which is a
  meaningfully bigger and slower piece of engineering than the other
  four models' flat feature vectors. It stays benchmark-only, same
  as before, and the dashboard says so rather than faking a number.

How live data flows:
  1. On every request to /api/live_score, this pulls live daily
     closes for a bank basket + Nifty Bank + India VIX + USD/INR from
     Yahoo Finance via yfinance. No key, no signup.
  2. Rebuilds the live-computable subset of the 39-column feature
     matrix from that pull (returns, volatility, VIX, FX, Nifty Bank
     technicals, basket correlation) — same logic the Vercel version
     used, not recomputed from scratch here.
  3. For LSTM specifically: takes the last 29 rows of the real
     historical feature matrix and appends one new row built from
     today's live pull, then scores that 30-day window. This means
     the LSTM's prediction genuinely changes with each live refresh
     (the newest day in its window is live), while still being
     grounded in real recent history for the other 29 days.
  4. Runs Logistic Regression, Random Forest, XGBoost and LSTM on
     the live-updated inputs and returns all four scores.

The dashboard (index.html) already knows how to call /api/live_score
and render whatever comes back — same frontend as before, nothing to
change there. It refreshes automatically on every page load/reload
and every 60 seconds after that.
"""
import os
import sys
import json
import pickle
import datetime
import webbrowser
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pandas as pd
import yfinance as yf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
INDEX_HTML = os.path.join(BASE_DIR, "index.html")
PORT = 5000

LIVE_BASKET = ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS",
               "BANKBARODA.NS", "PNB.NS", "IDFCFIRSTB.NS", "FEDERALBNK.NS", "INDUSINDBK.NS"]
NIFTY_BANK_TICKER = "^NSEBANK"
INDIA_VIX_TICKER = "^INDIAVIX"
USD_INR_TICKER = "INR=X"
SEQ_LEN = 30

print("Loading models and historical data once at startup — this takes a few seconds...")

# ---------- load everything once, not per-request ----------
with open(os.path.join(DATA_DIR, "feature_list.txt")) as f:
    FEATURE_COLS = [l.strip() for l in f if l.strip()]

_features_df = pd.read_csv(os.path.join(DATA_DIR, "features_india.csv"),
                            index_col=0, parse_dates=True).sort_index()
STATIC_DEFAULTS = _features_df[FEATURE_COLS].ffill().iloc[-1].to_dict()
LSTM_HISTORY_WINDOW = _features_df[FEATURE_COLS].ffill().iloc[-(SEQ_LEN - 1):].copy()


def _safe_load_pickle(path, label):
    """A corrupted or version-mismatched pickle for one model should
    never take down the whole server — this catches it, prints
    exactly what happened, and lets startup continue with that model
    simply unavailable."""
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"{label}: loaded OK")
        return obj
    except Exception as e:
        print(f"{label}: FAILED to load ({type(e).__name__}: {e}) — this model will be skipped, "
              f"not crash the server. If this is xgboost.pkl specifically, re-download it — "
              f"pickle files are binary and can get corrupted in transit (line-ending conversion, "
              f"a partial zip extraction, etc.).")
        return None


LR_MODEL = _safe_load_pickle(os.path.join(MODELS_DIR, "logistic_regression.pkl"), "Logistic Regression")
LR_SCALER = _safe_load_pickle(os.path.join(MODELS_DIR, "scaler_lr.pkl"), "LR scaler")
RF_MODEL = _safe_load_pickle(os.path.join(MODELS_DIR, "random_forest_binary.pkl"), "Random Forest")

# XGBoost: prefer the portable JSON format (XGBoost's own docs recommend
# this over pickle — pickle embeds a raw internal buffer that's fragile
# across versions/platforms/transfers, which is almost certainly what
# happened if you hit "input stream corrupted" loading xgboost.pkl).
XGB_MODEL = None
_xgb_json_path = os.path.join(MODELS_DIR, "xgboost_model.json")
if os.path.exists(_xgb_json_path):
    try:
        import xgboost as xgb
        XGB_MODEL = xgb.XGBClassifier()
        XGB_MODEL.load_model(_xgb_json_path)
        print("XGBoost: loaded OK (JSON format)")
    except Exception as e:
        print(f"XGBoost: FAILED to load JSON format ({type(e).__name__}: {e})")
if XGB_MODEL is None:
    XGB_MODEL = _safe_load_pickle(os.path.join(MODELS_DIR, "xgboost.pkl"), "XGBoost (pickle fallback)")

LSTM_MODEL = None
LSTM_SCALER = None
try:
    import torch
    import torch.nn as nn

    class BankingLSTM(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                 batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return torch.sigmoid(self.fc(out[:, -1, :])).squeeze(-1)

    with open(os.path.join(MODELS_DIR, "scaler_lstm.pkl"), "rb") as f:
        LSTM_SCALER = pickle.load(f)
    LSTM_MODEL = BankingLSTM(len(FEATURE_COLS), 64, 2, 0.3)
    LSTM_MODEL.load_state_dict(torch.load(os.path.join(MODELS_DIR, "lstm_model.pt"), map_location="cpu"))
    LSTM_MODEL.eval()
    print("LSTM loaded — will be live-scored.")
except Exception as e:
    print(f"LSTM not available for live scoring ({e}) — will show benchmark-only.")

print(f"Ready. {len(FEATURE_COLS)} features, {len(LIVE_BASKET)}-bank live basket.\n")


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
                       progress=False, threads=True, timeout=10)
    out = {}
    for t in all_tickers:
        s = _closes(raw, t)
        if s is not None and len(s) > 1:
            out[t] = s
    return out


def build_live_row():
    row = dict(STATIC_DEFAULTS)
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


def score_flat_models(row):
    X = pd.DataFrame([row])[FEATURE_COLS]
    scores = {}
    try:
        scores["Logistic Regression"] = float(LR_MODEL.predict_proba(LR_SCALER.transform(X))[0, 1])
    except Exception:
        scores["Logistic Regression"] = None
    try:
        scores["Random Forest"] = float(RF_MODEL.predict_proba(X)[0, 1])
    except Exception:
        scores["Random Forest"] = None
    try:
        scores["XGBoost"] = float(XGB_MODEL.predict_proba(X)[0, 1])
    except Exception:
        scores["XGBoost"] = None
    return scores


def score_lstm(row):
    if LSTM_MODEL is None:
        return None
    try:
        import torch
        window_df = pd.concat([LSTM_HISTORY_WINDOW, pd.DataFrame([row])[FEATURE_COLS]], ignore_index=True)
        window = window_df.values.astype(np.float32)  # (30, n_features) — last row is today's live pull
        scaled = LSTM_SCALER.transform(window)
        Xt = torch.tensor(scaled[np.newaxis, :, :], dtype=torch.float)
        with torch.no_grad():
            prob = LSTM_MODEL(Xt).item()
        return float(prob)
    except Exception as e:
        print(f"LSTM scoring failed: {e}")
        return None


def compute_payload():
    row, live_keys, fetch_error = build_live_row()
    scores = score_flat_models(row)
    lstm_score = score_lstm(row)
    if lstm_score is not None:
        scores["LSTM"] = lstm_score
    return {
        "probability": scores.get("Logistic Regression"),
        "model_scores": scores,
        "live_feature_count": len(live_keys),
        "live_feature_keys": live_keys,
        "feature_row": {k: row.get(k) for k in FEATURE_COLS},
        "source": "yfinance (Yahoo Finance), live pull — local server",
        "basket": LIVE_BASKET,
        "fetch_error": fetch_error,
        "computed_at": datetime.datetime.utcnow().isoformat() + "Z",
        "note": "Random Forest and XGBoost use the same live-basket features as Logistic "
                "Regression. LSTM scores a 30-day window ending in today's live pull. GNN is "
                "not re-scored live — see run_local.py docstring for why.",
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(INDEX_HTML, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/api/live_score"):
            try:
                payload = compute_payload()
                self._json(payload, 200)
            except Exception as e:
                self._json({"error": str(e), "error_type": type(e).__name__}, 200)
        else:
            self.send_response(404)
            self.end_headers()


def main():
    server = ThreadingHTTPServer(("localhost", PORT), Handler)
    url = f"http://localhost:{PORT}"
    print(f"Serving the dashboard at {url}")
    print("Live inference: Logistic Regression, Random Forest, XGBoost"
          + (", LSTM" if LSTM_MODEL is not None else "") + " — refreshed on every page load and every 60s.")
    print("Press Ctrl+C to stop.\n")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
        server.shutdown()


if __name__ == "__main__":
    main()
