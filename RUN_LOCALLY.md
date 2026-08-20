# Run this locally — no Vercel, no GitHub, no deployment

```bash
pip install -r requirements_local.txt
python run_local.py
```

That's it. It starts a local server and opens `http://localhost:5000`
in your browser automatically.

## What you get

- The full dashboard: hero, feature matrix, correlation heatmap,
  model diagnostic tabs (confusion matrix / ROC / probability
  histogram / feature importance per model), frozen benchmark table,
  discussion.
- A **Live Inference** panel that pulls real data from Yahoo Finance
  (via `yfinance`, no key needed) and re-scores **4 of the 5 models**
  live: Logistic Regression, Random Forest, XGBoost, and LSTM. Switch
  between them with the pills above the score — each is a real,
  independently computed live probability, not a shared number.
- It refreshes automatically on every page load and every 60 seconds
  after that, plus a manual "Refresh now" button.

## Why not the GNN too

The other four models score a flat feature vector — one row in,
one probability out. The GNN needs a fresh correlation *graph*
across the bank basket rebuilt from live data, which is a
meaningfully bigger and slower piece of engineering than the other
four. It stays benchmark-only (the frozen historical result), and
the dashboard says so directly rather than faking a live number for
it.

## If something goes wrong

- **"Address already in use"** — something else is already using
  port 5000. Edit `PORT = 5000` near the top of `run_local.py` to a
  different number (e.g. 5050) and re-run.
- **The live score never updates / stays at "—"** — check the
  terminal window where you ran `python run_local.py`. If Yahoo
  Finance is unreachable (no internet, or Yahoo rate-limiting your
  IP), it'll print a message there and the page will show 0 live
  features rather than crashing — the benchmark table and diagnostics
  are unaffected either way, they don't depend on live data at all.
- **`ModuleNotFoundError`** — re-run
  `pip install -r requirements_local.txt`; if `torch` specifically
  fails to install, LSTM will just be skipped (the server prints
  "LSTM not available for live scoring") and the other three models
  still work fine.

## Stopping it

`Ctrl+C` in the terminal.
