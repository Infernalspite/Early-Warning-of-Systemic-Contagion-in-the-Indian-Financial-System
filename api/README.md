# Live scoring backend (Tier 2) — optional

The dashboard (`web/index.html`) works fully without this. It always
tries `GET /api/live_score` first; if that 404s (which it will until
you deploy this), it falls back automatically to a client-side
partial live score (Tier 1) using the real trained Logistic
Regression weights and a genuinely live USD/INR read. That fallback
is disclosed on the page itself — nothing is silently faked.

This folder is what upgrades that to **real, full backend
re-inference across three models, scraped live from Yahoo Finance.**

## Data source: yfinance, no API key

This pulls live daily data straight from Yahoo Finance via the
`yfinance` package — the same source `feature_engineering.py` used
to build the original historical dataset in this repo. There's no
signup, no key, nothing to configure. The tradeoff: `yfinance` scrapes
an unofficial Yahoo endpoint, so it can occasionally rate-limit or
change shape without notice. When that happens this endpoint doesn't
error out — it just returns fewer live features and falls back to
the last known dataset value for the rest, which shows up honestly
in the dashboard's live feature snapshot table (tagged "training
avg" instead of "live").

This has to run **server-side**. Yahoo's endpoints don't send CORS
headers, so a browser can't call them directly — that's why this is
a Vercel function and not more client-side JavaScript.

## Why only 3 of 5 models, and why a reduced bank basket

Being upfront about the real constraints, not glossing over them:

- **The full benchmark uses 20 bank tickers.** Pulling live history
  for all 20 on every request adds latency for very little extra
  signal — five large, liquid banks (HDFC, ICICI, SBI, Axis, Kotak)
  move together closely enough during actual stress events to serve
  as a live basket. `LIVE_BASKET` in `live_score.py` is a plain list
  if you want to change it.
- **LSTM and GNN are not re-scored live.** Bundling PyTorch into a
  Vercel serverless function pushes past the free-tier deployment
  size limit, and the GNN specifically needs same-day graphs across
  all 20 banks — a reduced basket can't substitute for it honestly.
  Both stay benchmark-only even with this deployed.
- **CoVaR / SRISK / MES / Granger causality are NOT recomputed live**
  in this function — they're expensive (this is why
  `feature_engineering.py` takes ~15 minutes locally). This endpoint
  reuses their last computed values from
  `data/processed/features_india.csv` and only refreshes the columns
  that are cheap per-request: returns, volatility, a live basket
  correlation estimate, USD/INR, India VIX, and Nifty Bank
  technicals (5-day return, 30-day drawdown, RSI).

If you present this, say what it actually is: a genuine live partial
re-score across three lightest models, on a reduced feature set, not
a full live recomputation of the whole 39-column pipeline.

## Setup

1. `pip install -r api/requirements.txt`
2. Deploy (`vercel deploy` or via the Vercel dashboard/GitHub
   integration). Vercel auto-detects `api/live_score.py` as a Python
   serverless function — no environment variables needed.
3. Open the dashboard. The live panel will switch from "Tier 1 ·
   client-side" to "Tier 2 · full backend" automatically once it can
   reach `/api/live_score`, and the model pills for Random Forest and
   XGBoost will light up.

## Testing locally

```bash
python3 -c "
from api.live_score import handler
import json
print(json.dumps(handler(None), indent=2)[:2000])
"
```

If Yahoo Finance is unreachable (network, rate limit, or a changed
endpoint shape), this still returns a 200 with whatever live columns
it did manage to pull — check `live_feature_keys` in the response to
see how many actually came through, rather than assuming it's all
39 or none.

## Cold starts

`yfinance`'s first call inside a fresh serverless instance is
usually the slowest one (session setup + Yahoo's own latency). If
you hit Vercel's execution time limit on the free/hobby plan, trim
`LIVE_BASKET` in `live_score.py` down to two or three tickers — the
correlation feature needs at least two to compute at all.
