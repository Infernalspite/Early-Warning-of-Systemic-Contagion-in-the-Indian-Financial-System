# Deploying this dashboard (read this before deploying to Vercel)

## The structure that makes this work

```
/index.html          ← the dashboard itself. MUST be at repo root.
/api/live_score.py    ← optional live backend, auto-detected by Vercel
/web/index.html       ← identical copy, kept for reference/non-Vercel hosting
```

There is **no `vercel.json`** in this repo on purpose. Vercel's zero-config
detection handles both pieces automatically:
- Any static file at the repo root is served as-is → `index.html` becomes
  your site.
- Any Python file under `/api` that follows Vercel's function interface
  becomes a serverless function automatically at `/api/<filename>`.

**If you had an earlier version of this repo with a `vercel.json`
using an explicit `builds` array pointing only at `web/**`** — that
was the actual bug behind "nothing works, no live data, no models
loading." An explicit `builds` list tells Vercel to build *only*
what's listed and disables auto-detection for everything else,
including `/api`. That config silently prevented the backend from
ever deploying, no matter what the Python code did. It's been
removed. Don't re-add a `vercel.json` unless you know you need one —
zero-config is more robust here.

## Deploy

```bash
vercel deploy
```
That's it. No environment variables, no build command, no output
directory setting.

## Verify it actually worked (do this every time you redeploy)

1. Open `https://your-domain/` — the dashboard should render fully:
   hero, feature matrix, correlation heatmap, model diagnostic tabs,
   results table, discussion. All of this is static/embedded data and
   works with zero network access, so if any of it is missing/blank,
   that's a deployment problem, not a live-data problem — check the
   Vercel build log.
2. Open `https://your-domain/api/live_score` directly in your
   browser. You should get a JSON body back (even `{"error": "..."}`
   counts as working — the function ran). A blank page or a Vercel
   error screen means the function itself isn't deploying.
3. Back on the dashboard, look at the "Live Inference" section's tier
   tag near the top of that panel. It will say one of:
   - `Tier 2 · full backend` — the Python function is live and all
     three eligible models (LogReg, RF, XGBoost) are re-scoring on
     real Yahoo Finance data.
   - `Tier 1 · client-side (partial)` — the backend isn't reachable,
     but the page still runs a real live score in your browser using
     live USD/INR data. This is expected if you haven't deployed
     `/api`, and is not a bug.
   - `Offline · no tier reachable` — both live paths failed (e.g. a
     network that blocks outbound fetches entirely). The rest of the
     dashboard (benchmark table, diagnostics, correlation heatmap)
     is unaffected either way — those never depended on live data.

## Why it refreshes on every load, and every 60 seconds after that

`runLiveScore()` runs immediately when the page's script executes —
which is every page load and every reload, there's no caching layer
in front of it. After that it's on a 60-second timer via
`startCountdown()`. There's also a manual "Refresh now" button in the
Live Inference panel if you don't want to wait. None of this uses
`localStorage`/`sessionStorage` — every reload starts clean.
