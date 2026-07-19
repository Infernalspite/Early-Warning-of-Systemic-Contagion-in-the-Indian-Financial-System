# Deployment Guide: Frontend + Python ML Backend

Due to the heavy machine learning dependencies (like `scikit-learn` and `xgboost` totaling >800MB uncompressed), Vercel's Hobby/Free tier 500MB function size limit will block the backend from building there. 

To keep 100% of the repository's ML model capability intact without stripping down the models, we deploy:
1. **Frontend (Static)**: Vercel or Render Static (free, fast, zero configuration).
2. **Backend (API Service)**: Render Web Services (no size limits, fully supports scikit-learn/xgboost).

---

## 1. Deploy the Backend on Render

Render fully supports Python services without size constraints.

1. Go to [Render](https://render.com) and log in.
2. Click **New** -> **Web Service**.
3. Connect your GitHub repository.
4. Set the following configuration:
   - **Name**: `early-warning-systemic-contagion`
   - **Runtime**: `Python`
   - **Build Command**: `pip install -r requirements.txt` (This installs numpy, pandas, sklearn, xgboost, yfinance)
   - **Start Command**: `python api/live_score.py`
5. Click **Deploy Web Service**.
6. Render will assign you a URL like `https://early-warning-systemic-contagion.onrender.com`. Copy this URL!

---

## 2. Link Frontend to Render Backend

1. In the repository, open `index.html` at the root and `web/index.html`.
2. Find the constant `BACKEND_API_FALLBACK` near line 730:
   ```javascript
   const BACKEND_API_FALLBACK = 'https://early-warning-systemic-contagion.onrender.com';
   ```
3. Update this URL to match your deployed Render URL.
4. Commit and push the change to GitHub:
   ```bash
   git add -A
   git commit -m "Update fallback Render API URL"
   git push
   ```

---

## 3. Deploy the Frontend on Vercel

With `vercel.json` removed, Vercel zero-config will deploy the root `index.html` as a static page instantly:

1. Run:
   ```bash
   vercel deploy
   ```
2. Open your Vercel deployment URL.
3. The frontend will first probe Vercel's path `/api/live_score`, and when it fails (since we didn't deploy the backend there), it will seamlessly fallback to your Render API URL.
4. The live tier indicator will light up as **`Tier 2 · full backend`** and all three models (Logistic Regression, Random Forest, XGBoost) will evaluate live on freshly scraped Yahoo Finance data!

---

## Verification Checklist

1. Open `https://your-render-url.onrender.com/api/live_score` directly in the browser. It should return a full JSON response with `model_scores`.
2. Open your Vercel frontend URL. Check that all benchmark graphs (ROC curves, heatmaps) render immediately.
3. Check the "Live Inference" panel. The tier status must say `Tier 2 · full backend` and the model pills should be active.
4. Click "Refresh now" to confirm a live refresh works.
