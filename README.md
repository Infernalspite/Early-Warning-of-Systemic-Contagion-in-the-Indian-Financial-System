# Early Warning of Systemic Contagion in the Indian Financial System

An end-to-end **Machine Learning powered Early Warning System** for detecting and visualising systemic contagion risk across the Indian banking sector.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Infernalspite/Early-Warning-of-Systemic-Contagion-in-the-Indian-Financial-System)

> **Full-stack deployment** — Flask serves both the interactive dashboard and live ML inference (`/api/live_score`) from one Render Web Service. All models (Logistic Regression, Random Forest, XGBoost, LSTM, GNN) are included with live data from Yahoo Finance.

---

## What It Does

This project monitors **20 Indian banks** (PSU + Private) using daily stock prices, RBI policy rates, India VIX, and Nifty indices — all sourced exclusively from Indian financial markets — and trains a Random Forest classifier to predict **three systemic risk regimes**:

| Label | Regime | CRI Score |
|---|---|---|
| 0 | Normal | 0 – 30 |
| 1 | Pre-Crisis (30-day warning) | 30 – 60 |
| 2 | Active Crisis | 60 – 100 |

The **Contagion Risk Index (CRI)** = `(P(Crisis) × 0.7 + P(Pre-Crisis) × 0.3) × 100`

---

## Live Dashboard Features

| Tab | Description |
|---|---|
| **CRI Overview** | Area chart of contagion risk 2014–2026 with crisis event annotations |
| **Contagion Network** | Animated D3.js interbank correlation graph — scrub timeline or hit Play |
| **Bank Comparison** | Normalised price comparison of all 20 banks with checkbox selectors |
| **Model Insights** | Feature importances and model specification details |

---

## Data Sources (India Only)

| Source | Data |
|---|---|
| **NSE / Yahoo Finance** | Nifty 50, Nifty Bank, Nifty Financial Services, Nifty IT, India VIX |
| **NSE / Yahoo Finance** | 20 bank stock prices (daily OHLCV) |
| **RBI Monetary Policy** | Policy Repo Rate (official MPC decisions 2014–2026) |
| **Forex Markets** | USD/INR exchange rate |

No US or global commodity data is used.

---

## Model Performance

- **Classifier**: Random Forest (200 trees, balanced class weights)
- **Train**: 2014–2021 | **Test**: 2022–2026
- **Accuracy**: 80.4%
- **Top Features**: INR/USD rate, RBI Repo Rate, HDFC Bank volatility, avg. pairwise bank correlation

---

## Historical Crises Captured

| Event | Date |
|---|---|
| IL&FS Shadow Banking Collapse | Sep 2018 |
| Yes Bank Moratorium | Mar 2020 |
| COVID-19 Market Crash | Mar 2020 |
| Adani Group Short-Seller Shock | Jan 2023 |

---

## Run Locally

```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn yfinance matplotlib seaborn

# 2. Download data
python download_data.py

# 3. Label crises
python crisis_labels.py

# 4. Engineer features
python feature_engineering.py

# 5. Train model & export dashboard data
python train_model.py

# 6. Launch dashboard
python server.py
# Open: http://localhost:8000
```

---

## Project Structure

```
├── download_data.py          # Downloads NSE / RBI data via yfinance
├── crisis_labels.py          # Labels dates as Normal / Pre-Crisis / Crisis
├── feature_engineering.py    # Builds 31 systemic risk features (PCA, RSI, etc.)
├── train_model.py            # Trains RF model + exports dashboard_data.json
├── server.py                 # Local static file server (port 8000)
│
├── data/
│   ├── raw/                  # Raw downloaded CSVs (not in git)
│   └── processed/            # Feature matrix + labels
│
├── web/                      # Dashboard (deployed to GitHub Pages)
│   ├── index.html
│   ├── style.css             # Glassmorphism dark theme
│   ├── app.js                # ApexCharts + D3.js interactivity
│   └── data/
│       └── dashboard_data.json   # Pre-computed model output
│
└── models/
    └── random_forest_india.pkl   # Trained classifier
```

---

## Tech Stack

**Backend / ML**: Python · pandas · NumPy · scikit-learn · yfinance

**Frontend**: Vanilla HTML/CSS/JS · [ApexCharts](https://apexcharts.com) · [D3.js v7](https://d3js.org) · Google Fonts

---

*Data sourced from NSE, RBI, and Indian financial markets only.*