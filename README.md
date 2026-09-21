# Early Warning of Systemic Contagion in the Indian Financial System

An end-to-end **Machine Learning & Graph Neural Network powered Early Warning System** for detecting, quantifying, and visualising systemic contagion risk across the Indian banking and financial sector (2005-2026).

[![Live Dashboard](https://img.shields.io/badge/Live%20Dashboard-GitHub%20Pages-blue?style=for-the-badge&logo=github)](https://Infernalspite.github.io/Early-Warning-of-Systemic-Contagion-in-the-Indian-Financial-System)
[![Daily Auto-Update](https://img.shields.io/badge/Auto--Update-Daily%206%20PM%20IST-brightgreen?style=for-the-badge&logo=githubactions)](https://github.com/Infernalspite/Early-Warning-of-Systemic-Contagion-in-the-Indian-Financial-System/actions)

---

## Executive Summary

Systemic financial contagion occurs when localized distress in one institution propagates through interbank credit channels, market correlations, and liquidity spillovers. This project implements a comprehensive **Early Warning System (EWS)** specifically calibrated for the **Indian Financial System**.

- **5,359 Trading Days Evaluated** (Jan 2005 - Sep 2026)
- **20 Key Indian Banks Covered** (PSU + Private Sector Basket)
- **FinBERT NLP Sentiment Analysis** on 21 RBI Financial Stability & Policy Reports
- **6 Machine Learning Architectures Benchmarked** under a unified walk-forward split (2005-2021 Train | 2022-2026 Test)
- **Interactive Risk Calculator** running client-side Random Forest decision tree inference directly in the browser

---

## Live Interactive Dashboard Features

The dashboard features an **IBM Plex typography theme** with custom graphics, network force simulation, and real-time risk calculation across **7 distinct sections**:

| Section | Key Features |
|---|---|
| **Overview (Home)** | Hero metrics, Contagion Risk Index (CRI) gauge, market status pulse, quick risk regime breakdown |
| **Contagion Network** | Dynamic force-directed interbank correlation network graph with threshold controls and playback |
| **CRI Timeline** | 2005-2026 Contagion Risk Index timeline chart annotated with historic Indian financial crisis episodes |
| **Bank Stock Prices** | Interactive rebased price comparison (Base = 100) and raw INR prices across all 20 banks |
| **Correlation Matrix** | Heatmap matrix comparing bank return correlations across distinct historical stress regimes (2008 GFC, 2018 IL&FS, 2020 COVID, 2023-2026 Recent) |
| **Risk Calculator** | Interactive scenario simulator: tweak macroeconomic indicators and run client-side RF inference live in the browser |
| **Model & Data** | System specifications, 23 canonical feature importances, model comparison benchmark table, and risk regime taxonomy |

---

## Standardized Model Comparison Benchmark

All models were evaluated on the **exact same 23 canonical features** (`data/processed/features.csv`) using an out-of-sample time-based split (**Train: 2005-2021 | Test: 2022-2026**, 1,165 test trading days):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Primary Strength |
|---|---|---|---|---|---|---|---|
| **Dynamic GNN (GraphSAGE)** | **95.60%** | 0.744 | 0.612 | **0.671** | 0.9210 | **0.847** | **Best Overall F1 & Network Topology modeling** |
| **Logistic Regression** | 95.71% | 0.386 | **1.000** | 0.557 | 0.9850 | 0.812 | **Best Early Warning Lead-Time (100% Recall)** |
| **Voting Ensemble** | **97.94%** | 0.700 | 0.438 | 0.538 | 0.9846 | 0.825 | High overall accuracy |
| **XGBoost Classifier** | 97.77% | **1.000** | 0.188 | 0.316 | **0.9984** | 0.831 | Zero false positives |
| **Random Forest** | 94.51% | 0.280 | 0.219 | 0.246 | 0.9135 | 0.762 | Interpretable tree rules & Feature Importance |
| **LSTM Recurrent Net** | 97.18% | 0.000 | 0.000 | 0.000 | 0.9231 | 0.745 | Temporal sequence modeling |

*Note: In systemic risk early warning, **F1-Score, Recall, and ROC-AUC** are prioritized over raw accuracy due to the 2.7% class imbalance of true crisis days.*

---

## Historical Crisis Regimes Captured

| Event | Date | Observed CRI Peak | Key Risk Drivers |
|---|---|---|---|
| **Global Financial Crisis (GFC)** | Oct 2008 | **64.0 (High Risk)** | US VIX spillover, global liquidity squeeze |
| **IL&FS Shadow Banking Crisis** | Sep 2018 | **62.9 (High Risk)** | NBFC asset-liability mismatch, MIBOR spread spike |
| **YES Bank Moratorium** | Mar 2020 | **75.0 (High Risk)** | Single-node counterparty failure, credit default fear |
| **COVID-19 Market Crash** | Mar 2020 | **83.9 (Critical)** | Simultaneous market-wide volatility and liquidity freeze |
| **Adani / Hindenburg Shock** | Jan 2023 | **64.3 (High Risk)** | Sectoral contagion and governance panic |

---

## Automated Daily Pipeline

The project includes an automated **GitHub Actions workflow** (`.github/workflows/update_dashboard.yml`):
- Runs every weekday at **6:00 PM IST** (after NSE market close at 3:30 PM IST).
- Downloads fresh bank stock closes and macro indicators via `yfinance`.
- Recomputes the Contagion Risk Index and updates the correlation network.
- Auto-commits and deploys the updated dashboard to GitHub Pages automatically.

---

## Local Setup & Execution

```bash
# 1. Clone the repository
git clone https://github.com/Infernalspite/Early-Warning-of-Systemic-Contagion-in-the-Indian-Financial-System.git
cd Early-Warning-of-Systemic-Contagion-in-the-Indian-Financial-System

# 2. Install dependencies
pip install -r requirements.txt

# 3. Regenerate dashboard data and frontend interface
python build_dashboard_json.py

# 4. Launch local server
python -m http.server 8080 --directory web
# Open: http://localhost:8080
```

---

## Project Architecture

```
.github/workflows/
    update_dashboard.yml  # Daily GitHub Actions auto-update workflow
config/
    config.yaml           # System configuration & API keys
data/
    raw/                  # Bank stock prices, macro data, RBI PDFs
    processed/            # Master feature matrix (features.csv)
models/                   # Trained model artifacts & metrics
web/                      # Production frontend (HTML, CSS, JS)
    index.html
    data/
        dashboard_data.json
docs/                     # GitHub Pages deployment directory
build_dashboard_json.py   # Primary pipeline orchestrator & JSON builder
build_new_frontend.py     # Frontend HTML template compiler
evaluate_all_models.py    # Benchmark evaluation runner for all 6 models
PROJECT_REPORT.md         # Full academic write-up & technical documentation
```

---

*Data sourced from NSE, RBI, FRED, and Indian financial markets.*
