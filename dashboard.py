"""
dashboard.py
============
Live Streamlit dashboard for the
Indian Systemic Risk Contagion Engine

Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle
import os

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title = "🇮🇳 Indian Risk Engine",
    page_icon  = "🏦",
    layout     = "wide"
)

# ── LOAD DATA ────────────────────────────────────────────────
@st.cache_data
def load_all_data():

    features = pd.read_csv(
        "data/processed/features_india.csv",
        index_col=0, parse_dates=True
    )
    features.index = pd.to_datetime(
        features.index, dayfirst=True
    )

    prices = pd.read_csv(
        "data/raw/bank_prices_nse.csv",
        index_col=0, parse_dates=True
    )
    prices.index = pd.to_datetime(
        prices.index, dayfirst=True
    )

    indices = pd.read_csv(
        "data/raw/nifty_indices.csv",
        index_col=0, parse_dates=True
    )
    indices.index = pd.to_datetime(
        indices.index, dayfirst=True
    )

    labels = pd.read_csv(
        "data/labels/crisis_labels_india.csv",
        index_col=0, parse_dates=True
    )
    labels.index = pd.to_datetime(
        labels.index, dayfirst=True
    )

    return features, prices, indices, labels

@st.cache_resource
def load_model():
    with open("models/random_forest_india.pkl", "rb") as f:
        return pickle.load(f)

features, prices, indices, labels = load_all_data()
model = load_model()

# ── FEATURE COLUMNS ──────────────────────────────────────────
FEATURE_COLS = [
    "avg_volatility_5d", "max_volatility_5d",
    "avg_volatility_10d", "max_volatility_10d",
    "avg_volatility_30d", "max_volatility_30d",
    "avg_return_5d", "min_return_5d",
    "avg_return_10d", "min_return_10d",
    "avg_return_30d", "min_return_30d",
    "network_density_06", "india_vix",
    "india_vix_change", "nifty_bank_return_5d",
    "inr_usd", "inr_usd_change",
    "us_vix", "crude_oil_change",
    "yes_bank_vol_10d", "yes_bank_return_5d",
    "sbi_vol_10d", "hdfc_vol_10d",
]
FEATURE_COLS = [c for c in FEATURE_COLS
                if c in features.columns]

# ── COMPUTE CRI SCORE ────────────────────────────────────────
# CRI = Contagion Risk Index (0-100)
# Based on model probability of crisis

X_all  = features[FEATURE_COLS].dropna()
probs  = model.predict_proba(X_all)
labels_list = list(model.classes_)

# Get probability of Pre-Crisis + Crisis
if 2 in labels_list:
    crisis_prob = probs[:, labels_list.index(2)]
else:
    crisis_prob = probs[:, -1]

if 1 in labels_list:
    pre_prob = probs[:, labels_list.index(1)]
else:
    pre_prob = np.zeros(len(crisis_prob))

# CRI = weighted sum scaled to 0-100
cri = (crisis_prob * 0.7 + pre_prob * 0.3) * 100
cri_series = pd.Series(cri, index=X_all.index)

# ── CRISIS EVENTS ────────────────────────────────────────────
CRISIS_EVENTS = {
    "IL&FS Crisis"    : "2018-09-21",
    "Yes Bank Crisis" : "2020-03-05",
    "COVID Crash"     : "2020-03-23",
    "Adani Crisis"    : "2023-01-24",
}

# ════════════════════════════════════════════════════════════
# DASHBOARD LAYOUT
# ════════════════════════════════════════════════════════════

# ── HEADER ───────────────────────────────────────────────────
st.title("🇮🇳 Indian Systemic Risk Contagion Engine")
st.markdown(
    "**Real-time early warning system for Indian banking crises** "
    "| 20 Banks | 2014–2026 | Random Forest Model"
)
st.divider()

# ── CURRENT RISK STATUS ──────────────────────────────────────
latest_cri   = cri_series.iloc[-1]
latest_date  = cri_series.index[-1].strftime("%d %b %Y")

# Determine risk level
if latest_cri >= 60:
    risk_level = "🔴 HIGH RISK"
    risk_color = "red"
elif latest_cri >= 30:
    risk_level = "🟡 MEDIUM RISK"
    risk_color = "orange"
else:
    risk_level = "🟢 LOW RISK"
    risk_color = "green"

# Top metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label = "🎯 Current CRI Score",
        value = f"{latest_cri:.1f} / 100",
        delta = f"{latest_cri - cri_series.iloc[-6]:.1f} vs 5 days ago"
    )

with col2:
    st.metric(
        label = "⚠️ Risk Level",
        value = risk_level
    )

with col3:
    latest_vix = features["india_vix"].iloc[-1] \
        if "india_vix" in features.columns else "N/A"
    st.metric(
        label = "😰 India VIX",
        value = f"{latest_vix:.1f}" \
            if isinstance(latest_vix, float) else latest_vix
    )

with col4:
    latest_inr = features["inr_usd"].iloc[-1] \
        if "inr_usd" in features.columns else "N/A"
    st.metric(
        label = "💱 INR/USD",
        value = f"₹{latest_inr:.2f}" \
            if isinstance(latest_inr, float) else latest_inr
    )

st.divider()

# ── TABS ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 CRI Score",
    "🏦 Bank Prices",
    "🔗 Correlations",
    "📊 Model Info"
])

# ════════════════════════════════════════════════════════════
# TAB 1 — CRI SCORE OVER TIME
# ════════════════════════════════════════════════════════════

with tab1:

    st.subheader("Contagion Risk Index (CRI) — 2014 to Present")
    st.markdown(
        "CRI ranges from **0 (no risk)** to **100 (extreme risk)**. "
        "Red shading = actual crisis periods."
    )

    # Date range selector
    col_a, col_b = st.columns(2)
    with col_a:
        start_year = st.selectbox(
            "From Year",
            options=list(range(2014, 2027)),
            index=0
        )
    with col_b:
        end_year = st.selectbox(
            "To Year",
            options=list(range(2014, 2027)),
            index=12
        )

    # Filter CRI to selected range
    cri_filtered = cri_series[
        (cri_series.index.year >= start_year) &
        (cri_series.index.year <= end_year)
    ]

    fig, ax = plt.subplots(figsize=(14, 5))

    # Plot CRI
    ax.plot(cri_filtered.index, cri_filtered.values,
            color="darkred", linewidth=1.5,
            label="CRI Score")
    ax.fill_between(cri_filtered.index,
                    cri_filtered.values,
                    alpha=0.2, color="red")

    # Risk threshold lines
    ax.axhline(y=60, color="red", linestyle="--",
               linewidth=1, alpha=0.7, label="High Risk (60)")
    ax.axhline(y=30, color="orange", linestyle="--",
               linewidth=1, alpha=0.7, label="Medium Risk (30)")

    # Shade actual crisis periods
    for event, date in CRISIS_EVENTS.items():
        crisis_ts = pd.Timestamp(date)
        if (crisis_ts.year >= start_year and
                crisis_ts.year <= end_year):
            ax.axvline(x=crisis_ts, color="black",
                       linestyle=":", linewidth=1.5,
                       alpha=0.7)
            ax.text(crisis_ts, 92, event,
                    rotation=90, fontsize=7,
                    color="black", va="top")

    ax.set_ylabel("CRI Score (0-100)")
    ax.set_xlabel("Date")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title(
        f"Indian Contagion Risk Index ({start_year}–{end_year})",
        fontsize=12, fontweight="bold"
    )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Show highest risk days
    st.subheader("🚨 Top 10 Highest Risk Days")
    top_risk = cri_series.nlargest(10).reset_index()
    top_risk.columns = ["Date", "CRI Score"]
    top_risk["Date"] = top_risk["Date"].dt.strftime("%d %b %Y")
    top_risk["CRI Score"] = top_risk["CRI Score"].round(1)
    top_risk["Risk Level"] = top_risk["CRI Score"].apply(
        lambda x: "🔴 HIGH" if x >= 60
        else "🟡 MEDIUM" if x >= 30
        else "🟢 LOW"
    )
    st.dataframe(top_risk, use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 2 — BANK PRICES
# ════════════════════════════════════════════════════════════

with tab2:

    st.subheader("Indian Bank Stock Prices")

    # Bank selector
    selected_banks = st.multiselect(
        "Select banks to compare:",
        options=list(prices.columns),
        default=["State Bank of India",
                 "HDFC Bank",
                 "Yes Bank",
                 "ICICI Bank"]
    )

    if selected_banks:
        # Normalize to 100
        selected_prices = prices[selected_banks].dropna()
        normalized = (selected_prices /
                      selected_prices.iloc[0]) * 100

        fig, ax = plt.subplots(figsize=(14, 5))

        colors = plt.cm.tab10(
            np.linspace(0, 1, len(selected_banks))
        )
        for bank, color in zip(selected_banks, colors):
            ax.plot(normalized.index, normalized[bank],
                    linewidth=1.5, label=bank, color=color)

        # Crisis lines
        for event, date in CRISIS_EVENTS.items():
            ax.axvline(x=pd.Timestamp(date),
                       color="red", linestyle="--",
                       linewidth=1, alpha=0.6)
            ax.text(pd.Timestamp(date),
                    normalized.values.max() * 0.95,
                    event, rotation=90,
                    fontsize=7, color="red", va="top")

        ax.set_title("Bank Prices — Normalized to 100",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("Normalized Price")
        ax.set_xlabel("Date")
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%Y")
        )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Latest prices table
        st.subheader("📋 Latest Prices")
        latest = prices[selected_banks].tail(1).T
        latest.columns = ["Latest Price (INR)"]
        latest["Latest Price (INR)"] = latest[
            "Latest Price (INR)"
        ].round(2)
        st.dataframe(latest, use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 3 — CORRELATIONS
# ════════════════════════════════════════════════════════════

with tab3:

    st.subheader("Bank Correlation Heatmap")
    st.markdown(
        "Shows how closely Indian banks move together. "
        "**Higher correlation = higher contagion risk.**"
    )

    # Period selector
    period = st.selectbox(
        "Select period:",
        ["Full Period (2014-2026)",
         "IL&FS Crisis (2018)",
         "COVID Crash (2020)",
         "Recent (2023-2026)"]
    )

    period_map = {
        "Full Period (2014-2026)" : ("2014-01-01", "2026-12-31"),
        "IL&FS Crisis (2018)"     : ("2018-06-01", "2018-12-31"),
        "COVID Crash (2020)"      : ("2020-01-01", "2020-06-30"),
        "Recent (2023-2026)"      : ("2023-01-01", "2026-12-31"),
    }

    p_start, p_end = period_map[period]

    returns_raw = pd.read_csv(
        "data/processed/bank_returns_nse.csv",
        index_col=0, parse_dates=True
    )
    returns_raw.index = pd.to_datetime(
        returns_raw.index, dayfirst=True
    )

    returns_period = returns_raw[
        (returns_raw.index >= p_start) &
        (returns_raw.index <= p_end)
    ]

    corr = returns_period.corr()

    import seaborn as sns
    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True,
        fmt=".2f", cmap="RdYlGn",
        center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.5,
        ax=ax, annot_kws={"size": 7}
    )
    ax.set_title(
        f"Bank Return Correlations — {period}",
        fontsize=12, fontweight="bold"
    )
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    avg_corr = corr.where(
        ~np.eye(len(corr), dtype=bool)
    ).stack().mean()
    st.metric(
        "Average Pairwise Correlation",
        f"{avg_corr:.3f}",
        help="Higher = more dangerous homogeneity"
    )

# ════════════════════════════════════════════════════════════
# TAB 4 — MODEL INFO
# ════════════════════════════════════════════════════════════

with tab4:

    st.subheader("🤖 Model Information")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Model Details")
        st.markdown("""
        - **Model Type**: Random Forest Classifier
        - **Training Period**: 2014 – 2021
        - **Test Period**: 2022 – 2026
        - **Accuracy**: 82.3%
        - **Features**: 24 input features
        - **Labels**: 0=Normal, 1=Pre-Crisis, 2=Crisis
        """)

    with col2:
        st.markdown("### Crisis Events in Training Data")
        for event, date in CRISIS_EVENTS.items():
            st.markdown(f"- 🚨 **{event}** ({date})")

    st.markdown("### 📌 Feature List")
    feat_df = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Category": [
            "Volatility" if "vol" in f
            else "Momentum" if "return" in f
            else "Network" if "network" in f or "corr" in f
            else "Macro"
            for f in FEATURE_COLS
        ]
    })
    st.dataframe(feat_df, use_container_width=True)

    st.markdown("### 📁 Data Sources")
    st.markdown("""
    | Source | Data |
    |--------|------|
    | Yahoo Finance (NSE) | 20 Indian bank stock prices |
    | Yahoo Finance | Nifty 50, Nifty Bank, India VIX |
    | Yahoo Finance | INR/USD, Crude Oil, Gold, US VIX |
    | Custom Labels | IL&FS, Yes Bank, COVID, Adani crises |
    """)

# ── FOOTER ───────────────────────────────────────────────────
st.divider()
st.markdown(
    "🏦 **Indian Systemic Risk Contagion Engine** | "
    "Built with Python, scikit-learn & Streamlit | "
    "Data: Yahoo Finance / NSE"
)