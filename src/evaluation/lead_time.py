"""
src/evaluation/lead_time.py
===========================
Computes early warning lead-times (days of advance warning) for each model across all 6 crisis events.
Generates lead-time charts and comparative lead-time summary tables.
"""

import os
import pathlib
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def compute_lead_times(model_probs_dict, dates_index, threshold=0.4):
    """
    Calculates number of trading days of advance warning before each crisis event date.
    """
    cfg = load_config()
    crises = cfg["crises"]

    lead_time_records = []

    for c in crises:
        event_dt = pd.to_datetime(c["event_date"])
        start_window = event_dt - pd.Timedelta(days=60)

        for model_name, probs in model_probs_dict.items():
            prob_series = pd.Series(probs, index=dates_index)
            sub_window = prob_series[(prob_series.index >= start_window) & (prob_series.index <= event_dt)]
            
            # First date score crossed threshold
            crossed = sub_window[sub_window >= threshold]
            if len(crossed) > 0:
                first_alert_date = crossed.index[0]
                advance_days = (event_dt - first_alert_date).days
            else:
                advance_days = 0

            lead_time_records.append({
                "crisis_name": c["name"],
                "event_date": c["event_date"],
                "model": model_name,
                "advance_warning_days": advance_days
            })

    return pd.DataFrame(lead_time_records)

def plot_lead_time_chart(model_probs_dict, dates_index, save_path=None):
    """
    Plots continuous risk score timeline against the 6 crisis dates with lead-time callouts.
    """
    cfg = load_config()
    fig, ax = plt.subplots(figsize=(14, 6))

    crises = cfg["crises"]
    colors = ["#3b82f6", "#8b5cf6", "#10b981", "#f59e0b", "#ef4444", "#ec4899"]

    for i, (model_name, probs) in enumerate(model_probs_dict.items()):
        color = colors[i % len(colors)]
        ax.plot(dates_index, probs, label=model_name, color=color, alpha=0.85, linewidth=1.5)

    # Vertical crisis lines
    for c in crises:
        e_dt = pd.to_datetime(c["event_date"])
        ax.axvline(e_dt, color="red", linestyle="--", alpha=0.7)
        ax.text(e_dt, 0.95, c["name"], rotation=90, verticalalignment="top", fontsize=8, color="darkred", fontweight="bold")

    ax.axhline(y=0.4, color="black", linestyle=":", label="Warning Threshold (0.4)")
    ax.set_ylabel("Predicted Systemic Risk Score", fontsize=11)
    ax.set_title("Walk-Forward Lead-Time Analysis — Systemic Risk Warning Signals", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    return fig

if __name__ == "__main__":
    cfg = load_config()
    figures_dir = ROOT_DIR / cfg["paths"]["figures"]
    os.makedirs(figures_dir, exist_ok=True)
    
    dates = pd.date_range(start="2018-01-01", end="2023-12-31", freq="B")
    mock_probs = {"XGBoost": np.random.uniform(0.1, 0.6, len(dates)), "GNN": np.random.uniform(0.1, 0.7, len(dates))}
    plot_lead_time_chart(mock_probs, dates, save_path=figures_dir / "lead_time_chart.png")
    print("Lead time chart created successfully.")
