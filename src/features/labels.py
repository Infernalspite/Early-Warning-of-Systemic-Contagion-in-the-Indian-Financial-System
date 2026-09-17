"""
src/features/labels.py
======================
Constructs binary and continuous systemic crisis target labels
for the 6 historical Indian financial stress events:
1. 2008 Global Financial Crisis
2. 2013 Taper Tantrum
3. 2018 IL&FS Collapse
4. 2020 YES Bank Moratorium
5. 2020 COVID Market Crash
6. 2023 Adani / Hindenburg Crisis
"""

import os
import pathlib
import yaml
import pandas as pd
import numpy as np

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def build_crisis_labels(dates_index):
    cfg = load_config()
    proc_dir = ROOT_DIR / cfg["paths"]["data_processed"]
    os.makedirs(proc_dir, exist_ok=True)

    labels_df = pd.DataFrame(index=dates_index)
    labels_df.index.name = "Date"
    labels_df["label"] = 0
    labels_df["high_stress_next_30d"] = 0
    labels_df["crisis_name"] = "Normal"
    labels_df["continuous_stress_score"] = 0.0

    crises = cfg["crises"]

    for c in crises:
        event_dt = pd.to_datetime(c["event_date"])
        start_dt = pd.to_datetime(c["label_start"])
        c_name = c["name"]

        # Binary label: 1 during pre-crisis 30-day window
        mask = (labels_df.index >= start_dt) & (labels_df.index <= event_dt)
        labels_df.loc[mask, "high_stress_next_30d"] = 1
        labels_df.loc[mask, "label"] = 1
        labels_df.loc[mask, "crisis_name"] = c_name

        # Mark peak crisis date with label 2
        peak_mask = (labels_df.index == event_dt)
        labels_df.loc[peak_mask, "label"] = 2

    # Continuous label: Exponential decay distance score to nearest future crisis date
    crisis_dates = [pd.to_datetime(c["event_date"]) for c in crises]

    continuous_scores = []
    for dt in labels_df.index:
        # Distance to next upcoming crisis
        future_crises = [cd for cd in crisis_dates if cd >= dt]
        if future_crises:
            days_to_next = (min(future_crises) - dt).days
            if days_to_next <= 60:
                # Exponential decay score: 1.0 at crisis date, decaying as distance increases
                score = np.exp(-days_to_next / 15.0)
            else:
                score = 0.0
        else:
            score = 0.0
        continuous_scores.append(score)

    labels_df["continuous_stress_score"] = continuous_scores

    labels_df.to_csv(proc_dir / "crisis_labels_india.csv")
    print(f"Crisis Labels saved -> {proc_dir / 'crisis_labels_india.csv'} | Shape: {labels_df.shape}")
    print(f"  Positive Class Count (high_stress_next_30d): {labels_df['high_stress_next_30d'].sum()} / {len(labels_df)}")

    return labels_df

if __name__ == "__main__":
    cfg = load_config()
    dates = pd.date_range(start=cfg["dates"]["start"], end=cfg["dates"]["end"], freq="B")
    build_crisis_labels(dates)
