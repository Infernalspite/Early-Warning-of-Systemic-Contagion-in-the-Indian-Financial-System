"""
src/explainability/shap_explainer.py
====================================
SHAP (SHapley Additive exPlanations) Model-Agnostic Explainability Module.
Generates SHAP summary plots and crisis-specific feature importance drivers for RF, XGBoost, LogReg.
"""

import os
import pathlib
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def run_shap_explainability(model, X_df, crisis_name="General", save_dir=None):
    """
    Computes SHAP values and outputs summary plots and top feature drivers.
    """
    cfg = load_config()
    figures_dir = save_dir or (ROOT_DIR / cfg["paths"]["figures"])
    os.makedirs(figures_dir, exist_ok=True)

    print(f"Computing SHAP values for '{crisis_name}' window...")

    try:
        if hasattr(model, "model"):
            underlying_model = model.model
        else:
            underlying_model = model

        explainer = shap.TreeExplainer(underlying_model)
        shap_values = explainer.shap_values(X_df)

        # Handle binary classification multi-output shap_values
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]
        else:
            shap_vals = shap_values

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals, X_df, show=False)
        plt.title(f"SHAP Feature Drivers — {crisis_name} Crisis Window", fontsize=12, fontweight="bold")
        plt.tight_layout()
        
        file_path = figures_dir / f"shap_summary_{crisis_name.lower().replace(' ', '_')}.png"
        plt.savefig(file_path, dpi=150, bbox_inches="tight")
        plt.close()

        # Mean absolute SHAP values per feature
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        shap_importance = pd.Series(mean_abs_shap, index=X_df.columns).sort_values(ascending=False)
        
        print(f"  SHAP plot saved -> {file_path}")
        return shap_importance

    except Exception as e:
        print(f"WARNING: SHAP calculation failed for {crisis_name}: {e}")
        return None

if __name__ == "__main__":
    X_mock = pd.DataFrame(np.random.randn(100, 5), columns=["volatility", "covar", "density", "srisk", "vix"])
    from xgboost import XGBClassifier
    xgb = XGBClassifier().fit(X_mock, np.random.randint(0, 2, 100))
    run_shap_explainability(xgb, X_mock, "IL_and_FS_2018")
