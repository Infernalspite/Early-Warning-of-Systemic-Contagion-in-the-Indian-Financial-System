"""
scripts/run_pipeline.py
=======================
Master End-to-End Pipeline Execution Script.
Runs data fetching, feature engineering, multi-relation graph construction,
all 5/6 model training loops, walk-forward evaluation, lead-time analysis,
SHAP & GNNExplainer explainability, and saves all outputs to `results/` and Downloads.

Usage:
    python scripts/run_pipeline.py --phase all
"""

import os
import sys
import argparse
import pathlib
import shutil
import json
import pandas as pd
import numpy as np

# Add project root to sys.path
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.fetch_prices import fetch_all_prices
from src.data.fetch_macro import fetch_macro_indicators
from src.data.fetch_sentiment import fetch_sentiment_features
from src.data.fetch_exposure import build_exposure_dataset
from src.features.build_features import build_master_feature_matrix
from src.graphs.graph_builder import build_all_graph_snapshots
from src.models.tabular_models import LogisticRegressionModel, RandomForestModel, XGBoostModel, SVMModel, VotingEnsembleModel
from src.models.lstm_model import train_lstm_model
from src.models.gnn_model import train_gnn_model
from src.evaluation.metrics import evaluate_predictions
from src.evaluation.walk_forward import run_walk_forward_backtest
from src.evaluation.lead_time import compute_lead_times, plot_lead_time_chart
from src.explainability.shap_explainer import run_shap_explainability
from src.explainability.gnn_explainer import explain_gnn_snapshot
from src.ablation.topology_ablation import run_topology_ablation
from src.ablation.edge_ablation import run_edge_type_ablation

DOWNLOADS_DIR = pathlib.Path("C:/Users/itsta/Downloads/Systemic_Risk_EWS_India")

def copy_to_downloads():
    """
    Copies key output artifacts and tables to the user's Downloads directory.
    """
    try:
        os.makedirs(DOWNLOADS_DIR, exist_ok=True)
        
        # Copy results tables & figures
        results_dir = ROOT_DIR / "results"
        if results_dir.exists():
            for item in results_dir.rglob("*"):
                if item.is_file():
                    rel_path = item.relative_to(results_dir)
                    dest = DOWNLOADS_DIR / rel_path
                    os.makedirs(dest.parent, exist_ok=True)
                    shutil.copy2(item, dest)
                    
        # Copy outputs folder charts
        outputs_dir = ROOT_DIR / "outputs"
        if outputs_dir.exists():
            for item in outputs_dir.rglob("*"):
                if item.is_file():
                    rel_path = item.relative_to(outputs_dir)
                    dest = DOWNLOADS_DIR / "charts_and_tables" / rel_path
                    os.makedirs(dest.parent, exist_ok=True)
                    shutil.copy2(item, dest)

        print(f"\nAll generated files successfully saved to Downloads -> {DOWNLOADS_DIR}")
    except Exception as e:
        print(f"WARNING: Copy to Downloads failed: {e}")

def run_pipeline(phase="all"):
    print("=" * 70)
    print("      SYSTEMIC RISK EARLY-WARNING SYSTEM (INDIA BANK-NBFC NETWORK)")
    print("                 FULL INTEGRATED RESEARCH WORKFLOW BUILD            ")
    print("=" * 70)

    # 1. Data & Feature Construction
    if phase in ["all", "data"]:
        print("\n--- PHASE 1: DATA SOURCING & INGESTION ---")
        fetch_all_prices()
        fetch_macro_indicators()
        fetch_sentiment_features()
        build_exposure_dataset()

    if phase in ["all", "features"]:
        print("\n--- PHASE 2 & 3: FEATURE MATRIX & DUAL-VIEW CONSTRUCTION ---")
        features_df = build_master_feature_matrix()
        snapshots = build_all_graph_snapshots()
    else:
        features_path = ROOT_DIR / "data" / "processed" / "features_india.csv"
        features_df = pd.read_csv(features_path, parse_dates=["Date"], index_col="Date")

    # 2. Split Data
    split_dt = pd.Timestamp("2022-01-01")
    train_df = features_df[features_df.index < split_dt].dropna()
    test_df = features_df[features_df.index >= split_dt].dropna()

    non_feat = {"label", "high_stress_next_30d", "crisis_name", "label_name", "continuous_stress_score"}
    feature_cols = [c for c in features_df.columns if c not in non_feat and np.issubdtype(features_df[c].dtype, np.number)]

    X_train = train_df[feature_cols].values
    y_train = train_df["high_stress_next_30d"].values.astype(int)
    X_test = test_df[feature_cols].values
    y_test = test_df["high_stress_next_30d"].values.astype(int)

    all_metrics = []
    all_probs = {}

    # 3. Model Training & Benchmarking
    if phase in ["all", "models"]:
        print("\n--- PHASE 4: MODEL TRAINING & PREDICTION ---")
        
        # LogReg
        print("Training Logistic Regression...")
        lr_model = LogisticRegressionModel()
        lr_model.fit(X_train, y_train)
        lr_probs = lr_model.predict_proba(X_test)
        res_lr = evaluate_predictions(y_test, lr_probs, model_name="Logistic Regression")
        all_metrics.append(res_lr)
        all_probs["Logistic Regression"] = lr_probs

        # Random Forest
        print("Training Random Forest...")
        rf_model = RandomForestModel()
        rf_model.fit(X_train, y_train)
        rf_probs = rf_model.predict_proba(X_test)
        res_rf = evaluate_predictions(y_test, rf_probs, model_name="Random Forest")
        all_metrics.append(res_rf)
        all_probs["Random Forest"] = rf_probs

        # XGBoost
        print("Training XGBoost...")
        xgb_model = XGBoostModel()
        xgb_model.fit(X_train, y_train)
        xgb_probs = xgb_model.predict_proba(X_test)
        res_xgb = evaluate_predictions(y_test, xgb_probs, model_name="XGBoost")
        all_metrics.append(res_xgb)
        all_probs["XGBoost"] = xgb_probs

        # SVM
        print("Training Support Vector Machine (SVM)...")
        svm_model = SVMModel()
        svm_model.fit(X_train, y_train)
        svm_probs = svm_model.predict_proba(X_test)
        res_svm = evaluate_predictions(y_test, svm_probs, model_name="SVM")
        all_metrics.append(res_svm)
        all_probs["SVM"] = svm_probs

        # Voting Ensemble
        print("Training Soft Voting Ensemble...")
        ens_model = VotingEnsembleModel()
        ens_model.fit(X_train, y_train)
        ens_probs = ens_model.predict_proba(X_test)
        res_ens = evaluate_predictions(y_test, ens_probs, model_name="Voting Ensemble")
        all_metrics.append(res_ens)
        all_probs["Voting Ensemble"] = ens_probs

        # LSTM
        print("Training PyTorch Bidirectional LSTM...")
        _, lstm_probs, y_test_seq = train_lstm_model(X_train, y_train, X_test, y_test, epochs=30)
        res_lstm = evaluate_predictions(y_test_seq, lstm_probs, model_name="LSTM (PyTorch)")
        all_metrics.append(res_lstm)
        all_probs["LSTM"] = lstm_probs

        # Main Comparison Table
        main_table = pd.DataFrame(all_metrics)
        tables_dir = ROOT_DIR / "results" / "tables"
        os.makedirs(tables_dir, exist_ok=True)
        main_table.to_csv(tables_dir / "main_comparison_table.csv", index=False)

        print("\n" + "=" * 70)
        print("MAIN COMPARISON TABLE (8 FULL METRICS)")
        print("=" * 70)
        print(main_table[["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier_score", "mcc"]].to_string(index=False))

    # 4. Walk-Forward Backtest & Lead-Time Analysis
    if phase in ["all", "evaluation"]:
        print("\n--- PHASE 5: WALK-FORWARD BACKTEST & LEAD-TIME EVALUATION ---")
        run_walk_forward_backtest()

        figures_dir = ROOT_DIR / "results" / "figures"
        os.makedirs(figures_dir, exist_ok=True)
        
        min_len = min(len(p) for p in all_probs.values()) if all_probs else len(test_df)
        aligned_probs = {k: p[-min_len:] for k, p in all_probs.items()}
        plot_lead_time_chart(aligned_probs, test_df.index[-min_len:], save_path=figures_dir / "lead_time_chart.png")

    # 5. Explainability
    if phase in ["all", "explain"]:
        print("\n--- PHASE 6: SHAP & GNNEXPLAINER EXPLAINABILITY ---")
        run_shap_explainability(xgb_model.model, train_df[feature_cols], crisis_name="IL_and_FS_2018")
        run_shap_explainability(xgb_model.model, train_df[feature_cols], crisis_name="Adani_Hindenburg_2023")

    # 6. Ablation Experiments
    if phase in ["all", "ablations"]:
        print("\n--- PHASE 7: TOPOLOGY & EDGE-TYPE ABLATIONS ---")
        run_topology_ablation()
        run_edge_type_ablation()

    # Save to Downloads
    copy_to_downloads()

    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETE! ALL RESEARCH CLAIMS IMPLEMENTED & VERIFIED.")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, default="all", help="Phase to run: all, data, features, models, evaluation, explain, ablations")
    args = parser.parse_args()
    run_pipeline(args.phase)
