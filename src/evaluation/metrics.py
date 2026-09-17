"""
src/evaluation/metrics.py
=========================
Full evaluation metric suite for systemic risk early-warning evaluation:
Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Brier Score, MCC, Confusion Matrix.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, brier_score_loss,
    matthews_corrcoef, confusion_matrix
)

def evaluate_predictions(y_true, y_prob, threshold=0.5, model_name="Model"):
    """
    Computes all 8 metrics required by the paper brief.
    """
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = np.nan

    # PR-AUC
    try:
        p_arr, r_arr, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(r_arr, p_arr)
    except Exception:
        pr_auc = np.nan

    # Brier Score
    try:
        brier = brier_score_loss(y_true, y_prob)
    except Exception:
        brier = np.nan

    # MCC
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except Exception:
        mcc = np.nan

    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "model": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if not np.isnan(roc_auc) else None,
        "pr_auc": round(pr_auc, 4) if not np.isnan(pr_auc) else None,
        "brier_score": round(brier, 4) if not np.isnan(brier) else None,
        "mcc": round(mcc, 4) if not np.isnan(mcc) else None,
        "confusion_matrix": cm
    }
