"""
src/models/c2r.py
=================
Class-to-Regression (C2R) Systemic Risk Percentile Scoring Framework (Balmaseda et al. 2023).
Converts continuous systemic stress scores to coarse quantile risk classes,
trains a GNN/ML classifier, then refines class predictions into continuous percentile risk scores.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

class ClassToRegressionScorer:
    """
    Two-stage Class-to-Regression (C2R) model.
    """
    def __init__(self, num_classes=4):
        self.num_classes = num_classes
        self.regressors = {}
        self.quantiles = None

    def fit_classes(self, continuous_y):
        """
        Calculates quantile thresholds for C2R coarse risk classes.
        """
        self.quantiles = np.quantile(continuous_y, np.linspace(0, 1, self.num_classes + 1))
        class_labels = np.digitize(continuous_y, self.quantiles[1:-1])
        return class_labels

    def fit_refinement(self, X_features, class_predictions, continuous_y):
        """
        Trains per-class continuous regression heads to refine risk percentile predictions.
        """
        for c in range(self.num_classes):
            mask = (class_predictions == c)
            if mask.sum() > 5:
                reg = Ridge(alpha=1.0)
                reg.fit(X_features[mask], continuous_y[mask])
                self.regressors[c] = reg
            else:
                self.regressors[c] = None

    def predict_percentile(self, X_features, class_probs):
        """
        Produces continuous 0-1 systemic risk percentile score.
        """
        class_preds = np.argmax(class_probs, axis=1) if class_probs.ndim > 1 else (class_probs >= 0.5).astype(int)
        
        refined_scores = []
        for i in range(len(X_features)):
            c = class_preds[i]
            feat = X_features[i].reshape(1, -1)
            
            if c in self.regressors and self.regressors[c] is not None:
                score = float(self.regressors[c].predict(feat)[0])
            else:
                # Fallback to probability-weighted score
                score = float(class_probs[i]) if class_probs.ndim == 1 else float(class_probs[i, -1])
            
            refined_scores.append(np.clip(score, 0.0, 1.0))

        return np.array(refined_scores)
