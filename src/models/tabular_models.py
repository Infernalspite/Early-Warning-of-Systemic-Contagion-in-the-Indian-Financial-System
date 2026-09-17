"""
src/models/tabular_models.py
============================
Implements Logistic Regression, Random Forest, XGBoost, SVM, and Voting Ensemble models.
Supports class-imbalance weighting, probability estimation, and feature importance extraction.
"""

import os
import pathlib
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent

class BaseTabularModel:
    def __init__(self, name="TabularModel"):
        self.name = name
        self.model = None

    def fit(self, X_train, y_train):
        raise NotImplementedError

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, filepath):
        with open(filepath, "rb") as f:
            self.model = pickle.load(f)

class LogisticRegressionModel(BaseTabularModel):
    def __init__(self, C=0.1, max_iter=1000):
        super().__init__("LogisticRegression")
        self.model = LogisticRegression(C=C, class_weight="balanced", max_iter=max_iter, random_state=42)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

class RandomForestModel(BaseTabularModel):
    def __init__(self, n_estimators=500, max_depth=10):
        super().__init__("RandomForest")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def get_feature_importances(self, feature_names):
        return pd.Series(self.model.feature_importances_, index=feature_names).sort_values(ascending=False)

class XGBoostModel(BaseTabularModel):
    def __init__(self, n_estimators=500, max_depth=6, learning_rate=0.05):
        super().__init__("XGBoost")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

    def fit(self, X_train, y_train):
        n_pos = int(np.sum(y_train == 1))
        n_neg = int(np.sum(y_train == 0))
        scale_pos_weight = n_neg / max(n_pos, 1)

        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            verbosity=0
        )
        self.model.fit(X_train, y_train)

    def get_feature_importances(self, feature_names):
        return pd.Series(self.model.feature_importances_, index=feature_names).sort_values(ascending=False)

class SVMModel(BaseTabularModel):
    def __init__(self, C=1.0, kernel="rbf"):
        super().__init__("SVM")
        self.model = SVC(C=C, kernel=kernel, probability=True, class_weight="balanced", random_state=42)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

class VotingEnsembleModel(BaseTabularModel):
    def __init__(self):
        super().__init__("VotingEnsemble")
        lr = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=42)
        rf = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=42)
        xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0)

        self.model = VotingClassifier(
            estimators=[("lr", lr), ("rf", rf), ("xgb", xgb)],
            voting="soft"
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
