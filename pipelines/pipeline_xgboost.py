"""
Pipeline 3 – XGBoost

Based on the classification-analysis-with-xgboost.ipynb Kaggle notebook.

"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder

from pipeline_auditing.base import MLPipeline

_FEATURE_COLS = [
    "age",
    "workclass",
    "education-num",
    "marital-status",
    "occupation",
    "native-country",
    "relationship",
    "hours-per-week",
    "sex",
    "race",
]


class XGBoostPipeline(MLPipeline):

    """Dataset sub-sampling for performance reasons. 
    Include when running main.py with limited memory, exclude when using pytest tests"""

    def load_data(self, filepath: str) -> pd.DataFrame:
        data = super().load_data(filepath)
        return data.sample(50_000, random_state=42).reset_index(drop=True)

    def __init__(self) -> None:
        super().__init__()
        self.scaler_: StandardScaler | None = None
        self.categorical_cols_: list[str] = []
        self.feature_columns_: list[str] = []
        self.numeric_cols_: list[str] = []
        self.onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    def encode_train(self, data: pd.DataFrame):

        X = data[[c for c in _FEATURE_COLS if c in data.columns]].copy()

        # X = pd.get_dummies(X)
        X = pd.DataFrame(
            self.onehot.fit_transform(X),
            index=X.index,
            columns=self.onehot.get_feature_names_out(X.columns),
        )

        return X

    def encode_test(self, data: pd.DataFrame):

        X = data[[c for c in _FEATURE_COLS if c in data.columns]].copy()

        # X = pd.get_dummies(X)
        X = pd.DataFrame(
            self.onehot.transform(X),
            index=X.index,
            columns=self.onehot.get_feature_names_out(X.columns),
        )

        return X

    def get_model(self) -> XGBClassifier:
        return XGBClassifier(
            n_estimators=500, learning_rate=0.05, eval_metric="logloss", n_jobs=-1
        )
