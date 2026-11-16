"""
Pipeline 2 – Logistic Regression (One-Hot Encoding)

Based on the adult-income-prediction-with-logistic-regression.ipynb Kaggle notebook.

"""

from __future__ import annotations
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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


class LogisticRegressionPipeline(MLPipeline):

    """Dataset sub-sampling for performance reasons. 
    Include when running main.py with limited memory, exclude when using pytest tests"""

    def load_data(self, filepath: str) -> pd.DataFrame:
        data = super().load_data(filepath)
        return data.sample(50_000, random_state=42).reset_index(drop=True)


    def __init__(self) -> None:
        super().__init__()
        self.scaler_: StandardScaler | None = None
        self.categorical_cols_: List[str] = []
        self.feature_columns_: List[str] = []
        self.numeric_cols_: List[str] = []

    def encode_train(self, data: pd.DataFrame):
        # print ("X_Train encoding")
        X = data[[c for c in _FEATURE_COLS if c in data.columns]].copy()

        self.numeric_cols_ = [c for c in ["age", "hours-per-week"] if c in X.columns]
        self.categorical_cols_ = [c for c in X.columns if c not in self.numeric_cols_]

        if self.categorical_cols_:
            X = pd.get_dummies(
                X, columns=self.categorical_cols_, drop_first=True, dtype=float
            )

        self.scaler_ = StandardScaler()
        if self.numeric_cols_:
            X[self.numeric_cols_] = self.scaler_.fit_transform(X[self.numeric_cols_])

        self.feature_columns_ = X.columns.tolist()

        # print ("X_Train encoded")
        return X

    def encode_test(self, data: pd.DataFrame):
        # print ("X_Test encoding")
        X = data[[c for c in _FEATURE_COLS if c in data.columns]].copy()

        if self.categorical_cols_:
            X = pd.get_dummies(
                X, columns=self.categorical_cols_, drop_first=True, dtype=float
            )

        if self.numeric_cols_:
            X[self.numeric_cols_] = self.scaler_.transform(X[self.numeric_cols_])

        X = X.reindex(columns=self.feature_columns_, fill_value=0.0)

        # print ("X_Test encoded")
        return X

    def get_model(self) -> LogisticRegression:
        return LogisticRegression(max_iter=1000, random_state=42)
