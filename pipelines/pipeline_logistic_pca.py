"""
Pipeline 4 – Logistic Regression + PCA

Based on the eda-logistic-regression-pca.ipynb Kaggle notebook.

"""

from __future__ import annotations
from typing import List

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from pipeline_auditing.base import MLPipeline


_FEATURE_COLS = [
    "age",
    "workclass",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "sex",
    "race",
    "native-country",
    "hours-per-week",
]


class LogisticPCAPipeline(MLPipeline):
    def __init__(self) -> None:
        super().__init__()
        self.scaler_: StandardScaler = StandardScaler()
        self.label_encoders_: dict[str, preprocessing.LabelEncoder] = {}
        self.pca = PCA(n_components=0.90)

    def encode_train(self, data: pd.DataFrame):
        cols = [c for c in _FEATURE_COLS if c in data.columns]
        X = data[cols].copy()

        categorical = [
            "workclass",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]

        for feature in categorical:
            le = preprocessing.LabelEncoder()
            X[feature] = le.fit_transform(X[feature])
            self.label_encoders_[feature] = le

        self.scaler_ = StandardScaler()

        X = pd.DataFrame(
            self.scaler_.fit_transform(X.astype(float)),
            columns=cols,
            index=X.index,
        )

        X = pd.DataFrame(
            self.pca.fit_transform(X),
            index=X.index
            )
        return X

    def encode_test(self, data: pd.DataFrame):
        cols = [c for c in _FEATURE_COLS if c in data.columns]
        X = data[cols].copy()

        categorical = [
            "workclass",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]
        # Hier musste ich die Encodeten klassen manuell mappen, da LabelEncoder kein handling für werte hat, 
        # die währen dem Fitting nicht vorgekommen sind.
        for feature, le in self.label_encoders_.items():
            if feature in X.columns:
                mapping = {cls: idx for idx, cls in enumerate(le.classes_)}
                X[feature] = X[feature].astype(int).map(mapping).fillna(-1).astype(int)

        X = pd.DataFrame(
            self.scaler_.transform(X.astype(float)),
            columns=cols,
            index=X.index,
        )

        X = pd.DataFrame(
            self.pca.transform(X), 
            index=X.index
        )

        return X

    def get_model(self) -> LogisticRegression:
        return LogisticRegression(random_state=42)
