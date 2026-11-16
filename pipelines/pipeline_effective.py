"""
Pipeline 5 – Effective Pipeline (Random Forest)

Based on the effective-adult-income-classification-pipeline.ipynb Kaggle notebook.

"""

from __future__ import annotations
from pipeline_auditing.base import MLPipeline
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


FEATURE_COLS = [
    "age",
    "workclass",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "hours-per-week",
    "sex",
    "race",
    "native-country",
]


class EffectivePipeline(MLPipeline):

    """Dataset sub-sampling for performance reasons. 
    Include when running main.py with limited memory, exclude when using pytest tests"""

    def load_data(self, filepath: str) -> pd.DataFrame:
        data = super().load_data(filepath)
        return data.sample(50_000, random_state=42).reset_index(drop=True)



    def __init__(self) -> None:
        super().__init__()
        self.label_encoders_ = {}
        self.native_country_freq_ = None
        self.occupation_columns_ = []
        self.feature_columns_ = []
        self.scaler_: StandardScaler | None = None

    def encode_train(self, data):
        X = data[[c for c in FEATURE_COLS if c in data.columns]].copy()

        if "native-country" in X.columns and self.native_country_freq_ is not None:
            X["native_country_freq"] = (
                X["native-country"].map(self.native_country_freq_).fillna(0)
            )

        if "occupation" in X.columns:
            occupation_dummies = pd.get_dummies(
                X["occupation"], prefix="occupation"
            ).astype(int)
            self.occupation_columns_ = occupation_dummies.columns.tolist()
            X = pd.concat([X, occupation_dummies], axis=1)

        X = X.drop(["occupation", "native-country"], axis=1, errors="ignore")

        education_order = {
            "Preschool": 0,
            "1st-4th": 1,
            "5th-6th": 2,
            "7th-8th": 3,
            "9th": 4,
            "10th": 5,
            "11th": 6,
            "12th": 7,
            "HS-grad": 8,
            "Some-college": 9,
            "Assoc-voc": 10,
            "Assoc-acdm": 11,
            "Bachelors": 12,
            "Masters": 13,
            "Prof-school": 14,
            "Doctorate": 15,
        }

        if "education" in X.columns:
            X["education_ordinal"] = X["education"].map(education_order)
            X = X.drop("education", axis=1)

        self.scaler_ = StandardScaler()

        X = pd.DataFrame(
            self.scaler_.fit_transform(X.astype(float)),
            columns=X.columns,
            index=X.index,
        )

        self.feature_columns_ = X.columns.tolist()
        return X

    def encode_test(self, data: pd.DataFrame):
        X = data[[c for c in FEATURE_COLS if c in data.columns]].copy()

        for col, le in self.label_encoders_.items():
            if col in X.columns:
                X[col] = le.transform(X[col])

        if "native-country" in X.columns and self.native_country_freq_ is not None:
            X["native_country_freq"] = (
                X["native-country"].map(self.native_country_freq_).fillna(0)
            )

        if "occupation" in X.columns and self.occupation_columns_:
            occupation_dummies = (
                pd.get_dummies(X["occupation"], prefix="occupation")
                .reindex(columns=self.occupation_columns_, fill_value=0)
                .astype(int)
            )
            X = pd.concat([X, occupation_dummies], axis=1)

        X = X.drop(["occupation", "native-country"], axis=1, errors="ignore")

        education_order = {
            "Preschool": 0,
            "1st-4th": 1,
            "5th-6th": 2,
            "7th-8th": 3,
            "9th": 4,
            "10th": 5,
            "11th": 6,
            "12th": 7,
            "HS-grad": 8,
            "Some-college": 9,
            "Assoc-voc": 10,
            "Assoc-acdm": 11,
            "Bachelors": 12,
            "Masters": 13,
            "Prof-school": 14,
            "Doctorate": 15,
        }

        if "education" in X.columns:
            X["education_ordinal"] = X["education"].map(education_order)
            X = X.drop("education", axis=1)

        X = pd.DataFrame(
            self.scaler_.transform(X.astype(float)),
            columns=X.columns,
            index=X.index,
        )

        X = X.reindex(columns=self.feature_columns_, fill_value=0)
        return X

    def get_model(self) -> RandomForestClassifier:
        return RandomForestClassifier(random_state=42)
