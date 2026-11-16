"""
Shared pytest fixtures for all test modules.

The fixtures provide a small, deterministic synthetic dataset that mirrors
the structure of the ACS Income dataset, allowing tests to run quickly and
without requiring the actual data file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# --------------------------------------------------------------------------- #
# Constants mirroring the real dataset                                          #
# --------------------------------------------------------------------------- #

N_SAMPLES = 300


# --------------------------------------------------------------------------- #
# Synthetic DataFrame                                                           #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    """
    A 300-row synthetic DataFrame with the same schema as the ACS Income dataset.

    Feature columns use discrete values matching the real ACS codes.
    """
    rng = np.random.default_rng(seed=0)

    df = pd.DataFrame(
        {
            "age": rng.uniform(18, 90, N_SAMPLES).round(0),
            "workclass": rng.choice(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], N_SAMPLES
            ),
            "education-num": rng.integers(1, 22, N_SAMPLES).astype(float),
            "marital-status": rng.choice([1.0, 2.0, 3.0, 4.0, 5.0], N_SAMPLES),
            "occupation": rng.integers(10, 9920, N_SAMPLES).astype(float),
            "native-country": rng.choice(
                [1.0, 5.0, 13.0, 100.0, 200.0, 300.0, 400.0], N_SAMPLES
            ),
            "relationship": rng.integers(1, 18, N_SAMPLES).astype(float),
            "hours-per-week": rng.integers(1, 100, N_SAMPLES).astype(float),
            "sex": rng.choice([1.0, 2.0], N_SAMPLES),
            "race": rng.choice(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], N_SAMPLES
            ),
            "income": rng.choice([">50K", "<=50K"], N_SAMPLES),
        }
    )

    return df.reset_index(drop=True)


@pytest.fixture(scope="session")
def synthetic_csv_path(synthetic_df: pd.DataFrame, tmp_path_factory) -> Path:
    """
    Write *synthetic_df* to a temporary CSV file and return its path.

    This fixture allows pipeline tests to call ``load_data(filepath)`` with
    a real file path without depending on the actual data file.
    """
    tmp_dir = tmp_path_factory.mktemp("data")
    filepath = tmp_dir / "acs_income.csv"
    synthetic_df.to_csv(filepath, index=False)
    return filepath


@pytest.fixture(scope="session")
def binary_matched_df(synthetic_df: pd.DataFrame) -> pd.DataFrame:
    """
    A minimal matched DataFrame with synthetic y_true / y_pred columns
    suitable for testing the FairnessAuditor with a binary task.
    """
    rng = np.random.default_rng(seed=1)
    df = synthetic_df.copy()

    n = len(df)
    df["y_true"] = rng.integers(0, 2, n)
    # Introduce a slight disparity: sex==1.0 predicted positive more often
    base_pred = df["y_true"].copy()
    flip_mask = (df["sex"] == 2.0) & (rng.random(n) < 0.2)
    base_pred[flip_mask] = 1 - base_pred[flip_mask]
    df["y_pred"] = base_pred.values

    df["in_model"] = True
    df["in_test"] = True
    df["in_train"] = False
    return df


@pytest.fixture(scope="session")
def multiclass_matched_df(synthetic_df: pd.DataFrame) -> pd.DataFrame:
    """
    A matched DataFrame with synthetic y_true / y_pred columns for a
    multi-class task.
    """
    rng = np.random.default_rng(seed=2)
    df = synthetic_df.copy()

    grades = ["A", "B", "C", "D", "F"]
    n = len(df)
    df["y_true"] = rng.choice(grades, n)
    df["y_pred"] = rng.choice(grades, n)

    df["in_model"] = True
    df["in_test"] = True
    df["in_train"] = False
    return df
