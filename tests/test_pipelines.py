"""
Unit tests for the concrete pipeline implementations.

Each test verifies:
- The pipeline can be instantiated.
- load_data returns a DataFrame with the expected schema.
- filter_train preserves the index and returns a valid DataFrame.
- prepare_train returns (X, y) with matching indices and correct dtypes.
- prepare_test uses fitted encoders without refitting, producing the same column count as prepare_train.
- get_model returns an unfitted estimator.
- The full load → filter_train → prepare_train → prepare_test → train → predict sequence works end-to-end.

prepare_train/prepare_test are the runner-facing interface; encode_train/encode_test are pure
feature engineering methods called by those concrete helpers in the base class.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipelines import (
    RandomBaselinePipeline,
    ML816Pipeline,
    LogisticRegressionPipeline,
    XGBoostPipeline,
    LogisticPCAPipeline,
    EffectivePipeline,
)
from pipeline_auditing.base import MLPipeline


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

PIPELINE_CLASSES = [
    RandomBaselinePipeline,
    ML816Pipeline,
    LogisticRegressionPipeline,
    XGBoostPipeline,
    LogisticPCAPipeline,
    EffectivePipeline,
]

RAW_COLUMNS = {
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
    "income",
}


# --------------------------------------------------------------------------- #
# Instantiation                                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cls", PIPELINE_CLASSES)
def test_instantiation(cls):
    """Each pipeline can be instantiated and is a subclass of MLPipeline."""
    p = cls()
    assert isinstance(p, MLPipeline)


@pytest.mark.parametrize("cls", PIPELINE_CLASSES)
def test_predict_before_train_raises(cls, synthetic_csv_path):
    """Calling predict before train should raise RuntimeError."""
    p = cls()
    raw = p.load_data(str(synthetic_csv_path))
    filtered = p.filter_train(raw)
    X, _ = p.prepare_train(filtered)
    with pytest.raises(RuntimeError, match="not been trained"):
        p.predict(X)


# --------------------------------------------------------------------------- #
# load_data                                                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cls", PIPELINE_CLASSES)
def test_load_data_returns_dataframe(cls, synthetic_csv_path):
    p = cls()
    df = p.load_data(str(synthetic_csv_path))
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


@pytest.mark.parametrize("cls", PIPELINE_CLASSES)
def test_load_data_has_integer_index(cls, synthetic_csv_path):
    """load_data should reset the index to a RangeIndex."""
    p = cls()
    df = p.load_data(str(synthetic_csv_path))
    assert df.index.is_monotonic_increasing
    assert df.index[0] == 0


@pytest.mark.parametrize("cls", PIPELINE_CLASSES)
def test_load_data_has_expected_columns(cls, synthetic_csv_path):
    """All raw feature and target columns should be present after loading."""
    p = cls()
    df = p.load_data(str(synthetic_csv_path))
    lower_cols = {c.strip().lower() for c in df.columns}
    assert RAW_COLUMNS.issubset(lower_cols), (
        f"Missing columns: {RAW_COLUMNS - lower_cols}"
    )


# --------------------------------------------------------------------------- #
# filter_train                                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cls", PIPELINE_CLASSES)
def test_filter_train_returns_dataframe(cls, synthetic_csv_path):
    p = cls()
    raw = p.load_data(str(synthetic_csv_path))
    filtered = p.filter_train(raw)
    assert isinstance(filtered, pd.DataFrame)
    assert len(filtered) > 0


@pytest.mark.parametrize("cls", PIPELINE_CLASSES)
def test_filter_train_index_is_subset(cls, synthetic_csv_path):
    """The filtered index must be a subset of the raw index."""
    p = cls()
    raw = p.load_data(str(synthetic_csv_path))
    filtered = p.filter_train(raw)
    assert set(filtered.index).issubset(set(raw.index)), (
        "filter_train must preserve the original index of surviving rows."
    )


@pytest.mark.parametrize("cls", PIPELINE_CLASSES)
def test_filter_train_does_not_modify_input(cls, synthetic_csv_path):
    """filter_train must not modify the input DataFrame in-place."""
    p = cls()
    raw = p.load_data(str(synthetic_csv_path))
    raw_copy = raw.copy()
    p.filter_train(raw)
    pd.testing.assert_frame_equal(raw, raw_copy)


@pytest.mark.parametrize("cls", PIPELINE_CLASSES)
def test_compute_target_returns_binary_series(cls, synthetic_csv_path):
    """compute_target must return a binary integer Series with the same index."""
    p = cls()
    raw = p.load_data(str(synthetic_csv_path))
    filtered = p.filter_train(raw)
    y = p.compute_target(filtered)

    assert isinstance(y, pd.Series)
    assert len(y) == len(filtered)
    pd.testing.assert_index_equal(y.index, filtered.index)
    assert y.isin([0, 1]).all(), "compute_target must return binary (0/1) labels"


# --------------------------------------------------------------------------- #
# encode_train                                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cls", PIPELINE_CLASSES)
def test_encode_train_returns_correct_shapes(cls, synthetic_csv_path):
    p = cls()
    raw = p.load_data(str(synthetic_csv_path))
    filtered = p.filter_train(raw)
    X, y = p.prepare_train(filtered)

    assert isinstance(X, pd.DataFrame), "prepare_train must return a DataFrame as X"
    assert isinstance(y, pd.Series), "prepare_train must return a Series as y"
    assert len(X) == len(filtered)
    assert len(y) == len(filtered)


@pytest.mark.parametrize("cls", PIPELINE_CLASSES)
def test_encode_train_preserves_index(cls, synthetic_csv_path):
    """X and y must share the same index as the filtered DataFrame."""
    p = cls()
    raw = p.load_data(str(synthetic_csv_path))
    filtered = p.filter_train(raw)
    X, y = p.prepare_train(filtered)

    pd.testing.assert_index_equal(X.index, filtered.index)
    pd.testing.assert_index_equal(y.index, filtered.index)


@pytest.mark.parametrize("cls", PIPELINE_CLASSES)
def test_encode_train_X_is_numeric(cls, synthetic_csv_path):
    """All columns of X must be numeric after encoding."""
    p = cls()
    raw = p.load_data(str(synthetic_csv_path))
    filtered = p.filter_train(raw)
    X, _ = p.prepare_train(filtered)

    non_numeric = X.select_dtypes(exclude="number").columns.tolist()
    assert non_numeric == [], f"Non-numeric columns in X after encoding: {non_numeric}"


@pytest.mark.parametrize("cls", PIPELINE_CLASSES)
def test_encode_train_no_score_leakage(cls, synthetic_csv_path):
    """The income column must not appear in X (target leakage prevention)."""
    p = cls()
    raw = p.load_data(str(synthetic_csv_path))
    filtered = p.filter_train(raw)
    X, _ = p.prepare_train(filtered)

    assert "income" not in X.columns, "income column found in X (target leakage!)"


# --------------------------------------------------------------------------- #
# get_model                                                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cls", PIPELINE_CLASSES)
def test_get_model_returns_unfitted_estimator(cls):
    from sklearn.base import BaseEstimator

    p = cls()
    model = p.get_model()
    assert isinstance(model, BaseEstimator)
    # Fitted sklearn estimators have instance attributes ending with '_'.
    # Use vars() (not dir()) to check only instance attributes and avoid
    # false positives from class-level properties like _repr_html_.
    fitted_attrs = [
        a for a in vars(model) if a.endswith("_") and not a.startswith("__")
    ]
    assert fitted_attrs == [], (
        f"get_model() returned an already-fitted estimator "
        f"(fitted attributes: {fitted_attrs})"
    )


# --------------------------------------------------------------------------- #
# End-to-end pipeline run                                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cls", PIPELINE_CLASSES)
def test_pipeline_end_to_end(cls, synthetic_csv_path):
    """Full load → filter_train → prepare_train → prepare_test → train → predict should succeed."""
    from sklearn.model_selection import train_test_split

    p = cls()
    raw = p.load_data(str(synthetic_csv_path))
    filtered = p.filter_train(raw)

    train_data, test_data = train_test_split(filtered, test_size=0.2, random_state=42)
    X_train, y_train = p.prepare_train(train_data)
    X_test, _ = p.prepare_test(test_data)

    p.train(X_train, y_train)
    preds = p.predict(X_test)

    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(X_test)


@pytest.mark.parametrize("cls", PIPELINE_CLASSES)
def test_encode_test_column_count_matches_train(cls, synthetic_csv_path):
    """prepare_test must produce the same number of columns as prepare_train."""
    from sklearn.model_selection import train_test_split

    p = cls()
    raw = p.load_data(str(synthetic_csv_path))
    filtered = p.filter_train(raw)

    train_data, test_data = train_test_split(filtered, test_size=0.2, random_state=42)
    X_train, _ = p.prepare_train(train_data)
    X_test, _ = p.prepare_test(test_data)

    assert X_test.shape[1] == X_train.shape[1], (
        f"prepare_test produced {X_test.shape[1]} columns but prepare_train produced {X_train.shape[1]}"
    )


SCALED_PIPELINE_CLASSES = [
    ML816Pipeline,
    LogisticRegressionPipeline,
    LogisticPCAPipeline,
]


@pytest.mark.parametrize("cls", SCALED_PIPELINE_CLASSES)
def test_encode_test_does_not_refit_scaler(cls, synthetic_csv_path):
    """prepare_test must not refit the scaler (scaler mean must be unchanged)."""
    from sklearn.model_selection import train_test_split

    p = cls()
    raw = p.load_data(str(synthetic_csv_path))
    filtered = p.filter_train(raw)

    train_data, test_data = train_test_split(filtered, test_size=0.2, random_state=42)
    p.prepare_train(train_data)

    mean_before = p.scaler_.mean_.copy()
    p.prepare_test(test_data)
    mean_after = p.scaler_.mean_

    np.testing.assert_array_equal(
        mean_before,
        mean_after,
        err_msg="encode_test refitted the scaler (mean changed after encode_test call)",
    )
