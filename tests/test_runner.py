"""
Unit tests for PipelineRunner and PipelineResult.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline_auditing.runner import PipelineResult, PipelineRunner
from pipelines import (
    LogisticRegressionPipeline,
    EffectivePipeline,
)

PIPELINE_CLASSES = [
    LogisticRegressionPipeline,
    EffectivePipeline,
]


# --------------------------------------------------------------------------- #
# Runner construction                                                           #
# --------------------------------------------------------------------------- #


def test_runner_default_params():
    runner = PipelineRunner()
    assert runner.test_size == 0.2
    assert runner.random_state == 42
    assert runner.stratify is True


def test_runner_custom_params():
    runner = PipelineRunner(test_size=0.3, random_state=7, stratify=False)
    assert runner.test_size == 0.3
    assert runner.random_state == 7
    assert runner.stratify is False


# --------------------------------------------------------------------------- #
# Successful runs                                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cls", PIPELINE_CLASSES)
def test_runner_returns_pipeline_result(cls, synthetic_csv_path):
    runner = PipelineRunner()
    result = runner.run(cls(), str(synthetic_csv_path))
    assert isinstance(result, PipelineResult)


@pytest.mark.parametrize("cls", PIPELINE_CLASSES)
def test_result_has_correct_split_sizes(cls, synthetic_csv_path):
    """Test split should be ~ test_size fraction of raw data."""
    runner = PipelineRunner(test_size=0.2)
    result = runner.run(cls(), str(synthetic_csv_path))

    n_raw = len(result.raw_data)
    n_test = len(result.X_test)
    actual_fraction = n_test / n_raw
    assert abs(actual_fraction - 0.2) < 0.05, (
        f"Test fraction {actual_fraction:.2f} deviates too much from 0.20"
    )


@pytest.mark.parametrize("cls", PIPELINE_CLASSES)
def test_result_train_test_indices_disjoint(cls, synthetic_csv_path):
    """Training and test indices must be disjoint."""
    runner = PipelineRunner()
    result = runner.run(cls(), str(synthetic_csv_path))

    train_idx = set(result.X_train.index)
    test_idx = set(result.X_test.index)
    assert train_idx.isdisjoint(test_idx), "Training and test sets share row indices."


@pytest.mark.parametrize("cls", PIPELINE_CLASSES)
def test_result_predictions_length(cls, synthetic_csv_path):
    """y_pred must have the same length as X_test."""
    runner = PipelineRunner()
    result = runner.run(cls(), str(synthetic_csv_path))
    assert len(result.y_pred) == len(result.X_test)


@pytest.mark.parametrize("cls", PIPELINE_CLASSES)
def test_result_metrics_contain_accuracy(cls, synthetic_csv_path):
    runner = PipelineRunner()
    result = runner.run(cls(), str(synthetic_csv_path))
    assert "accuracy" in result.metrics
    assert 0.0 <= result.metrics["accuracy"] <= 1.0


@pytest.mark.parametrize("cls", PIPELINE_CLASSES)
def test_result_metrics_contain_f1(cls, synthetic_csv_path):
    runner = PipelineRunner()
    result = runner.run(cls(), str(synthetic_csv_path))
    assert "f1" in result.metrics


@pytest.mark.parametrize("cls", PIPELINE_CLASSES)
def test_result_X_index_subset_of_raw(cls, synthetic_csv_path):
    runner = PipelineRunner()
    result = runner.run(cls(), str(synthetic_csv_path))
    # X covers train (possibly deduplicated) + test; test rows are always all present
    assert set(result.X.index).issubset(set(result.raw_data.index))
    assert set(result.X_test.index).issubset(set(result.raw_data.index))


# --------------------------------------------------------------------------- #
# PipelineResult derived properties                                             #
# --------------------------------------------------------------------------- #


def test_result_name_stored(synthetic_csv_path):
    runner = PipelineRunner()
    result = runner.run(
        LogisticRegressionPipeline(), str(synthetic_csv_path), name="TestPipeline"
    )
    assert result.name == "TestPipeline"


# --------------------------------------------------------------------------- #
# Reproducibility                                                               #
# --------------------------------------------------------------------------- #


def test_runner_is_reproducible(synthetic_csv_path):
    """Two runs with the same seed must produce identical predictions."""
    runner = PipelineRunner(random_state=99)
    r1 = runner.run(LogisticRegressionPipeline(), str(synthetic_csv_path))
    r2 = runner.run(LogisticRegressionPipeline(), str(synthetic_csv_path))
    np.testing.assert_array_equal(r1.y_pred, r2.y_pred)
