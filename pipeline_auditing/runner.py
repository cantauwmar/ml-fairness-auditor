"""
Pipeline runner: load → split → filter_train → encode → train → evaluate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .base import MLPipeline


# --------------------------------------------------------------------------- #
# Result container                                                              #
# --------------------------------------------------------------------------- #


@dataclass
class PipelineResult:
    """
    Container for every artefact produced during a single pipeline run.

    Attributes
    ----------
    raw_data : pd.DataFrame
        Data as returned by ``pipeline.load_data``.
    X : pd.DataFrame
        Encoded feature matrix covering the post-filter training rows
        (after ``filter_train``) plus all test rows. Reconstructed as
        ``concat([X_train, X_test])``.
    y : pd.Series
        Target vector aligned to ``X``. Reconstructed as
        ``concat([y_train, y_test])``.
    X_train, X_test : pd.DataFrame
        Train and test feature splits (index-preserving).
    y_train, y_test : pd.Series
        Train and test target splits (index-preserving).
    y_pred : np.ndarray
        Model predictions on the **test** split.
    metrics : dict
        Evaluation metrics computed on the test split.
    pipeline : MLPipeline
        The fitted pipeline instance.
    name : str
        Optional display name for logging / reporting.
    """

    raw_data: pd.DataFrame

    X: pd.DataFrame
    y: pd.Series

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    y_pred: np.ndarray

    pipeline: MLPipeline = field(repr=False)
    metrics: Dict[str, Any] = field(default_factory=dict)
    name: str = ""
    n_train_before_filter: int = 0
    n_train_after_filter: int = 0


# --------------------------------------------------------------------------- #
# Runner                                                                        #
# --------------------------------------------------------------------------- #


class PipelineRunner:
    """
    Orchestrates a complete ML pipeline run.

    The runner is responsible for splitting the data, calling the pipeline's
    stages in the correct order, and computing evaluation metrics.  It is
    intentionally kept separate from the pipeline itself so that the same
    runner can evaluate multiple pipelines under identical conditions.

    Crucially, ``filter_train`` is called **only on the training slice**;
    the test set always contains every row that was not selected for training.

    Parameters
    ----------
    test_size : float
        Fraction of (cleaned) data held out for evaluation (default 0.2).
    random_state : int
        Random seed used for the train/test split (default 42).
    stratify : bool
        Whether to stratify the split by the target label (default ``True``).
    """

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
    ) -> None:
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run(
        self,
        pipeline: MLPipeline,
        filepath: str,
        name: Optional[str] = None,
    ) -> PipelineResult:
        """
        Execute *pipeline* end-to-end and return a :class:`PipelineResult`.

        Steps
        -----
        1. ``pipeline.load_data(filepath)``
        2. ``pipeline.compute_target(raw_data)`` → ``y_full`` (for stratification)
        3. Train/test split on ``raw_data`` indices
        4. ``pipeline.filter_train(train_slice)`` → filtered training rows only
        5. ``pipeline.prepare_train(filtered_train)`` → ``(X_train, y_train)``
        6. ``pipeline.prepare_test(test_slice)`` → ``(X_test, y_test)``
        7. ``pipeline.train(X_train, y_train)``
        8. ``pipeline.predict(X_test)`` → ``y_pred``
        9. Compute evaluation metrics on ``(y_test, y_pred)``

        Parameters
        ----------
        pipeline : MLPipeline
            A *fresh* (not yet trained) pipeline instance.
        filepath : str
            Path to the raw ``acs_income.csv`` data file.
        name : str, optional
            Human-readable label stored in the result.

        Returns
        -------
        PipelineResult
        """
        run_name = name or type(pipeline).__name__

        # 1. Load
        raw_data = pipeline.load_data(filepath)

        # 2. Compute target on full data (pure threshold, no fitting).
        #    Used only for stratified splitting; no feature encoders are touched.
        y_full = pipeline.compute_target(raw_data)

        if len(y_full) != len(raw_data):
            raise ValueError(
                "compute_target must return a Series with the same number of "
                f"rows as raw_data (expected {len(raw_data)}, got {len(y_full)})."
            )

        # 3. Split data indices BEFORE encoding to avoid leakage.
        stratify_labels = y_full if self.stratify else None
        train_idx, test_idx = train_test_split(
            raw_data.index,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_labels,
        )

        # 4. Apply training-only filters (e.g. drop_duplicates) to the train
        #    slice only.  The test slice is never filtered.
        train_slice = raw_data.loc[train_idx]
        train_data = pipeline.filter_train(train_slice)
        test_data = raw_data.loc[test_idx]

        # 5. Encode training data: fits all encoders/scalers, returns (X_train, y_train).
        X_train, y_train = pipeline.prepare_train(train_data)

        # 6. Encode test data: applies encoders fitted in step 5, no refitting.
        X_test, y_test = pipeline.prepare_test(test_data)

        # Reconstruct full X and y for the result object.
        X = pd.concat([X_train, X_test])
        y = pd.concat([y_train, y_test])

        # 7. Train
        pipeline.train(X_train, y_train)

        # 8. Predict
        y_pred = pipeline.predict(X_test)

        # 9. Evaluate
        metrics = self._compute_metrics(y_test, y_pred, pipeline, X_test)

        return PipelineResult(
            raw_data=raw_data,
            X=X,
            y=y,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            y_pred=y_pred,
            metrics=metrics,
            pipeline=pipeline,
            name=run_name,
            n_train_before_filter=int(len(train_slice)),
            n_train_after_filter=int(len(train_data)),
        )

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_metrics(
        y_test: pd.Series,
        y_pred: np.ndarray,
        pipeline: MLPipeline,
        X_test: pd.DataFrame,
    ) -> Dict[str, Any]:
        n_classes = len(np.unique(y_test))
        avg = "binary" if n_classes == 2 else "macro"

        metrics: Dict[str, Any] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred, average=avg, zero_division=0)),
            "precision": float(
                precision_score(y_test, y_pred, average=avg, zero_division=0)
            ),
            "recall": float(recall_score(y_test, y_pred, average=avg, zero_division=0)),
            "classification_report": classification_report(
                y_test, y_pred, zero_division=0
            ),
            "n_test": int(len(y_test)),
            "n_classes": int(n_classes),
        }

        # ROC-AUC (binary only, requires predict_proba support)
        if n_classes == 2:
            try:
                proba = pipeline.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
            except (NotImplementedError, AttributeError, ValueError):
                pass  # not all models support probability estimates

        return metrics
