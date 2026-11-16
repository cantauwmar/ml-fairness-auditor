"""
Fairness metrics: low-level metric utilities and per-group metric computation.

All functions operate on plain numpy arrays / Python scalars so that they can
be called from both the pipeline runner (overall evaluation) and the fairness
auditor (per-group evaluation) without any dependency on sklearn.metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple


import numpy as np


# --------------------------------------------------------------------------- #
# Confusion-matrix primitives                                                  #
# --------------------------------------------------------------------------- #


def confusion_matrix_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_label: Any = 1,
) -> Tuple[int, int, int, int]:
    """
    Return (TP, FP, TN, FN) for a binary classification problem.

    Parameters
    ----------
    y_true : array-like of shape (n,)
    y_pred : array-like of shape (n,)
    positive_label :
        The value treated as the positive class.

    Returns
    -------
    tp, fp, tn, fn : int
    """
    tp = np.sum((y_true == positive_label) & (y_pred == positive_label))
    fp = np.sum((y_true != positive_label) & (y_pred == positive_label))
    tn = np.sum((y_true != positive_label) & (y_pred != positive_label))
    fn = np.sum((y_true == positive_label) & (y_pred != positive_label))
    return (tp, fp, tn, fn)


# --------------------------------------------------------------------------- #
# Derived scalar metrics                                                        #
# --------------------------------------------------------------------------- #


def safe_divide(
    numerator: float, denominator: float, default: float = float("nan")
) -> float:
    """Return numerator / denominator, or *default* when denominator is zero."""
    if denominator != 0:
        return numerator / denominator
    else:
        return default


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of correct predictions."""
    tp, fp, tn, fn = confusion_matrix_counts(y_true, y_pred)
    return safe_divide(tp + tn, tp + tn + fp + fn)


def positive_rate(y_pred: np.ndarray, positive_label: Any = 1) -> float:
    """Fraction of predictions equal to *positive_label*."""
    return safe_divide(np.sum(y_pred == positive_label), len(y_pred))


def true_positive_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_label: Any = 1,
) -> float:
    """TPR = TP / (TP + FN).  Also called recall or sensitivity."""
    tp, fp, tn, fn = confusion_matrix_counts(y_true, y_pred)
    return safe_divide(tp, tp + fn)


def false_positive_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_label: Any = 1,
) -> float:
    """FPR = FP / (FP + TN).  Also called fall-out."""
    tp, fp, tn, fn = confusion_matrix_counts(y_true, y_pred)
    return safe_divide(fp, fp + tn)


def precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_label: Any = 1,
) -> float:
    """Precision = TP / (TP + FP).  Also called positive predictive value."""
    tp, fp, tn, fn = confusion_matrix_counts(y_true, y_pred)
    return safe_divide(tp, tp + fp)


def f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_label: Any = 1,
) -> float:
    """F1 = 2 * precision * recall / (precision + recall)."""
    pre = precision(y_true, y_pred, positive_label)
    rec = true_positive_rate(y_true, y_pred, positive_label)
    return safe_divide(2 * pre * rec, pre + rec)


# --------------------------------------------------------------------------- #
# Per-group metrics                                                             #
# --------------------------------------------------------------------------- #


@dataclass
class GroupMetrics:
    """
    Performance metrics computed for a single group within a protected attribute.

    Attributes
    ----------
    group_value : Any
        The value of the protected attribute that defines this group.
    n_samples : int
        Number of test-set rows belonging to this group.

    The remaining attributes are scalar metric values (``float``).
    ``NaN`` is used when a metric is undefined (e.g. TPR when there are no
    true positives in the group).
    """

    group_value: Any
    n_samples: int

    positive_rate: float

    # Binary-only metrics (NaN for multi-class)
    tpr: float = float("nan")  # true positive rate (recall / sensitivity)
    fpr: float = float("nan")  # false positive rate
    prec: float = float("nan")  # precision (positive predictive value)
    acc: float = float("nan")  # accuracy


def compute_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_value: Any,
    is_binary: bool,
    positive_label: Any,
) -> GroupMetrics:
    """Compute all metrics for a single group."""
    n = len(y_true)

    pos_rate_val = positive_rate(y_pred, positive_label)

    if is_binary:
        tp, fp, tn, fn = confusion_matrix_counts(y_true, y_pred, positive_label)
        tpr_val = safe_divide(tp, tp + fn)
        fpr_val = safe_divide(fp, fp + tn)
        prec_val = safe_divide(tp, tp + fp)
        acc_val = accuracy(y_true, y_pred)
    else:
        tpr_val = fpr_val = float("nan")
        prec_val = float("nan")
        acc_val = float("nan")

    return GroupMetrics(
        group_value=group_value,
        n_samples=n,
        positive_rate=pos_rate_val,
        tpr=tpr_val,
        fpr=fpr_val,
        prec=prec_val,
        acc=acc_val,
    )
