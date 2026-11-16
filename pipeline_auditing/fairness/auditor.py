"""
Fairness auditor: slice-based group fairness analysis.

All fairness metrics are implemented from scratch using only ``numpy`` and
``pandas``.  No external fairness library (e.g. AIF360, Fairlearn) is used.

Supported fairness metrics (binary classification)
--------------------------------------------------
For each protected attribute the auditor splits the data into the
**privileged group** and the **unprivileged group** (as specified by the
caller) and reports:

* **Demographic Parity Difference** (DPD)
    ``P(Ŷ=1 | A=unprivileged) − P(Ŷ=1 | A=privileged)``
    Ideal value: 0.  Negative values indicate the unprivileged group
    receives fewer positive predictions.

* **Equal Opportunity Difference** (EOD)
    ``TPR(unprivileged) − TPR(privileged)``
    Ideal value: 0.  Negative values mean the model misses more true
    positives for the unprivileged group.

* **Predictive Parity Difference** (PPD)
    ``Precision(unprivileged) − Precision(privileged)``
    Ideal value: 0.

* **Accuracy Parity Difference** (APD)
    ``Accuracy(unprivileged) − Accuracy(privileged)``
    Ideal value: 0.

Per-group breakdowns are always available in
``AttributeAuditResult.group_metrics``.

For multi-class targets, binary-only metrics (EOD, PPD, APD)
are NaN.  DPD is always defined.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Union

import numpy as np
import pandas as pd

from pipeline_auditing.fairness.metrics import (
    GroupMetrics,
    compute_group_metrics,
)

if TYPE_CHECKING:
    from pipeline_auditing.fairness.groups import UnprivilegedGroups


# --------------------------------------------------------------------------- #
# Auditor                                                                     #
# --------------------------------------------------------------------------- #


def _as_list(value: Any) -> list:
    """Normalise a single value or collection to a plain list."""
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _group_label(values: list) -> Any:
    """Derive a concise label from a list of group values.

    Single-element lists return the element itself; multi-element lists
    return a sorted tuple.
    """
    if len(values) == 1:
        return values[0]
    return tuple(sorted(values, key=str))


class FairnessAuditor:
    """
    Slice-based group fairness auditor.

    Given a DataFrame that contains original feature columns, ground-truth
    labels and model predictions, this class computes group-fairness metrics
    for a list of protected attributes entirely from scratch - no fairness library used.

    Parameters
    ----------
    positive_label : Any
        The value in the prediction / label columns that represents the
        "positive" outcome.  Default is ``1``.

    Usage
    -----
    ::

        auditor = FairnessAuditor()
        report  = auditor.audit(
            data=auditable_df,
            protected_attributes=["sex", "race"],
            unprivileged_groups={"sex": 1.0, "race": [1.0, 2.0, 3.0, 4.0]},
            pipeline_name="LogisticRegressionPipeline",
        )
        report.print_report()
    """

    def __init__(self, positive_label: Any = 1) -> None:
        self.positive_label = positive_label

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def audit(
        self,
        data: pd.DataFrame,
        protected_attributes: List[str],
        y_true_col: str = "y_true",
        y_pred_col: str = "y_pred",
        pipeline_name: str = "",
        unprivileged_groups: Union[Dict[str, Any], "UnprivilegedGroups"] = None,
    ) -> AuditReport:
        """
        Compute group-fairness metrics for each protected attribute.

        Parameters
        ----------
        data : pd.DataFrame
            The auditable subset (rows where both ``y_true`` and ``y_pred``
            are non-missing).
        protected_attributes : list of str
            Column names to slice by.
        y_true_col / y_pred_col : str
            Names of the label / prediction columns.
        pipeline_name : str
            Display name stored in the returned report.
        unprivileged_groups : dict or UnprivilegedGroups
            Mapping of attribute name to unprivileged value(s).  All rows
            whose attribute value is **not** in the unprivileged set are
            treated as the privileged group.

        Returns
        -------
        AuditReport
        """
        if unprivileged_groups is None:
            raise ValueError("unprivileged_groups must be provided")

        if callable(unprivileged_groups):
            unprivileged_groups = unprivileged_groups()

        df = data.dropna(subset=[y_true_col, y_pred_col]).copy()

        all_labels = np.unique(
            np.concatenate([df[y_true_col].values, df[y_pred_col].values])
        )
        is_binary = len(all_labels) == 2

        attribute_results: Dict[str, AttributeAuditResult] = {}
        for attr in protected_attributes:
            if attr not in df.columns:
                raise KeyError(
                    f"Protected attribute '{attr}' is not a column in data. "
                    f"Available columns: {list(df.columns)}"
                )
            attribute_results[attr] = self._audit_attribute(
                df=df,
                attribute=attr,
                y_true_col=y_true_col,
                y_pred_col=y_pred_col,
                is_binary=is_binary,
                unprivileged_values=unprivileged_groups[attr],
            )

        return AuditReport(
            pipeline_name=pipeline_name,
            n_samples_audited=len(df),
            is_binary_task=is_binary,
            attribute_results=attribute_results,
        )

    # ------------------------------------------------------------------ #
    # Attribute-level computation - TODO!                                #
    # ------------------------------------------------------------------ #

    def _audit_attribute(
        self,
        df: pd.DataFrame,
        attribute: str,
        y_true_col: str,
        y_pred_col: str,
        is_binary: bool,
        unprivileged_values: Any,
    ) -> AttributeAuditResult:
        """Compute per-group metrics and fairness differences for one attribute."""
        unpriv_list = _as_list(unprivileged_values)
        unpriv_mask = df[attribute].isin(unpriv_list)
        priv_mask = ~unpriv_mask

        unpriv_label = _group_label(unpriv_list)
        priv_values = df.loc[priv_mask, attribute].unique().tolist()
        priv_label = _group_label(priv_values)

        # use compute_group_metrics from pipeline_auditing/fairness/metrics to compute the unpriviledged group metrics
        # (unpriv_gm) and the priviledged group metrics (priv_gm)
        unpriv_gm = compute_group_metrics(
            y_true=df.loc[unpriv_mask, y_true_col].values,
            y_pred=df.loc[unpriv_mask, y_pred_col].values,
            group_value=unpriv_label,
            is_binary=is_binary,
            positive_label=self.positive_label,
        )
        priv_gm = compute_group_metrics(
            y_true=df.loc[priv_mask, y_true_col].values,
            y_pred=df.loc[priv_mask, y_pred_col].values,
            group_value=priv_label,
            is_binary=is_binary,
            positive_label=self.positive_label,
        )

        # demographic parity difference dpd
        dpd = unpriv_gm.positive_rate - priv_gm.positive_rate

        if is_binary:
            eod = unpriv_gm.tpr - priv_gm.tpr
            ppd = unpriv_gm.prec - priv_gm.prec
            apd = unpriv_gm.acc - priv_gm.acc
        else:
            eod = ppd = apd = float("nan")

        groups = [unpriv_label, priv_label]
        group_metrics = {unpriv_label: unpriv_gm, priv_label: priv_gm}

        return AttributeAuditResult(
            attribute=attribute,
            groups=groups,
            group_metrics=group_metrics,
            is_binary_task=is_binary,
            privileged_group=priv_label,
            unprivileged_group=unpriv_label,
            demographic_parity_difference=dpd,
            equal_opportunity_difference=eod,
            predictive_parity_difference=ppd,
            accuracy_parity_difference=apd,
        )


# --------------------------------------------------------------------------- #
# Data classes                                                                #
# --------------------------------------------------------------------------- #


@dataclass
class AttributeAuditResult:
    """
    Fairness audit result for a single protected attribute.

    Attributes
    ----------
    attribute : str
        Name of the protected attribute (e.g. ``"sex"``).
    groups : list of Any
        The two group labels: [unprivileged_group, privileged_group].
    group_metrics : dict
        Maps each group label to its :class:`GroupMetrics`.
    is_binary_task : bool
        ``True`` when the prediction target is binary.
    privileged_group : Any
        Label for the privileged group.
    unprivileged_group : Any
        Label for the unprivileged group.
    demographic_parity_difference : float
    equal_opportunity_difference : float
        ``NaN`` for multi-class tasks.
    predictive_parity_difference : float
        ``NaN`` for multi-class tasks.
    accuracy_parity_difference : float
        ``NaN`` for multi-class tasks.
    """

    attribute: str
    groups: List[Any]
    group_metrics: Dict[Any, GroupMetrics]
    is_binary_task: bool

    privileged_group: Any
    unprivileged_group: Any

    demographic_parity_difference: float
    equal_opportunity_difference: float
    predictive_parity_difference: float
    accuracy_parity_difference: float


@dataclass
class AuditReport:
    """
    Full fairness audit report for a single pipeline run.

    Attributes
    ----------
    pipeline_name : str
    n_samples_audited : int
        Number of test-set rows used for auditing.
    is_binary_task : bool
    attribute_results : dict
        Maps protected attribute name → :class:`AttributeAuditResult`.
    """

    pipeline_name: str
    n_samples_audited: int
    is_binary_task: bool
    attribute_results: Dict[str, AttributeAuditResult] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Reporting helpers                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _fmt_group(val: Any) -> Any:
        """Convert whole-number floats (and tuples thereof) to int for display."""
        if isinstance(val, float) and val.is_integer():
            return int(val)
        if isinstance(val, tuple):
            converted = tuple(
                int(v) if isinstance(v, float) and v.is_integer() else v for v in val
            )
            if len(converted) > 5:
                return f"({converted[0]}, ..., {converted[-1]}) [{len(converted)}]"
            return converted
        return val

    def summary_table(self) -> pd.DataFrame:
        """
        Return a DataFrame with one row per protected attribute containing
        the top-level fairness metrics.
        """
        rows = []
        for attr, res in self.attribute_results.items():
            row = {
                "attribute": attr,
                "n_groups": len(res.groups),
                "privileged_group": self._fmt_group(res.privileged_group),
                "unprivileged_group": self._fmt_group(res.unprivileged_group),
                "DPD": round(res.demographic_parity_difference, 4),
                "EOD": round(res.equal_opportunity_difference, 4),
                "PPD": round(res.predictive_parity_difference, 4),
                "APD": round(res.accuracy_parity_difference, 4)
                if not math.isnan(res.accuracy_parity_difference)
                else float("nan"),
            }
            rows.append(row)
        return pd.DataFrame(rows).set_index("attribute")

    def group_table(self, attribute: str) -> pd.DataFrame:
        """
        Return a per-group metrics table for a specific protected attribute.

        Parameters
        ----------
        attribute : str
            Must be a key in ``self.attribute_results``.
        """
        if attribute not in self.attribute_results:
            raise KeyError(
                f"Attribute '{attribute}' not in this report. "
                f"Available: {list(self.attribute_results)}"
            )
        res = self.attribute_results[attribute]
        rows = []
        for gv, gm in res.group_metrics.items():
            rows.append(
                {
                    "group": gv,
                    "n": gm.n_samples,
                    "pos_rate": round(gm.positive_rate, 4),
                    "TPR": round(gm.tpr, 4) if not math.isnan(gm.tpr) else float("nan"),
                    "FPR": round(gm.fpr, 4) if not math.isnan(gm.fpr) else float("nan"),
                    "precision": round(gm.prec, 4)
                    if not math.isnan(gm.prec)
                    else float("nan"),
                    "accuracy": round(gm.acc, 4)
                    if not math.isnan(gm.acc)
                    else float("nan"),
                }
            )
        return pd.DataFrame(rows).set_index("group")

    def print_report(self) -> None:  # pragma: no cover
        """Print a human-readable summary to stdout."""
        print("=" * 70)
        print(f"FAIRNESS AUDIT REPORT  –  {self.pipeline_name}")
        print(f"Samples audited: {self.n_samples_audited}")
        print(f"Task type:       {'binary' if self.is_binary_task else 'multi-class'}")
        print("=" * 70)

        try:
            from tabulate import tabulate

            use_tabulate = True
        except ImportError:
            use_tabulate = False

        for attr, res in self.attribute_results.items():
            print(f"\n── Protected attribute: {attr} ──")
            print(
                f"   Privileged group  : {res.privileged_group}  "
                f"(pos_rate = {res.group_metrics[res.privileged_group].positive_rate:.3f})"
            )
            print(
                f"   Unprivileged group: {res.unprivileged_group}  "
                f"(pos_rate = {res.group_metrics[res.unprivileged_group].positive_rate:.3f})"
            )
            print()

            summary_data = {
                "Demographic Parity Difference (DPD)": res.demographic_parity_difference,
                "Equal Opportunity Difference   (EOD)": res.equal_opportunity_difference,
                "Predictive Parity Difference   (PPD)": res.predictive_parity_difference,
                "Accuracy Parity Difference     (APD)": res.accuracy_parity_difference,
            }
            for metric_name, value in summary_data.items():
                if not math.isnan(value):
                    print(f"   {metric_name}: {value:+.4f}")

            print()
            group_df = self.group_table(attr)
            if use_tabulate:
                print(
                    tabulate(
                        group_df, headers="keys", tablefmt="simple", floatfmt=".4f"
                    )
                )
            else:
                print(group_df.to_string())

        print("=" * 70)
