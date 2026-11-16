"""
main.py – Full fairness auditing workflow demonstration.

Run this script after ensuring the dataset is present:

    python main.py

The script:
  1. Runs all pipelines on the dataset.
  2. Prints a performance summary for each pipeline.
  3. Matches predictions back to the raw data.
  4. Audits each pipeline for group fairness across protected attributes.
  5. Prints a fairness comparison table across all pipelines.

!  IN CASE OF CRASH WHEN RUNNING MAIN.PY INCLUDE DATASET SUBSAMPLING 
IN EFFEECTIVE, XGBOOST AND LOGREG  !
  
"""





from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from pipeline_auditing.runner import PipelineRunner
from pipelines import (
    RandomBaselinePipeline,
    ML816Pipeline,
    LogisticRegressionPipeline,
    XGBoostPipeline,
    LogisticPCAPipeline,
    EffectivePipeline,
)
from pipeline_auditing.fairness.matcher import PredictionMatcher
from pipeline_auditing.fairness.auditor import FairnessAuditor, AuditReport
from pipeline_auditing.fairness.groups import ACSIncomeGroups

# Instantiate the reference unprivileged-groups definition.
UNPRIVILEGED_GROUPS = ACSIncomeGroups()

# Protected attributes to audit
PROTECTED_ATTRIBUTES = list(UNPRIVILEGED_GROUPS.get_groups().keys())

PIPELINE_CONFIGS = [
    ("Pipeline 0 – Random Baseline", RandomBaselinePipeline),
    ("Pipeline 1 – ML816 (Logistic+Threshold)", ML816Pipeline),
    ("Pipeline 2 – Logistic Regression (OHE)", LogisticRegressionPipeline),
    ("Pipeline 3 – XGBoost", XGBoostPipeline),
    ("Pipeline 4 – Logistic + PCA", LogisticPCAPipeline),
    ("Pipeline 5 – Effective (Random Forest)", EffectivePipeline),
]


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #


def _print_table(df: pd.DataFrame) -> None:
    try:
        from tabulate import tabulate

        print(tabulate(df, headers="keys", tablefmt="simple", floatfmt="+.4f"))
    except ImportError:
        print(df.to_string())


def print_section(title: str) -> None:
    width = 72
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_performance(result) -> None:
    m = result.metrics
    print(f"  Accuracy  : {m['accuracy']:.4f}")
    print(f"  F1        : {m['f1']:.4f}")
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    if "roc_auc" in m:
        print(f"  ROC-AUC   : {m['roc_auc']:.4f}")
    print(f"  Test rows : {m['n_test']}")


def build_comparison_table(
    reports: dict[str, AuditReport],
    attribute: str,
    results: dict[str, object],
) -> pd.DataFrame:
    """Build a fairness comparison table across all pipelines for one attribute."""
    rows = []
    for pipeline_name, report in reports.items():
        if attribute not in report.attribute_results:
            continue
        res = report.attribute_results[attribute]
        rows.append(
            {
                "Pipeline": pipeline_name,
                "DPD": round(res.demographic_parity_difference, 4),
                "EOD": round(res.equal_opportunity_difference, 4),
                "PPD": round(res.predictive_parity_difference, 4),
                "APD": round(res.accuracy_parity_difference, 4),
                "Accuracy - (DPD+EOD)" : (results[pipeline_name].metrics['accuracy'])-
                (
                    abs(round(res.demographic_parity_difference, 4))+
                    abs(round(res.equal_opportunity_difference, 4))
                ),
                "F1 - (DPD+EOD)" : (results[pipeline_name].metrics['f1'])-
                (
                    abs(round(res.demographic_parity_difference, 4))+
                    abs(round(res.equal_opportunity_difference, 4))
                ),
                "Unprivileged": AuditReport._fmt_group(res.unprivileged_group),
                "Privileged": AuditReport._fmt_group(res.privileged_group)
            }
        )
    df = pd.DataFrame(rows).set_index("Pipeline")
    return df


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #

DATA_PATH = "data/acs_income.csv.gz"


def main() -> None:
    runner = PipelineRunner(test_size=0.2, random_state=42)
    matcher = PredictionMatcher()
    auditor = FairnessAuditor(positive_label=1)

    reports: dict[str, AuditReport] = {}

    results: dict[str, object] = {}

    for display_name, pipeline_cls in PIPELINE_CONFIGS:
        print_section(display_name)

        try:
            # ── run pipeline ─────────────────────────────────────────────
            pipeline = pipeline_cls()
            result = runner.run(pipeline, DATA_PATH, name=display_name)

            results[display_name] = result
            print("\n  ── Performance ─────────────────────────────────────")
            print_performance(result)

            # ── match + audit ─────────────────────────────────────────────
            matched_df = matcher.match(result)
            auditable_df = PredictionMatcher.get_auditable_subset(matched_df)

            # Filter to attributes that exist in the auditable data
            available_attrs = [
                a for a in PROTECTED_ATTRIBUTES if a in auditable_df.columns
            ]

            report = auditor.audit(
                data=auditable_df,
                protected_attributes=available_attrs,
                pipeline_name=display_name,
                unprivileged_groups=UNPRIVILEGED_GROUPS,
            )
            reports[display_name] = report

            print("\n  ── Fairness (summary) ──────────────────────────────")
            _print_table(report.summary_table())

        except (NotImplementedError, TypeError):
            print(f"  Pipeline {display_name!r} not implemented yet.")
            continue

    # ── Cross-pipeline comparison ────────────────────────────────────────
    print_section("Cross-Pipeline Fairness Comparison")
    for attr in PROTECTED_ATTRIBUTES:
        print(f"\n  Protected attribute: {attr}")
        cmp_table = build_comparison_table(reports, attr, results)
        _print_table(cmp_table)

    print()
    print("Fairness metric legend:")
    print("  DPD   Demographic Parity Difference     (ideal: 0)")
    print("  EOD   Equal Opportunity Difference       (ideal: 0)")
    print("  PPD   Predictive Parity Difference       (ideal: 0)")
    print("  APD   Accuracy Parity Difference         (ideal: 0)")
    print()
    print("Negative DPD / EOD / PPD / APD = unprivileged group is disadvantaged.")

    # Pipeline Fairness-Accuracy Difference


# --------------------------------------------------------------------------- #
# Entry point                                                                   #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    if not Path(DATA_PATH).exists():
        print(
            f"ERROR: data file not found at '{DATA_PATH}'.\n"
            "The dataset should already be present at data/acs_income.csv.gz.\n"
            "If missing, run: python data/download_acs_income.py",
            file=sys.stderr,
        )
        sys.exit(1)

    main()
