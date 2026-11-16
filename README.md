# Fairness Auditing Practical


---

## Overview

In this practical you will build a **fairness auditing system** for machine
learning (ML) pipelines, applied to the
[ACS Income dataset](https://www.census.gov/programs-surveys/acs) derived from
the US Census American Community Survey.

You will:

1. Understand an abstract ML pipeline base class and adapt four real Kaggle
   notebooks to use it.
2. Build a pipeline runner that evaluates predictive performance.
3. Implement a **prediction matcher** that maps model predictions back to
   the original (possibly filtered) source data.
4. Implement a **fairness auditor** that slices predictions by protected
   attributes and computes group-fairness metrics *from scratch* — no
   fairness library allowed. This includes implementing all metric
   functions, defining which groups are considered unprivileged and why,
   and wiring everything into the end-to-end demo.

---

## Repository Structure

```
.
├── data/
│   ├── acs_income.csv.gz            ← dataset (already present, gzip-compressed)
│   ├── download_acs_income.py       ← script to regenerate the dataset
│   └── example-notebooks/           ← Kaggle notebooks to refactor (already present)
│       ├── adult-income-prediction-ml-81-6.ipynb
│       ├── adult-income-prediction-with-logistic-regression.ipynb
│       ├── classification-analysis-with-xgboost.ipynb
│       ├── eda-logistic-regression-pca.ipynb
│       └── effective-adult-income-classification-pipeline.ipynb
├── pipeline_auditing/               ← framework code
│   ├── base.py                      ← abstract MLPipeline class         [Step 1]
│   ├── runner.py                    ← PipelineRunner + PipelineResult   [Step 2]
│   └── fairness/
│       ├── matcher.py               ← PredictionMatcher                 [Step 3]
│       ├── metrics.py               ← low-level metric helpers           [Step 4a]
│       ├── groups.py                ← UnprivilegedGroups base class      [Step 4b]
│       └── auditor.py               ← FairnessAuditor + AuditReport     [Step 4c]
├── pipelines/                       ← concrete pipeline implementations
│   ├── pipeline_baseline.py         ← Pipeline 0: Random Baseline       [Step 1]
│   ├── pipeline_ml816.py            ← Pipeline 1: ML816 (Logistic+Threshold) [Step 1]
│   ├── pipeline_logistic_regression.py ← Pipeline 2: Logistic Regression (OHE) [Step 1]
│   ├── pipeline_xgboost.py          ← Pipeline 3: XGBoost               [Step 1]
│   ├── pipeline_logistic_pca.py     ← Pipeline 4: Logistic + PCA        [Step 1]
│   └── pipeline_effective.py        ← Pipeline 5: Effective (Random Forest) [Step 1]
├── tests/
│   ├── conftest.py                  ← shared fixtures (synthetic data)
│   ├── test_pipelines.py
│   ├── test_runner.py
│   ├── test_matcher.py
│   ├── test_auditor.py
│   └── test_groups.py
├── main.py                          ← end-to-end demo script
├── lint.sh                          ← code quality checks
└── requirements.txt
```

---

## Setup

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Verify your setup

The dataset (`data/acs_income.csv.gz`) is already present in the repository —
no download required.

If you need to regenerate the dataset, run:

```bash
python data/download_acs_income.py
```

```bash
# Run all tests
python -m pytest tests/ -v

# Run the end-to-end demo
python main.py

# Check code quality
bash lint.sh
```

**Performance note**: On the full 1.6 million row ACS dataset, Random Forest
and tree-based models can be slow to train.  During development, consider
sampling a subset:

```python
df = pd.read_csv("data/acs_income.csv.gz").sample(50_000, random_state=42)
```

---

## Dataset

The dataset contains income records from the US Census ACS, with a binary
target indicating whether income exceeds $50,000 per year.

| Column | Type | Description | Original Column Name |
|---|---|---|---|
| `age` | float | Age in years | AGEP |
| `workclass` | float | ACS employment class code | COW |
| `education-num` | float | Years of education (numeric) | SCHL |
| `marital-status` | float | ACS marital status code | MAR |
| `occupation` | float | ACS occupation code | OCCP |
| `native-country` | float | ACS country-of-birth code | POBP |
| `relationship` | float | ACS relationship to householder code | RELP |
| `hours-per-week` | float | Usual hours worked per week | WKHP |
| `sex` | float | 1.0 = Male, 2.0 = Female | SEX |
| `race` | float | 1.0 = White, 2.0 = Black/African American, 3.0 = American Indian, 4.0 = Alaska Native, 5.0 = American Indian + Alaska Native, 6.0 = Asian, 7.0 = Native Hawaiian/Pacific Islander, 8.0 = Some Other Race, 9.0 = Two or More Races | RAC1P |
| `income` | str | `">50K"` or `"<=50K"` (prediction target) | PINCP |

Note: This dataset is similar but not identical to the dataset used in the Kaggle notebooks, the exploratory data 
analyses from the Kaggle notebooks was not performed on this dataset.

---

## Steps

### Step 1 – Abstract Pipeline & Concrete Implementations

**Files**: [`pipeline_auditing/base.py`](pipeline_auditing/base.py) and the files in [`pipelines/`](pipelines/)

The `MLPipeline` abstract base class defines the pipeline stages.  The
abstract methods every concrete pipeline **must** implement are:

| Method | Responsibility |
|---|---|
| `encode_train(data)` | **Fit** encoders/scalers on training features and return a numeric `DataFrame` (same index as input) |
| `encode_test(data)` | **Transform** test features using encoders fitted in `encode_train` — must **not** call `fit` |
| `get_model()` | Return a fresh, untrained sklearn estimator |

All other steps of the pipeline (e.g. loading the data, training, predicting, etc.) are already implemented in the base 
class and do not need to be overridden, you will use them later for the runner (step 2). Exception: it is acceptable to 
override `predict` only for explicit post-processing of model outputs (e.g. applying a custom probability threshold as in [`ML816Pipeline`](pipelines/pipeline_ml816.py)).

The pipelines wired into `main.py` are:

| ID | Class | Algorithm | Source Notebook |
|---|---|---|---|
| Pipeline 0 | `RandomBaselinePipeline` | Random predictions | — (baseline) |
| Pipeline 1 | `ML816Pipeline` | Logistic Regression + threshold=0.6 | `adult-income-prediction-ml-81-6.ipynb` |
| Pipeline 2 | `LogisticRegressionPipeline` | Logistic Regression (OHE) | `adult-income-prediction-with-logistic-regression.ipynb` |
| Pipeline 3 | `XGBoostPipeline` | XGBoost | `classification-analysis-with-xgboost.ipynb` |
| Pipeline 4 | `LogisticPCAPipeline` | Logistic Regression + PCA | `eda-logistic-regression-pca.ipynb` |
| Pipeline 5 | `EffectivePipeline` | Random Forest | `effective-adult-income-classification-pipeline.ipynb` |

**Your tasks**:

1. Read [`pipeline_auditing/base.py`](pipeline_auditing/base.py) and ensure you understand the interface.
2. Look at the example pipeline [`pipelines/pipeline_baseline.py`](pipelines/pipeline_baseline.py) that is a minimal pipeline that satisfies the interface.
3. Study the five Kaggle notebooks provided in [`data/example-notebooks/`](data/example-notebooks/).
4. Inspect the example notebook [`data/example-notebooks/adult-income-prediction-ml-81-6.ipynb`](data/example-notebooks/adult-income-prediction-ml-81-6.ipynb) and the corresponding 
   refactored pipeline [`pipelines/pipeline_ml816.py`](pipelines/pipeline_ml816.py) and understand how they relate to each other.
5. Refactor the remaining four notebooks into the corresponding pipeline skeletons by
   implementing the abstract methods (`encode_train`, `encode_test`, `get_model`).
   Note: In case that several options were tried and compared in the original notebook, always pick the option
   with the highest accuracy for refactoring.
6. Run `pytest tests/test_pipelines.py -v` — all tests must pass.

---

### Step 2 – Pipeline Runner

**File**: [`pipeline_auditing/runner.py`](pipeline_auditing/runner.py)

Implement `PipelineRunner.run(pipeline, filepath)` which:

1. Calls `pipeline.load_data`.
2. Calls `pipeline.compute_target` to computes target on full data for stratification.
3. Performs a stratified 80/20 train/test split (use `sklearn.model_selection.train_test_split`).
4. Calls `pipeline.filter_train` **on the training slice only**.
5. Calls `pipeline.prepare_train` on the filtered training slice (fits encoders, returns `X_train, y_train`).
6. Calls `pipeline.prepare_test` on the test slice (transforms using fitted encoders, returns `X_test, y_test`).
7. Calls `pipeline.train`.
8. Calls `pipeline.predict` on the test split.
9. Computes evaluation metrics (accuracy, F1, precision, recall, ROC-AUC).
10. Returns a `PipelineResult` dataclass containing all intermediate artefacts.

This unified framework allows you to directly compare the performance of the five different pipelines under identical settings.

**Questions to answer** (in the final written report):

* Which pipeline achieves the highest accuracy?  Which achieves the best F1?
* Scenario A: Imagine you would be a data scientist working loyally for a bank, and you would use this algorithm to predict the income of a group
 of potential customers. Based on your results, the bank would reach out and offer those your algorithm deems as high-income an exclusive offer, so
 it is important that you are as certain as possible that those deemed high-income by your algorithm are actually high earners. Which of the five
 pipelines would you choose for this purpose and why?
* Scenario B: Now imagine you would be a data scientist working for the city administration and there would be a new legislation concerning high-income individuals.
 Given this algorithm, you are supposed to predict the income of the city residents, such that the city administration can send out letters to all potentially
 high-income individuals. You should be as certain as possible that your algorithm picks up on all high-income individuals and does not miss out a high-income earner,
 if a letter would be sent accidentally to a low-income earner, it would not be a big problem. Which of the five pipelines would you choose in this case and why?

Run `pytest tests/test_runner.py -v` to validate your implementation.

---

### Step 3 – Prediction Matcher

**File**: [`pipeline_auditing/fairness/matcher.py`](pipeline_auditing/fairness/matcher.py)

After a pipeline run, we only have the bare predictions on the **test split** of the
dataset. To perform fairness analysis we need to relate predictions back to the original demographic data for all rows.

Implement `PredictionMatcher.match(result)` which:

1. Takes a `PipelineResult`.
2. Returns a copy of `raw_data` augmented with five new columns:

| Column | Meaning |
|---|---|
| `y_true` | ground-truth label (`NaN` for non-filtered rows) |
| `y_pred` | model prediction (`NaN` for filtered / train rows) |
| `in_model` | `True` if the row was not filtered out |
| `in_test` | `True` if the row is in the test split |
| `in_train` | `True` if the row is in the training split |

Also implement the static helper `get_auditable_subset(matched_df)` which
returns only the rows where both `y_true` and `y_pred` are available (i.e. the test split).

Run `pytest tests/test_matcher.py -v`.

---

### Step 4 – Fairness Auditor

**Files**: [`pipeline_auditing/fairness/metrics.py`](pipeline_auditing/fairness/metrics.py), [`pipeline_auditing/fairness/groups.py`](pipeline_auditing/fairness/groups.py), [`pipeline_auditing/fairness/auditor.py`](pipeline_auditing/fairness/auditor.py)

Implement the full fairness auditing stack from scratch.  You may **not** use
any external fairness library (AIF360, Fairlearn, etc.).  All metrics must be
implemented using only `numpy`.

#### Step 4a – Metric utilities ([`pipeline_auditing/fairness/metrics.py`](pipeline_auditing/fairness/metrics.py))

Implement the following functions:

```python
def confusion_matrix_counts(y_true, y_pred, positive_label=1) -> (TP, FP, TN, FN)
def accuracy(y_true, y_pred) -> float
def positive_rate(y_pred, positive_label=1) -> float
def true_positive_rate(y_true, y_pred, positive_label=1) -> float   # TPR / recall
def false_positive_rate(y_true, y_pred, positive_label=1) -> float  # FPR
def precision(y_true, y_pred, positive_label=1) -> float            # PPV
def f1(y_true, y_pred, positive_label=1) -> float
def safe_divide(numerator, denominator, default=NaN) -> float
```

Return `NaN` (via `float('nan')`) whenever a metric is undefined
(e.g. TPR when there are no true positives in the group).

Run `pytest tests/test_auditor.py -v` (metric helpers are exercised there).

#### Step 4b – Unprivileged group definitions ([`pipeline_auditing/fairness/groups.py`](pipeline_auditing/fairness/groups.py))

Fairness metrics compare an **unprivileged group** against a **privileged
group** for each protected attribute. You should decide as a group which values for each
attribute should be in the unpriviledged group based on world knowledge 
(e.g. socio-economic reasoning, knowledge about historical patterns of discrimination, etc.)
and based on the data distribution in the dataset you operate on. 
All other values not included in the unpriviledged group automatically fall into the priviledged group.
You can find the meaning of each attribute value in this 
[`data dictionary`](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2018.pdf),
note that the column names were changed as in [`data/download_acs_income.py`](data/download_acs_income.py).

The file already contains the `UnprivilegedGroups` abstract base class, which you are suppose to implement in the
ACSIncomeGroups class.

Return format of `get_groups()`:

* **Single value** — if only one value is unpriviledged: `{"sex": 1.0}`
* **List of values** — auditor **combines** them into a single unprivileged group
  and computes metrics over the union of rows:
  `{"race": [1.0, 2.0, 3.0, 4.0]}`

**Your tasks**:

1. Read [`pipeline_auditing/fairness/groups.py`](pipeline_auditing/fairness/groups.py) and understand the interface.
2. Decide as a group which attributes you deem as protected attributes, **justify this choice
   in the written report**. You may want to look up anti-discrimination laws and regulations regarding protected attributes.
3. Examine the dataset: look at the unique values and group sizes for each of
   the protected attributes.
4. Decide which values you deem as unpriviledged for each of the protected attributes and 
   **justify each choice** in your written report (citing either the data
   distribution or socioeconomic reasoning).
5. Implement the ACSIncomeGroups class that is a subclass of  `UnprivilegedGroups`, 
   and implement `get_groups()`, which returns the dictionary with protected attribute names
   as keys and one or more attribute values as dictionary values.

Run `pytest tests/test_groups.py -v` to validate the interface contract.

#### Step 4c – Fairness Auditor ([`pipeline_auditing/fairness/auditor.py`](pipeline_auditing/fairness/auditor.py))

Implement `FairnessAuditor._audit_attribute()` method which for each protected attribute:

1. Splits the data into unprivileged and privileged groups based on the provided 
   UnprivilegedGroups definition.
2. For both groups (priviledged and unpriviledged), computes `GroupMetrics` (positive rate, TPR, FPR,
   precision for binary tasks).
3. Computes aggregate fairness metrics:

| Metric | Formula | Ideal |
|---|---|---|
| **Demographic Parity Difference** (DPD) | `pos_rate(unpriv) − pos_rate(priv)` | 0 |
| **Equal Opportunity Difference** (EOD) | `TPR(unpriv) − TPR(priv)` | 0 |
| **Predictive Parity Difference** (PPD) | `precision(unpriv) − precision(priv)` | 0 |
| **Accuracy Parity Difference** (APD) | `accuracy(unpriv) − accuracy(priv)` | 0 |

4. Returns an `AttributeAuditResult` that then gets aggregated with per-attribute results and helper methods
   `summary_table()` and `group_table(attribute)`.

Run `pytest tests/test_auditor.py -v`.

#### Step 4d – Analysis (for written final report)

Run `python main.py` and answer:

1. Inspect the results of the fairness audit and answer: **Which of the five pipelines is the fairest** with respect to `sex` and `race`?  Justify
   using Demographic Parity Difference (DPD) and Equality of Opportunity Difference (EOD). 
2. Can you explain **why** this pipeline is fairer than the others? 
3. Why is this pipeline still **not** perfectly fair towards `sex` and `race`? 
4. Is there a **fairness–accuracy trade-off** between the five pipelines?
   Discuss using the metrics from Steps 2 and 4.
5. **Reflect on your group choices** (from Step 4b).  Would using a
   different unprivileged group definition change which pipeline appears
   fairest?  Try swapping one definition and report what changes.

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=pipeline_auditing --cov=pipelines --cov-report=term-missing

# Run a specific test file
pytest tests/test_auditor.py -v
```

---

## Fairness Concepts Glossary

**Demographic Parity** (Statistical Parity): A model satisfies demographic
parity if the probability of a positive prediction is equal across groups.

**Equal Opportunity**: A model satisfies equal opportunity if the true
positive rate (recall) is equal across groups.

**Predictive Parity**: A model satisfies predictive parity if precision
(positive predictive value) is equal across groups.

**Proxy discrimination**: Occurs when a model avoids using a protected
attribute directly but instead uses correlated features (proxies) that
reproduce the same discriminatory effect.

**Note**: These criteria are generally *mutually incompatible* in the
presence of base-rate differences between groups (Chouldechova 2017,
Kleinberg et al. 2017). Part of the analysis asks you to observe this
in practice.
