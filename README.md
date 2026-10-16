# Credit scoring project guide

## Current portfolio presentation

- [Slides (PDF)](reports/portfolio_v1/results_presentation.pdf)
- [Editable PowerPoint](reports/portfolio_v1/results_presentation.pptx)

The current 35-slide presentation reports the audited development study. Historical test results are exploratory. The theory PDF remains local.


<!-- PORTFOLIO_CURRENT_START -->
## Current portfolio study

This project is a reproducible investigation of credit-model signal, preprocessing,
feature selection, calibration and explanation reliability. It reports failures
and tradeoffs alongside improvements. **The historical 600-row test has already
been inspected repeatedly. Every new experiment excludes those indices; the
current results are development cross-validation, not untouched final assessment.**

The 2,400 remaining rows contain 146 defaults. Two fixed repeats of three-fold
nested stratified CV use disjoint model-search, calibration, threshold and outer
assessment roles. Each outer base fit has 960 rows, calibration 320, threshold
selection 320, and outer assessment 800. Preprocessing/selection is learned inside
inner folds. No borrower identifier or usable timestamp is supplied, so borrower
and temporal independence cannot be verified. Default horizon and prediction-time
feature availability are undocumented.

| Nested family procedure | Mean AP ± fold SD | ROC-AUC | Brier | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| dummy | 0.061 ± 0.001 | 0.500 | 0.0571 | 6.1% | 100.0% |
| glassbox | 0.131 ± 0.033 | 0.659 | 0.0570 | 8.4% | 67.0% |
| blackbox | 0.131 ± 0.010 | 0.706 | 0.0556 | 10.2% | 65.4% |

These rows use independent sigmoid calibration and development-selected F2
thresholds. Fold SD is descriptive; overlapping folds/repeated rows are dependent.
Full counts, flagged volume, conditional paired uncertainty and failures are in
the [technical audit](docs/portfolio-audit.md). There is no statistically established
family winner and no claimed deployment-ready model.

The planned study includes dummy, weighted/unweighted L2, clipping/scaling,
missingness and category ablations, whole-field supervised selection, redundancy
reduction, embedded L1, frozen historical references and matched XGBoost search
paths. A **separately labeled adaptive intercept study** addresses liblinear's
penalized intercept; it is not independent confirmation. SHAP now covers 800
held-out development rows/family across three backgrounds. LIME covers 32
success/error cases/family across three seeds and records poor tree-surrogate fidelity.

### Reproduce the current work

Use CPython 3.12 and the pinned dependencies below. From a clean checkout:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.experiments.protocol
.\.venv\Scripts\python.exe -m src.experiments.intercept_study
.\.venv\Scripts\python.exe -m src.experiments.interpretation
.\.venv\Scripts\python.exe -m src.experiments.validation
.\.venv\Scripts\python.exe -m pip install -r requirements-reporting.txt
.\.venv\Scripts\python.exe tools/build_portfolio_reports.py
.\.venv\Scripts\python.exe tools/verify_portfolio_reports.py
```

On macOS/Linux use `.venv/bin/python`. The recorded full primary run took about
7 minutes on this Windows/CPython 3.12.14 environment; timings are hardware-dependent.
The adaptive study and interpretation are additional stages. Model predictions
were verified after loading all 126 saved base models. Nineteen tests pass locally.
The [CI workflow](.github/workflows/tests.yml) installs dependencies and runs these
checks on Ubuntu and Windows; see its [live status](https://github.com/MichalTalalaj/Interpretability-and-Callibration-of-a-high-risk-model/actions/workflows/tests.yml).

Existing experiment directories are protected from overwrite. `--resume` requires
identical code/data/configuration. For a separate reproduction, use `--output-dir`
on the two experiment commands, `--run-dir` on interpretation/validation/reporting,
`--followup-dir` on validation/reporting, and `--report-dir` on reporting/verification.
The report builder's `--local-only` preserves the tracked summary during a rerun.
The recorded runs also retain original source snapshots. A cold-refit
check of two complete recipes over all six folds reproduced saved predictions,
thresholds and labels using the portable tracked exclusion manifest.

Publication verification also refits the primary and adaptive studies in an export
of the staged repository, without copying ignored artifacts. Its results are recorded
in [publication_check.json](docs/evidence/portfolio_v1/publication_check.json).
Some inner L1 search scores vary slightly between refits; exact inner-search
bitwise reproducibility is not claimed. The check separately compares selected
settings, outer metrics and saved development predictions against the recorded run.
It uses the existing pinned environment; CI installs dependencies independently.

### Evidence and limits

- [Technical report](docs/portfolio-audit.md), [data card](docs/data-card.md),
  [model card](docs/model-card.md), and [tracked fold-level evidence](docs/evidence/portfolio_v1/README.md).
- `config/historical_test_indices.csv` freezes the original exclusion indices
  without labels. `dataset_identity.json` verifies content/order and index membership
  using CRLF/LF-normalized bytes, while retaining the original raw-byte hash too.
- New local outputs: `artifacts/portfolio_v1`, `artifacts/portfolio_intercept_v1`,
  and `reports/portfolio_v1/{results_presentation.pptx,results_presentation.pdf,theory_guide.pdf,technical_report.pdf}`.
- Original baseline/selected models, metrics, explanations and earlier PDFs remain
  unchanged. The current presentation PDF and PowerPoint are published; fitted artifacts,
  the theory PDF and other generated reports remain Git-ignored.
- Verified audit findings include severe missingness, rare categories, duplicate
  predictor columns and extreme/infinite ratios. Borrower identity, timing and
  feature availability need source documentation before real risk decisions.

The remaining sections document the **historical baseline and legacy CLI** for
reproduction and API reference. They are not the current assessment protocol.
<!-- PORTFOLIO_CURRENT_END -->


## Historical baseline and legacy workflow

This project compares regularized logistic regression (glassbox) with XGBoost
(blackbox) on financial statement features. Both use identical stratified splits,
shared evaluation and the same `fit`, `predict`, `predict_proba`, `evaluate`,
`explain` interface. Positive class `default = 1` denotes default.

| Approach | Performance tradeoff | Interpretation |
| --- | --- | --- |
| Logistic regression | Fast linear log-odds baseline; limited interactions without feature engineering. | Signed coefficients and exact additive log-odds contributions. Correlated features complicate coefficient interpretation. |
| XGBoost | Captures nonlinear effects and interactions; requires stronger overfitting controls. | Global gain importance and local Tree SHAP; explanations describe behavior, not causality. |

The default run on the included 3,000-row CSV produced the following results on
the shared 600-row test partition (seed 42, no calibration):

| Metric | Glassbox | Blackbox |
| --- | ---: | ---: |
| ROC-AUC | 0.6727 | 0.7562 |
| PR-AUC (average precision) | 0.1222 | 0.1934 |
| Gini | 0.3454 | 0.5125 |
| KS | 0.2772 | 0.4960 |
| Brier score | 0.0765 | 0.0565 |
| PSI (training reference) | 0.0124 | 0.0152 |
| Recall at threshold 0.5 | 0.1081 | 0.0270 |

XGBoost improves ranking and probability losses in this baseline, while logistic
regression provides direct signed coefficients and exact additive explanations.
Both have low default recall at 0.5; choose an operational threshold on development
data according to costs and risk appetite. These single-split results are not a
claim of general superiority. Notebook scores are historical and not directly
comparable: the new baseline changes preprocessing, partitions and modeling choices. The
[audit](docs/refactoring-audit.md) documents the exact migration and removal list.

## Installation

Use 64-bit **CPython 3.10–3.12** (3.12 recommended). This pinned baseline does not
claim support for every subsequent Python release. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py --help
```

On macOS/Linux, use `python3.12 -m venv .venv` and `.venv/bin/python` instead.
Direct dependencies and SHAP's Numba/LLVM dependencies are pinned; pip resolves
other transitive dependencies. This is not a universal hash-locked environment.
Baseline release records: [scikit-learn](https://pypi.org/project/scikit-learn/1.5.2/),
[XGBoost](https://pypi.org/project/xgboost/2.1.4/),
[SHAP](https://pypi.org/project/shap/0.46.0/).
No Jupyter server or existing model pickle is needed.

## Run training, evaluation and explanations

The default CLI trains, evaluates every partition, and produces native and SHAP
explanations for a deterministic test sample:

```powershell
.\.venv\Scripts\python.exe main.py --model glassbox
.\.venv\Scripts\python.exe main.py --model blackbox
.\.venv\Scripts\python.exe main.py --model glassbox --explain native shap lime
.\.venv\Scripts\python.exe main.py --model blackbox --skip-explain
.\.venv\Scripts\python.exe main.py --model blackbox --config config/config.yaml --data path/to/data.csv --output-dir artifacts/experiment
```

Training and evaluation always run; `--skip-explain` disables explanation only.
Fresh, imbalance-aware parameter searches are available with `--tune`:

```powershell
.\.venv\Scripts\python.exe main.py --model glassbox --tune --output-dir artifacts/tuned --skip-explain
.\.venv\Scripts\python.exe main.py --model blackbox --tune --output-dir artifacts/tuned --skip-explain
```

Each search evaluates 40 fresh candidates over five shuffled, stratified training
folds, selecting the highest mean **average precision** (PR-AUC). ROC-AUC is also
recorded. This uses scikit-learn's
[multi-metric randomized search](https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.RandomizedSearchCV.html).
Previous model parameters and saved models are not used as starting points.
Preprocessing is refitted within every fold. Logistic regression searches L1/L2
penalties, regularization across five orders of magnitude, and unweighted,
balanced or partially weighted classes. XGBoost searches tree count/depth,
learning rate, sampling, child weights, L1/L2 regularization, split penalties and
positive-class weighting. Its positive weight is `balance_multiplier * n_negative /
n_positive`, calculated separately on each fit's labels; `null` means weight 1.
Unweighted candidates remain eligible because weighting does not always improve
minority-class ranking.

The final classifier fits only the training partition. Its decision threshold
maximizes validation **F2**, favoring default recall at the expense of precision.
Set `tuning.threshold_beta` to change this preference. If calibration is enabled,
validation is split into disjoint stratified halves for calibration and threshold
selection; no class weights are applied to calibration. The historical test partition
is reported for reproduction only; it has already been inspected and is not a final
independent assessment. Validation scores are development scores and are
optimistic after selection. F2 is an explicit default preference, not a threshold
derived from lending costs. Class-weighted probabilities may need calibration;
inspect Brier score and reliability curves before treating them as PD estimates.

Override search budgets with `--n-iter`, `--cv-folds` and `--search-jobs`, or edit
the `tuning` configuration section. `tuning.json` records the search space,
winning parameters, CV scores, class counts, threshold selection and row indices;
`cv_results.csv` records every candidate and fold score. The regular model,
prediction and effective-configuration artifacts are also saved. Keep a separate
output directory to preserve the original baseline.

The completed dataset run is documented in [the tuning report](docs/tuning-results.md),
with reusable configurations in [tuned_glassbox.yaml](config/tuned_glassbox.yaml)
and [tuned_blackbox.yaml](config/tuned_blackbox.yaml). To fit the selected parameters
without repeating the searches:

```powershell
.\.venv\Scripts\python.exe main.py --model glassbox --config config/tuned_glassbox.yaml
.\.venv\Scripts\python.exe main.py --model blackbox --config config/tuned_blackbox.yaml
```

The API supports loading and evaluating a saved model without retraining:

```python
import joblib
from src.config import load_config
from src.data import CreditDataLoader
from src.models import GlassboxModel, BlackboxModel

config = load_config("config/config.yaml")
loader = CreditDataLoader(config)
X, y = loader.load()
splits = loader.split(X, y)
X_train, y_train = splits["train"]
X_val, y_val = splits["validation"]
X_test, y_test = splits["test"]
model = GlassboxModel(config)  # BlackboxModel implements the same interface
model.fit(X_train, y_train, X_val, y_val)
labels = model.predict(X_test)
probabilities = model.predict_proba(X_test)  # (n, 2), classes [0, 1]
metrics = model.evaluate(X_test, y_test, reference=model.predict_proba(X_train)[:, 1])
report = model.explain(X_test, background=X_train, methods=["native", "shap"])
saved = joblib.load("artifacts/glassbox/model.joblib")
new_metrics = saved.evaluate(X_test, y_test)
```

Load only trusted joblib files. Use the loader for new labeled CSVs to normalize
categorical types and infinite values. The model API expects similarly prepared
DataFrames. Unseen categories are ignored by one-hot encoding. Invalid numerical
text fails explicitly; numerical missing values use training imputation statistics.

## Configuration and leakage prevention

`config/config.yaml` centralizes paths, feature lists, seed, split fractions,
preprocessing, hyperparameters, calibration, decision threshold, PSI bins, and
explanation budgets. YAML paths are relative to the YAML file; CLI overrides are
relative to the working directory. `features: null` selects all columns except the
target and dropped columns. Use an explicit allowlist when extending the dataset
to avoid accidentally including identifiers or post-outcome fields.

Defaults are **60% train / 20% validation / 20% test**, stratified with a fixed seed.
Imputation, encoding, constant removal and scaling fit only on training data.
Industry `pkdKod` uses unsupervised one-hot encoding, replacing supervised WoE.
Logistic regression scales numerical features and missing indicators; XGBoost
does not scale. Entirely missing training features pass through imputation and
are removed if constant. Explanation calls transform only and never refits.

Set `calibration.method` to `sigmoid` or `isotonic` for validation-only calibration.
Default is `none`. Sigmoid fits logistic regression to clipped base-model logits.
The base classifier is not refitted. Hyperparameters, calibration choice and
thresholds in this legacy workflow were selected on development data. The
current portfolio protocol excludes the historical test, whose results are exploratory. Threshold defaults to 0.5. No forced 4% PD shift, test-selected
threshold or test-derived rating cutoffs are applied. XGBoost uses a fixed tree
count without early stopping. See the audit for other deliberate baseline changes.

Random splits do not establish borrower-group or out-of-time independence. For
repeated borrowers or temporal deployment, configure a suitable group/time split
before interpreting results as deployment estimates. Feature availability at the
decision date must be established from source-data provenance.

## Metrics and outputs

Runs write `artifacts/<model>/` or `<output-dir>/<model>/`. Repeating a model run
replaces its files; use different output roots to retain experiments. Skipping
explanation removes the old explanation file.

| Artifact | Contents |
| --- | --- |
| `model.joblib` | Wrapper, fitted preprocessing, classifier, optional calibrator |
| `config.yaml` | Effective configuration with resolved paths and overrides |
| `environment.json` | Python and runtime package versions |
| `metrics.json` | Train, validation and test metrics plus calibration curves |
| `predictions_<split>.csv` | Original row index, label, PD, prediction |
| `explanations.json` | Feature names, row indices, native/SHAP/LIME explanations |

- ROC-AUC measures ranking; **Gini = 2 × ROC-AUC − 1**.
- PR-AUC is **average precision**, not trapezoidal PR integration.
- KS is the maximum absolute empirical CDF difference between default and
  non-default score distributions (two-sample KS, with ties handled consistently).
- PSI compares scores to training predictions, using reference-only quantile
  edges. Duplicate quantiles collapse; constant/near-constant reference scores
  use fixed probability bins. Infinite outer edges include scores outside the
  reference range; normalized additive smoothing keeps logs finite. Train PSI is
  zero. Train/test PSI also reflects overfitting, so it is not alone evidence of
  temporal drift; no universal alert threshold is assumed.
- Accuracy, precision, recall, F1 and confusion matrix use the configured threshold.
  Matrix order is `[[TN, FP], [FN, TP]]`; undefined precision/recall use zero.
- Brier score, log loss, equal-width expected calibration error (ECE) and quantile
  reliability curves assess probability quality. Lower losses are better.

Metrics require both classes, finite probabilities and aligned labels. Very small
datasets may not support a stratified three-way split.

## Explanation scope

Logistic native contributions reconstruct base log odds as intercept plus their
sum. Numerical coefficients refer to training standard deviations, categorical
coefficients to indicator changes. XGBoost native importance is normalized gain.
SHAP exports local contributions, base values and mean absolute contributions for
the sampled global summary, all in base-model log-odds units.

LIME approximates base default probabilities in the transformed feature space;
perturbations may create unrealistic one-hot combinations or correlated financial
ratios. Calibration is excluded from native/SHAP/LIME explanations. Reports
explicitly include both base and calibrated PDs. Arrays align with exported feature
names and sampled indices. Supply training rows as background. Default global
SHAP summaries cover only 20 explained rows; increase budgets for wider analysis.

## Repository map

```text
.
├── main.py
├── requirements.txt
├── config/                      # baseline, frozen exclusions and current protocols
├── src/                         # importable package; each directory has __init__.py
│   ├── config.py
│   ├── experiments/             # canonical development-only experiment workflow
│   ├── data/
│   │   ├── loader.py
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── base.py
│   │   ├── glassbox.py
│   │   └── blackbox.py
│   ├── explainability/explainers.py
│   └── evaluation/metrics.py
├── tests/                       # numerical, partition and interpretation checks
├── docs/                        # technical report, cards and compact evidence
├── tools/                       # diagnosis, reproduction and report generation
├── reports/portfolio_v1/        # published presentation PDF and PowerPoint
├── zbiór_10.csv
└── artifacts/                   # generated, ignored by Git
```

## Historical report generator

The earlier report package is generated locally under the Git-ignored `reports/`
directory: `results_presentation.pptx`, a matching `results_presentation.pdf`, and
`theory_guide.pdf`. It covers baseline and tuned results, both models' pros and
cons, actual native/SHAP outputs, and a labeled supplementary one-case LIME
demonstration. Saved models are loaded without retraining. For the current published
presentation, use the links above and `tools/build_portfolio_reports.py`.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-reporting.txt
.\.venv\Scripts\python.exe tools/build_reports.py
.\.venv\Scripts\python.exe tools/verify_reports.py
```

`reports/README.md` describes the local exports and checks. Source narratives are
in `tools/report_content.py` and `tools/theory_content.md`.

## Validation

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
```

Tests cover disjoint deterministic splits, target exclusion, exact metric examples,
PSI edge cases, unseen categories, entirely missing features, unchanged fitted
preprocessing during explanation, serialization, calibration isolation, SHAP
additivity and LIME for both models. Historical notebooks and presentation material
have been removed; use Git history to recover the original experiments.
