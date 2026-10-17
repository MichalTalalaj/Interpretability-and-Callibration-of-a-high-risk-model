# Interpretable credit-default modeling

A reproducible comparison of regularized logistic regression and XGBoost on
imbalanced financial data. The study evaluates preprocessing, feature selection,
class weighting, probability calibration, decision thresholds and explanation
reliability. The dataset contains **3,000 observations and 183 defaults (6.1%)**.

[Technical report](docs/portfolio-audit.md) · [Data card](docs/data-card.md) ·
[Model card](docs/model-card.md) · [Experiment evidence](docs/evidence/portfolio_v1/README.md) ·
[Presentation PDF](reports/portfolio_v1/results_presentation.pdf) ·
[PowerPoint](reports/portfolio_v1/results_presentation.pptx)

## Results

Results assess model-family selection procedures using two repeats of three-fold
nested **development cross-validation**. Recipes and hyperparameters are selected
by inner-fold average precision (AP).

| Procedure | AP ± fold SD | ROC-AUC | Brier loss | Precision | Recall | Flagged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Constant prior | 0.061 ± 0.001 | 0.500 | 0.0571 | 6.1% | 100.0% | 100.0% |
| Logistic regression | 0.131 ± 0.033 | 0.659 | 0.0570 | 8.4% | 67.0% | 49.3% |
| XGBoost | 0.131 ± 0.010 | 0.706 | 0.0556 | 10.2% | 65.4% | 40.3% |

Probability and decision metrics use independently fitted sigmoid calibration
and development-selected F2 thresholds. F2 expresses a preference for recall;
it does not estimate lending costs. Fold SD is descriptive, not a confidence
interval: repeated observations and overlapping training folds are dependent.
The evidence does not establish a statistically superior model family.

Clipping extreme ratios improves the logistic reference in most paired folds.
Additional feature selection has mixed results, and class weighting does not
consistently improve ranking. SHAP covers 800 held-out development observations
per family with three backgrounds. LIME evaluates 32 successes/errors per family
across three seeds; its lower fidelity and stability for XGBoost are reported.
Complete ablations, uncertainty estimates and runtimes are in the technical report.

## Evaluation and limitations

The historical 600-row test partition has been inspected repeatedly and is
excluded from further fitting, selection, calibration, thresholding and explanation.
The remaining 2,400 observations contain 146 defaults. Within each outer split:

| Role | Observations | Purpose |
| --- | ---: | --- |
| Model fit | 960 | Three-fold inner search; fold-fitted preprocessing and selection |
| Calibration | 320 | Fixed-complexity sigmoid at natural class prevalence |
| Threshold selection | 320 | F2 decision threshold |
| Outer assessment | 800 | Development metrics after selection |

Borrower identifiers, usable timestamps, default horizon and prediction-time
feature availability are undocumented. Stratified random splits cannot establish
borrower or temporal independence. Extreme ratios, correlated predictors, severe
missingness and rare categories limit interpretation. The calibration and
threshold subsets contain only about 20 defaults each. There is no untouched
final assessment cohort or evidence of deployment readiness.

Native/SHAP explanations concern base-model log odds; LIME approximates base
probabilities. Calibration is a separate score mapping. Explanations are not
causal, and LIME perturbations can create implausible financial/category profiles.

## Reproduction

Use 64-bit CPython 3.12. Direct dependencies are pinned; the recorded package
inventory is an environment reference, not a cross-platform hash lock.

```text
python -m venv .venv
```

Activate with `.\.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` on macOS/Linux, then run:

```text
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.experiments.protocol
python -m src.experiments.intercept_study
python -m src.experiments.interpretation
python -m src.experiments.validation
```

The primary study and the explicitly adaptive intercept follow-up use
[separate protocols](config/). Existing experiment directories are protected
from overwrite. [Reproduction details](docs/reproduction.md) explain output paths,
resumption, validation and saved artifacts. The recorded primary run took about
seven minutes; interpretation and uncertainty analysis add runtime.

To render reports from completed, validated runs:

```text
python -m pip install -r requirements-reporting.txt
python tools/build_portfolio_reports.py
python tools/verify_portfolio_reports.py
```

Reports are generated under `reports/generated/`; model artifacts and the theory
PDF remain in ignored output directories. Report generation does not fit models
or rewrite repository documentation.

Saved-model checks cover 126 fitted bundles. Independent refits reproduced the
recorded outer metrics, selected settings and development predictions in the
pinned environment. Some inner L1 search scores vary slightly between refits;
exact inner-search bitwise reproducibility is not claimed. See the
[validation records](docs/evidence/portfolio_v1/README.md) for scope and tolerances.
[GitHub CI](https://github.com/MichalTalalaj/Interpretability-and-Callibration-of-a-high-risk-model/actions/workflows/tests.yml)
checks the implementation on Ubuntu and Windows.

## Repository guide

| Path | Contents |
| --- | --- |
| `src/experiments/` | Canonical development protocol, ablations, interpretation and validation |
| `src/data/`, `src/models/`, `src/evaluation/`, `src/explainability/` | Shared model and evaluation implementation |
| `config/` | Feature configuration, experiment protocols and frozen historical exclusions |
| `tests/` | Numerical, partition-isolation, model and reporting checks |
| `docs/` | Technical report, cards, reproduction guide and compact evidence |
| `tools/` | Diagnosis, report generation and shared rendering utilities |
| `reports/portfolio_v1/` | Recorded presentation PDF and PowerPoint |
| `zbiór_10.csv` | Research dataset |

The [historical reference](docs/historical-reference.md) explains the earlier
single-split results and retained configurations. The separate
[`main.py` API](docs/reference-api.md) supports that reference workflow; it does
not implement the nested evaluation reported above.
