# Recorded development evidence

This compact package supports review without model binaries. The full local runs
are in ignored `artifacts/portfolio_v1` and `artifacts/portfolio_intercept_v1`.
The latter is an explicitly adaptive intercept follow-up.

- `comparison.csv`, `fold_results.csv`, `repeat_results.csv`: complete planned
  recipe/procedure summaries, individual folds/settings and separate pooled-repeat scores.
- `paired_ablations.csv`: all predeclared paired AP/loss/decision contrasts.
- `experiment_plan.json`: hypotheses, candidate spaces and selection eligibility.
- `feature_audit.csv`, `data_audit.json`, `correlated_pairs.csv`: numerical audit
  with unresolved source-data limitations recorded explicitly.
- `historical_diagnosis.json`: frozen descriptive evidence, including train/validation
  gaps, old CV/search coverage and old tree structure. Historical test metrics
  inside it are exploratory; no new run uses them to select settings.
- `conditional_bootstrap.json`: paired original-row resampling conditional on
  already fitted predictions. Intervals exclude retraining/adaptive-design uncertainty.
- `permutation_control.json`: fixed clipped-L2 diagnostic and all 99 null scores.
- `feature_selection_stability.csv`: active/retained raw-field overlap across folds.
- `model_fit_diagnostics.csv`: convergence counts, intercepts, class prevalence,
  fitted raw mean scores and independent calibration parameters.
- `intercept_*`: all corresponding adaptive follow-up results and validation.
- `interpretation_*`: all case outcomes, LIME diagnostics and stability summary;
  detailed SHAP arrays/weights remain local.
- `environment.json`, `completion.json`, `validation.json`, `preservation_check.json`,
  `reproduction_check.json`: execution, role isolation, model roundtrips, original
  artifact preservation and cold-refit checks.
- `environment-freeze.txt`: exact installed-package inventory, including optional
  reporting dependencies, from the recorded Windows/CPython 3.12.14 environment.
  It is not a universal cross-platform or hash-locked dependency specification.
- `publication_check.json`: independent refits of both studies, selected
  settings, outer metrics and development-prediction comparisons. Small observed
  inner L1 search-score variation is recorded separately; its cause was not isolated.

Mean AP is the six-fold outer-development mean. Fold SD is descriptive; rows are
repeated across seeds and training folds overlap. Each base fit uses 960 rows,
with 320 independent calibration and 320 independent threshold rows. These
results are not comparable to historical scores as an untouched-test improvement.

The dataset identity and portable 600-row exclusion list are in `config/`.
Reproduce using the current README commands. No default horizon, true borrower
independence, future cohort performance or deployment readiness is established.
