# Project reading guide

Read in the order below. The sequence starts with the research question, follows
the current experiment from input data to results, and then covers the reference
API, reporting utilities and verification checks.

The project compares logistic regression (**glassbox**) with XGBoost
(**blackbox**) for predicting `default = 1`. Its current deliverable is a
reproducible development study. The already inspected historical 600-row test
partition is excluded from that study; the remaining 2,400 rows form its
development pool. Current results come from repeated nested cross-validation.

There are two execution paths to keep track of:

```text
Current study:
config/portfolio.yaml
  -> src/experiments/protocol.py
     -> development_data() -> loader + frozen historical exclusions
     -> recipes() -> financial transforms + preprocessing + classifier
     -> fit-only inner search -> base-model refit
     -> independent calibration -> independent threshold selection
     -> outer development assessment -> saved results
  -> intercept_study.py     (separate adaptive follow-up)
  -> interpretation.py      (explanations of primary-study models)
  -> validation.py          (saved-model checks, stability, uncertainty)
  -> tools/build_portfolio_reports.py -> reports/generated/
     -> report_rendering.py + report_references.py
  -> tools/verify_portfolio_reports.py -> layout checks + previews

Single-split reference workflow:
main.py -> CreditDataLoader.load()/split()
        -> optional tune_model() -> GlassboxModel / BlackboxModel
        -> evaluation + optional explanations -> artifacts/<model>/
```

## 1. Understand the question and the limits

1. Read [README.md](../README.md): **Results**, **Evaluation and limitations**,
   **Reproduction**, then **Repository guide**. These introduce the current study,
   headline evidence, execution sequence and repository structure.
2. Read [data-card.md](data-card.md). Learn the target, 3,000-row dataset,
   imbalance, missingness, extreme values and unresolved provenance questions.
3. Read [model-card.md](model-card.md). Learn what the experiment establishes,
   how the four data roles work, and what the saved fold models represent.
4. Skim [portfolio-audit.md](portfolio-audit.md). Focus first on the study design,
   headline comparison and limitations. Return to the detailed tables after
   reading the implementation.

**Checkpoint:** Explain why a historical test score and a current outer-CV score
are different kinds of evidence, and why neither establishes future deployment
performance here.

## 2. Learn the inputs and configuration

5. Read [config/config.yaml](../config/config.yaml). Identify the CSV path,
   `default` target, excluded field, categorical fields, baseline model settings,
   evaluation settings and explanation settings. Its 60/20/20 split belongs to
   the historical workflow.
6. Read [src/config.py](../src/config.py). `load_config()` resolves data/output
   paths relative to the YAML file and validates basic evaluation settings.
7. Read [config/portfolio.yaml](../config/portfolio.yaml). This controls the
   current study: two seeds, three outer folds, three inner folds, independent
   fit/calibration/threshold fractions and diagnostic/explanation budgets.
8. Inspect [config/dataset_identity.json](../config/dataset_identity.json), then
   the header and a few entries of
   [config/historical_test_indices.csv](../config/historical_test_indices.csv).
   These protect dataset content/order and preserve historical exclusions; the
   header and sample entries establish the manifest format.
9. Inspect the header and a few rows of [zbiór_10.csv](../zbiór_10.csv), using the
   data card as a reference. A small sample establishes the schema; later,
   `feature_audit.csv` gives a more useful field-by-field view.

**Checkpoint:** Distinguish baseline model configuration from current experiment
configuration. The current candidate estimators and search spaces are declared
in `recipes.py`; they are not simply the baseline model settings in YAML.

## 3. Follow the data into the current study

10. Read [src/data/loader.py](../src/data/loader.py), starting with
    `CreditDataLoader.load()`. Follow column selection, target validation,
    categorical normalization, numeric conversion and infinity-to-missing
    conversion. Read `split()` afterward as historical-workflow context.
11. Read [src/experiments/data.py](../src/experiments/data.py) in this order:
    `load_protocol()` -> `development_data()` -> `audit()` -> preservation and
    serialization helpers. This connects the YAML, dataset hashes, frozen
    exclusions and development-only audits. Optional checks against old local
    artifacts do not make those artifacts mandatory for a clean reproduction.

**Checkpoint:** Trace how the project obtains 2,400 development rows and 146
defaults while keeping the 600 historical test rows out of the current study.

## 4. Understand how a candidate model is constructed

12. Read [src/experiments/transforms.py](../src/experiments/transforms.py), first
    `pipeline()` and `preprocessing()`, then `FinancialFeatures.fit()` and
    `transform()`, and finally `CheckedLogisticRegression`.

    The pipeline is raw-field selection/clipping -> imputation/encoding/scaling
    -> classifier. Learn which statistics are learned during `fit()` and reused
    during `transform()`. Numerical clipping, missingness filtering, redundancy
    reduction and supervised selection operate on raw fields. Selecting a
    categorical field keeps its complete one-hot group. Nonconverged logistic
    candidates fail explicitly.
13. Read [src/models/weighted_xgboost.py](../src/models/weighted_xgboost.py).
    `FoldWeightedXGBClassifier.fit()` recomputes the positive-class weight from
    each fit's labels, including inner folds.
14. Read [src/experiments/recipes.py](../src/experiments/recipes.py). Start with
    `Recipe`, then follow `recipes()`: dummy benchmark, logistic ablations,
    historical reference configurations and matched XGBoost candidate paths.
    Check each recipe's hypothesis, search grid and `selectable` flag.

**Checkpoint:** Explain a recipe as a preprocessing/model/search specification,
and explain why some historical reference recipes are reported but excluded
from family selection.

## 5. Separate ranking, probability calibration and decisions

15. Read [src/evaluation/metrics.py](../src/evaluation/metrics.py), starting with
    `classification_metrics()`. Separate ranking metrics (AP/ROC-AUC), probability
    losses (Brier/log loss) and threshold-based outcomes (precision/recall/confusion
    matrix). The project's `pr_auc` is average precision. Read the PSI helper
    afterward; it is optional in the shared metric API.
16. Read **only `select_threshold()`** in
    [src/tuning.py](../src/tuning.py) for now. It selects the F2 threshold from
    labeled threshold data, breaking score ties toward the higher threshold.
    The rest of this file implements the historical randomized search.
17. Read [src/experiments/calibration.py](../src/experiments/calibration.py).
    `MonotoneSigmoid` maps clipped base logits through a fitted sigmoid with a
    nonnegative slope, penalized slope and unpenalized intercept.

**Checkpoint:** Explain why improving probability levels can improve Brier/log
loss without improving ranking, and why threshold selection needs its own rows.

## 6. Read the current experiment orchestrator

18. Read [src/experiments/protocol.py](../src/experiments/protocol.py) in this
    order, rather than strictly from top to bottom:

    1. `main()` and the beginning of `run()`: configuration, output protection,
       input identity and loading.
    2. `partitions()`: disjoint fit, calibration, threshold and assessment roles.
    3. `fit_recipe()`: inner `GridSearchCV` on fit rows; base-model refit;
       independent calibration; raw/sigmoid threshold selection; assessment;
       model, prediction and result exports.
    4. The repeat/fold loop in `run()`: identical partitions across recipes and
       family recipe selection using **inner AP**.
    5. `evaluate()` and `summarize()`: fold tables, mean/SD summaries, separate
       pooled-repeat scores and paired ablations.
    6. `permutation_control()` and completion/preservation checks.

    Per outer fold, the 2,400 development rows divide into 800 assessment rows
    and 1,600 other rows. Those 1,600 divide into 960 fit/search, 320 calibration
    and 320 threshold rows. Inner preprocessing and selection are refitted within
    the 960-row fit subset's training folds. There are six outer folds in total
    across the two repeats.

    `selector_glassbox` and `selector_blackbox` are procedures that choose an
    eligible recipe independently in each outer fold. They are not additional
    separately trained estimator classes.

**Checkpoint:** Trace one recipe through one fold and identify exactly which
labels can affect its parameters, calibrator, threshold and reported metrics.

## 7. Read the adaptive follow-up

19. Read [config/portfolio_intercept.yaml](../config/portfolio_intercept.yaml),
    then [src/experiments/intercept_study.py](../src/experiments/intercept_study.py).
    It reuses partitioning, fitting and summaries from the primary protocol.
    Follow the three logistic specifications: two `lbfgs` L2 variants and a
    `liblinear` L1 variant with larger `intercept_scaling`.

**Checkpoint:** Explain the intercept-penalization motivation and why this
separately labeled adaptive study is not independent confirmation of the primary
results. The L2 comparison changes both solver and intercept treatment.

## 8. Follow explanations and their reliability checks

20. Read [src/experiments/interpretation.py](../src/experiments/interpretation.py).
    Start with `run()`, then inspect `raw_field()` and `connected_groups()`.
    Follow inner-AP recipe choice for fixed repeat 0/fold 0, fitted transforms,
    all 800 assessment-row SHAP explanations, three fit-only backgrounds,
    success/error case sampling, LIME seed comparisons and saved diagnostics.

    Native/SHAP explanations concern base-model log odds. LIME approximates base
    probabilities. Calibrated probabilities and decisions are recorded separately.
    Background/perturbation stability does not establish retraining stability.

**Checkpoint:** Explain the difference between exact SHAP reconstruction of a
base score and a faithful, stable explanation of a calibrated credit decision.

## 9. Check how the project verifies itself

21. Read [src/experiments/validation.py](../src/experiments/validation.py):
    `validate_run()` -> `selection_stability()` -> `bootstrap()` -> `main()`.
    It checks role separation and historical exclusions, reloads saved models,
    recomputes predictions/metrics and exports fit/stability diagnostics. Bootstrap
    intervals are conditional on fitted predictions; the bootstrap does not refit.
22. Read [tests/test_experiments.py](../tests/test_experiments.py), especially
    `test_assessment_labels_cannot_change_fitted_models_calibration_or_thresholds`.
    Also inspect frozen-transform, disjoint-role, monotone-calibration,
    matched-candidate and portable-exclusion tests.
23. Read the verification scope in [reproduction.md](reproduction.md), then
    inspect [reproduction_check.json](evidence/portfolio_v1/reproduction_check.json)
    and [publication_check.json](evidence/portfolio_v1/publication_check.json).
    These record two-recipe and full-study refit comparisons, respectively.
    The full-study record includes tolerances, selected-setting checks,
    prediction comparisons and observed inner L1 score variation. The explanation
    study was not repeated in the full-refit check.

**Checkpoint:** Distinguish unit tests, saved-model reload checks and cold-refit
reproduction. Each checks a different part of reproducibility.

## 10. Read the evidence, now that its columns have meaning

24. Read [docs/evidence/portfolio_v1/README.md](evidence/portfolio_v1/README.md),
    then inspect its files in this order:

    | Order | Files | What to look for |
    | --- | --- | --- |
    | 1 | `experiment_plan.json` | Recipe hypotheses, candidate spaces and selection eligibility. |
    | 2 | `data_audit.json`, `feature_audit.csv`, `correlated_pairs.csv` | Dataset limitations and measured field problems. |
    | 3 | `comparison.csv`, `fold_results.csv`, `repeat_results.csv` | Aggregate results, fold settings/variation and separately pooled repeat scores. Compare the same `mode` and procedure. |
    | 4 | `paired_ablations.csv`, `conditional_bootstrap.json`, `permutation_control.json` | Controlled differences, conditional uncertainty and the fixed-pipeline signal diagnostic. |
    | 5 | `model_fit_diagnostics.csv`, `feature_selection_stability.csv` | Convergence/intercepts/calibration and retained/active-field overlap. |
    | 6 | `intercept_*` | Corresponding adaptive follow-up outputs, kept separately labeled. |
    | 7 | `interpretation_summary.json`, `interpretation_cases.csv`, `interpretation_lime_diagnostics.csv`, `interpretation_lime_stability.csv` | Explained models, actual errors, surrogate fidelity and seed sensitivity. |
    | 8 | `completion.json`, `validation.json`, `intercept_completion.json`, `intercept_validation.json`, `preservation_check.json`, `reproduction_check.json`, `publication_check.json` | Recorded execution and validation claims and their scopes. |
    | 9 | `environment.json`, `environment-freeze.txt` | Recorded runtime and installed package inventory. |
    | 10 | `historical_diagnosis.json` | Frozen historical context; its old test results remain exploratory. |

25. Reread [portfolio-audit.md](portfolio-audit.md) fully, tracing each major claim
    back to its implementation and evidence file.

## 11. Understand the single-split reference API and CLI

26. Read [reference-api.md](reference-api.md). It documents the retained
    `main.py` interface, simpler split/calibration path, output files and metric
    conventions. This path does not implement the nested family-selection study.
27. Read [src/data/preprocessing.py](../src/data/preprocessing.py), then
    [src/models/base.py](../src/models/base.py). Follow the shared `CreditModel`
    interface: `fit`, `predict_proba`, `predict`, `evaluate`, `explain`. This uses
    the simpler baseline preprocessing and its own optional calibration path.
28. Read [src/models/glassbox.py](../src/models/glassbox.py),
    [src/models/blackbox.py](../src/models/blackbox.py) and
    [src/models/__init__.py](../src/models/__init__.py). These choose estimators
    behind the common API and expose the CLI's `MODELS` registry.
29. Read the rest of [src/tuning.py](../src/tuning.py): `search_spec()` and
    `tune_model()`. Contrast this broad randomized search with the current
    protocol's declared recipe grids.
30. Read [src/explainability/explainers.py](../src/explainability/explainers.py).
    This is the smaller shared-API explanation path, distinct from the current
    larger interpretation reliability study.
31. Read [main.py](../main.py) from top to bottom. With its dependencies familiar,
    the historical load -> split -> optional tune -> fit -> threshold -> evaluate
    -> save -> explain sequence should be straightforward.
32. Read [tests/test_pipeline.py](../tests/test_pipeline.py), then
    [tests/test_tuning.py](../tests/test_tuning.py), for the API and tuning
    behavior contracts.
33. Read [historical-reference.md](historical-reference.md), then
    [config/tuned_glassbox.yaml](../config/tuned_glassbox.yaml) and
    [config/tuned_blackbox.yaml](../config/tuned_blackbox.yaml). These preserve
    selected historical settings; the current study evaluates historical
    references alongside its declared ablations. The reference document explains
    the earlier evaluation design and why those test scores are exploratory.

**Checkpoint:** Explain why running `main.py` does not reproduce the current
nested portfolio study, even though both workflows share loading, metrics,
threshold selection and some estimator support.

## 12. Finish with reporting and repository support

34. Read [reports/portfolio_v1/README.md](../reports/portfolio_v1/README.md), then
    the [current presentation PDF](../reports/portfolio_v1/results_presentation.pdf).
    The matching PowerPoint is editable; it need not be read separately.
35. Read [tools/portfolio_theory.md](../tools/portfolio_theory.md) for the methods
    explanation and [tools/report_references.py](../tools/report_references.py)
    for its primary references. The theory source also serves as a reference
    while reading the implementation.
36. Read [tools/report_rendering.py](../tools/report_rendering.py): chart helpers,
    font registration, `wrap()` and `Deck`. These shared utilities implement
    matching PDF/PowerPoint geometry and editable slide text. They are independent
    of model fitting and experiment selection.
37. Read [tools/build_portfolio_reports.py](../tools/build_portfolio_reports.py):
    `main()` -> `load()` -> `charts()` -> `report_markdown()` -> `presentation()`
    -> PDF helpers -> `report_readme()`. It reads completed, validated runs and
    writes figures, report Markdown, slides and PDFs to `reports/generated/` by
    default. `--run-dir`, `--followup-dir` and `--report-dir` select alternate
    inputs and outputs. It does not fit models or rewrite repository documentation.
    The recorded presentation snapshot remains in `reports/portfolio_v1/`.
38. Read [tools/verify_portfolio_reports.py](../tools/verify_portfolio_reports.py)
    for report structure, text bounds, slide bounds and rendered previews.
39. Read [tests/test_reporting.py](../tests/test_reporting.py). Its checks cover
    matching slide exports, preservation of curated documentation during Markdown
    generation, and a CLI help invocation without output side effects. Then skim
    [tools/diagnose_portfolio.py](../tools/diagnose_portfolio.py) for read-only data
    and frozen-history inspection without original model artifacts.
40. Finish with [requirements.txt](../requirements.txt),
    [requirements-reporting.txt](../requirements-reporting.txt),
    [.github/workflows/tests.yml](../.github/workflows/tests.yml),
    [.gitignore](../.gitignore) and the remaining `__init__.py` files. These
    explain dependency groups, Ubuntu/Windows CI including reporting tests,
    ignored generated outputs and package exports. Reporting tests require the
    optional reporting dependencies; CI installs both dependency groups.

The generated `artifacts/` directories contain fitted models, detailed arrays and
run metadata. For a concrete execution trace after step 18, inspect a completed
fold's `partitions.json`, `inner_splits.json`, `inner_cv.csv`, `results.json` and
`assessment_predictions.csv`. A `model.joblib` is a serialized object rather than
a readable source file. The compact evidence package is the portable starting point.

The recorded presentation is in `reports/portfolio_v1/`; report generation defaults
to the separate, ignored `reports/generated/` directory. The theory PDF and fitted
model artifacts remain local. Following this reading order does not require
rerunning experiments.
