# Reproduction and artifacts

The canonical entry points are the modules in `src/experiments/`. Run commands
from the repository root with the virtual environment activated. Installation
and the standard sequence are in the [README](../README.md#reproduction).

## Inputs and protocols

- [Primary protocol](../config/portfolio.yaml): two fixed outer seeds, three outer
  folds, three inner folds, independent calibration and threshold roles, AP search,
  permutation control and conditional bootstrap settings.
- [Adaptive follow-up](../config/portfolio_intercept.yaml): separate intercept
  treatment experiments using the same partitions. This is adaptive development
  analysis, not independent confirmation of an outer-selected winner.
- [Feature configuration](../config/config.yaml): CSV loading, target, categorical
  fields and excluded constant field. Paths in YAML are relative to that file.
- [Dataset identity](../config/dataset_identity.json) and
  [historical exclusions](../config/historical_test_indices.csv): protect data
  content/order and the exact 600 excluded row indices. Only CRLF/LF differences
  are normalized in the portable dataset hash; the raw-byte hash is also retained.

No original model pickle or ignored baseline artifact is needed. A clean checkout
uses the recorded historical diagnosis as descriptive reference evidence. It does
not reassess historical test observations.

## Stages and outputs

| Command | Output | Requirements |
| --- | --- | --- |
| `python -m src.experiments.protocol` | `artifacts/portfolio_v1/` | Dataset and primary configuration |
| `python -m src.experiments.intercept_study` | `artifacts/portfolio_intercept_v1/` | Dataset and follow-up configuration |
| `python -m src.experiments.interpretation` | Primary run's `interpretation/` | Completed primary run and saved fold models |
| `python -m src.experiments.validation` | Validation, stability and uncertainty records in both run directories | Both completed runs |
| `python tools/build_portfolio_reports.py` | `reports/generated/` | Both validated runs, interpretation, bootstrap and optional reporting dependencies |
| `python tools/verify_portfolio_reports.py` | Layout checks and previews in the report directory | Generated PDFs and PowerPoint |

The primary run records the experiment plan, effective identity, inner candidate
scores, partition indices, fold metrics, development predictions, fit diagnostics
and selected settings. Each fitted bundle contains its pipeline, calibrator,
partitions, recipe and seed. These bundles are fold-specific research estimators,
not a full-data deployment model. Load only trusted joblib files.

Validation reloads saved models, recomputes predictions and metrics, checks
disjoint roles and historical exclusions, and quantifies conditional uncertainty.
The bootstrap resamples original development rows with consistent multiplicities
across recipes and repeats; it excludes retraining and experiment-design uncertainty.
`--skip-bootstrap` rechecks saved outputs without repeating that bootstrap.

## Independent output directories

Experiment directories are immutable by default. `--resume` accepts only matching
code, data, environment and configuration identities. A changed experiment needs
a new output directory. For example:

```text
python -m src.experiments.protocol --output-dir artifacts/reproduction_primary
python -m src.experiments.intercept_study --output-dir artifacts/reproduction_intercept
python -m src.experiments.interpretation --run-dir artifacts/reproduction_primary
python -m src.experiments.validation --run-dir artifacts/reproduction_primary --followup-dir artifacts/reproduction_intercept
python tools/build_portfolio_reports.py --run-dir artifacts/reproduction_primary --followup-dir artifacts/reproduction_intercept --report-dir reports/reproduction
python tools/verify_portfolio_reports.py --report-dir reports/reproduction
```

Report generation reads saved outputs and writes only its report directory. It
does not update the README, technical report, frozen evidence or recorded
presentations in `reports/portfolio_v1/`. It emits matching slide PDF/PowerPoint,
a technical-report Markdown/PDF, theory PDF and chart assets. Verification adds
rendered previews and checks for text/shape bounds.

## Environment and reproducibility scope

CPython 3.12 is used by CI. [Core dependencies](../requirements.txt) and optional
[reporting dependencies](../requirements-reporting.txt) pin direct packages.
The [recorded inventory](evidence/portfolio_v1/environment-freeze.txt) includes
transitive/reporting packages from the measured Windows environment; it is not
a universal lock file. Runtime depends on hardware and parallelism.

Saved-model roundtrips and independent refits are distinct checks. The recorded
refits matched selected settings, outer metrics and predictions at stated
tolerances. Inner L1 search scores showed small variation; its cause was not
isolated and exact bitwise search reproducibility is not established. Explanation
diagnostics concern a fixed held-out development fold and were not repeated in
the full-refit check. See [validation evidence](evidence/portfolio_v1/README.md).
