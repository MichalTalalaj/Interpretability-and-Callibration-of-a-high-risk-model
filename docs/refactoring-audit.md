# Historical refactoring audit and migration decisions

This note records the earlier refactor and the subsequent removal of obsolete
root-level files from `main`. The original README identified `OLD_` files as
outdated. The files listed below were superseded by the supported implementation
in `src/` and `main.py`, and by the current presentation in `reports/portfolio_v1/`.
Their last retained versions are available in commit
[`3a494a4`](https://github.com/MichalTalalaj/Interpretability-and-Callibration-of-a-high-risk-model/tree/3a494a4ee5c03f67e79904b2a4398804c007e15a).
The current dataset, frozen experiment evidence and local baseline artifacts are
preserved. The development protocol is documented at the top of the README.

| Original files | Decision and replacement |
| --- | --- |
| `OLD_run_Regression_analysis.py`, `OLD_run_XGBoost_analysis.py` | Removed obsolete copies of notebook exports. |
| `OLD_regression_analysis.ipynb`, `OLD_best_regression_model.pkl`, `OLD_wyniki_analizy.joblib` | Removed archived notebook and stale artifacts; the latter was only two bytes. |
| `run_Regression_analysis.py`, `run_XGBoost_analysis.py` | Merged into `src/` and `main.py`; duplicated transforms, metrics, calibration, plotting and cached case selection are replaced. |
| `run_Regression.ps1`, `run_XGBoost.ps1` | Removed runners that recreated and deleted environments, and in one case deleted the model output. Use the CLI. |
| `XGBoost_model_worse_version.ipynb` | Removed explicitly superseded experiment. |
| `best_regression_model.pkl`, `XGBoost_model_pipeline.pkl`, `XGBoost_model_ready.pkl`, `wyniki_analizy.joblib` | Removed stale generated models and explanation cache tied to script-local classes and old feature spaces; regenerate under ignored `artifacts/`. |
| `regression_analysis.ipynb`, `XGBoost_model.ipynb` | Removed historical research notebooks at the user's request; original versions remain in Git history. |
| `Presentation.pdf`, `bibliography.txt` | Removed historical presentation and research bibliography at the user's request. |
| `zbiór_10.csv` | Kept as the dataset used by the supported pipeline. |

## Problems addressed

- Training required unpickling old estimators or precomputed SHAP/case dictionaries.
- The regression feature-name helper could refit transforms after model training.
- One-hot encoding and the blackbox WoE step lived outside the fitted pipeline.
- Train/test partitions differed between approaches; blackbox splits were not stratified.
- Some threshold, cost and rating exploration used test data. Undefined notebook
  variables such as `final_pd` also made the exported blackbox script non-runnable.
- Broad warning suppression, commented duplicate experiments, blocking plot windows,
  unpinned dependencies and machine-relative paths obscured reproducibility.

## Deliberate baseline changes

This is a new comparable baseline, not bit-for-bit reproduction of notebook results.
Industry `pkdKod` uses unsupervised one-hot encoding instead of target-derived WoE.
Train-only imputation, constant removal, and numerical scaling (logistic only) stay
inside the same fitted pipeline. Correlation-based selection/sign flipping and
correlation-derived XGBoost monotonicity constraints were removed: empirical
association alone does not establish a justified risk constraint. Explicit model
parameters can be supplied through YAML, including domain-approved XGBoost settings.
The blackbox baseline uses a configured fixed tree count, without early stopping.
Calibration is optional, fitted only on validation; the arbitrary 4% PD shift was
removed. Threshold is configured in advance rather than optimized on the test set.
Automated tuning, cost simulations, rating generation, PDP/ICE plots and beta
calibration from the exploratory notebooks are outside the new baseline pipeline.

## Exact supported structure

See the repository map in `README.md`. `src` is the importable Python package and
all subpackages have `__init__.py`; running from the checkout needs no sys.path hack.
