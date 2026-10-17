# Historical experiment reference

The current assessment is repeated nested development cross-validation, described
in the [technical report](portfolio-audit.md). Earlier work used one stratified
1,800/600/600 train/validation/test split (seed 42). It contained 110, 36 and 37
defaults respectively. That test partition was repeatedly inspected; its results
are exploratory and its indices are excluded from the current protocol.

## Earlier results and their interpretation

| Historical test metric | Logistic baseline | Logistic selected | XGBoost baseline | XGBoost selected |
| --- | ---: | ---: | ---: | ---: |
| AP | 0.122159 | 0.129473 | 0.193370 | 0.115353 |
| ROC-AUC | 0.672723 | 0.703087 | 0.756229 | 0.701743 |
| Brier loss | 0.076486 | 0.223772 | 0.056510 | 0.056426 |
| Recall | 0.108108 | 0.378378 | 0.027027 | 0.729730 |
| Threshold | 0.500000 | 0.566104 | 0.500000 | 0.056098 |

The earlier searches evaluated 40 candidates in five stratified training folds,
selected mean AP, and chose validation F2 thresholds without calibration.
Weighting worsened logistic probability quality; the selected tree model traded
ranking performance for recall at a lower threshold. These configurations are
references, not recommended operating models. The current protocol re-evaluates
frozen configurations on the same development partitions as its other ablations,
so those comparisons do not depend on historical test scores.

The fitting budgets differ: the current outer base fit uses 960 observations,
with separate calibration and threshold subsets. Its scores cannot be presented
as direct improvements over the historical test results.

## Retained reference configuration

- [Baseline loading/model configuration](../config/config.yaml).
- [Selected logistic configuration](../config/tuned_glassbox.yaml): balanced L1,
  `C=0.1`, liblinear; mean search AP 0.137024.
- [Selected XGBoost configuration](../config/tuned_blackbox.yaml): 250 trees,
  learning rate 0.03, child weight 10, gamma 2 and L2 100; mean search AP 0.156934.
- [Frozen diagnosis](evidence/portfolio_v1/historical_diagnosis.json): complete
  earlier metrics, search coverage and fitted tree-structure diagnostics.
- [Single-split reference API](reference-api.md): loading and model interfaces
  retained for reference reconstruction.

The implementation uses fold-fitted imputation, encoding and selection. It
replaces the original notebook's target-derived encoding, arbitrary probability
shift and test-informed decision exploration. Correlation-derived sign constraints
are not imposed without a documented domain justification. Original notebooks,
script exports and model binaries remain available in Git history.
