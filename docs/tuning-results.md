# Fresh imbalance-aware tuning

Run date: 2026-10-16. Data: 3,000 rows, 218 features, 183 defaults (6.1%).
Seed 42; shared stratified 1,800/600/600 train/validation/test partitions.
Training contains 110 defaults, validation 36, and test 37.

Each model searched 40 candidates over five shuffled stratified training folds.
Selection maximized mean average precision (AP); all preprocessing and class
weights were learned within the training folds. No saved model or previous
hyperparameters initialized the searches. The full search spaces and all
fold scores are in `artifacts/tuned/<model>/tuning.json` and `cv_results.csv`.

The decision threshold maximized validation F2, favoring recall. Calibration
was disabled for these runs. Hyperparameters and thresholds were fixed before
test evaluation. Validation threshold scores are selection scores, not unbiased
performance estimates. The existing baseline uses a 0.5 threshold, so changes
in recall/precision reflect both parameter and threshold changes.

| Test metric | Glassbox baseline | Glassbox tuned | Blackbox baseline | Blackbox tuned |
| --- | ---: | ---: | ---: | ---: |
| ROC-AUC | 0.672723 | 0.703087 | 0.756229 | 0.701743 |
| AP (PR-AUC) | 0.122159 | 0.129473 | 0.193370 | 0.115353 |
| Precision | 0.181818 | 0.118644 | 0.250000 | 0.100372 |
| Recall | 0.108108 | 0.378378 | 0.027027 | 0.729730 |
| F1 | 0.135593 | 0.180645 | 0.048780 | 0.176471 |
| Brier score | 0.076486 | 0.223772 | 0.056510 | 0.056426 |
| Threshold | 0.500000 | 0.566104 | 0.500000 | 0.056098 |

The no-skill AP reference for the test partition is 37/600 = 0.061667.

Glassbox ranking improves slightly, but its balanced class weights substantially
worsen probability calibration: Brier score rises from 0.076486 to 0.223772.
Blackbox recall improves at the selected threshold, while AP and ROC-AUC fall
below the original baseline. These selected configurations are retained as
experiments; the baseline configuration remains available. Neither result
establishes a general improvement across ranking, recall and probability quality.

The completed glassbox search uses `C` from 0.00001 through 1, a 1,000-iteration
solver limit and tolerance 0.001. Weakly regularized L1 fits beyond this range
were prohibitively slow. This bound is a computational search restriction;
the test outcomes did not determine it.

## Glassbox

Winning CV AP: 0.137024 (fold SD 0.035823); CV ROC-AUC: 0.616460.

Selected classifier configuration:

```yaml
C: 0.1
class_weight: balanced
max_iter: 1000
penalty: l1
solver: liblinear
tol: 0.001
```

Validation threshold selection: {"threshold": 0.5661036429874251, "objective": "F2", "beta": 2.0, "score": 0.37254901960784315, "precision": 0.17117117117117117, "recall": 0.5277777777777778, "rows": 600, "positives": 36}

Test confusion matrix `[[TN, FP], [FN, TP]]`: `[[459, 104], [23, 14]]`.

Refit the selected parameters without repeating the search:

```powershell
.\.venv\Scripts\python.exe main.py --model glassbox --config config/tuned_glassbox.yaml
```

## Blackbox

Winning CV AP: 0.156934 (fold SD 0.017634); CV ROC-AUC: 0.712682.

Selected classifier configuration:

```yaml
balance_multiplier: null
colsample_bytree: 1.0
eval_metric: aucpr
gamma: 2.0
learning_rate: 0.03
max_depth: 4
min_child_weight: 10
n_estimators: 250
n_jobs: 1
objective: binary:logistic
reg_alpha: 0.0
reg_lambda: 100.0
subsample: 1.0
tree_method: hist
```

Validation threshold selection: {"threshold": 0.056097958236932755, "objective": "F2", "beta": 2.0, "score": 0.3994845360824743, "precision": 0.12704918032786885, "recall": 0.8611111111111112, "rows": 600, "positives": 36}

Test confusion matrix `[[TN, FP], [FN, TP]]`: `[[321, 242], [10, 27]]`.

Refit the selected parameters without repeating the search:

```powershell
.\.venv\Scripts\python.exe main.py --model blackbox --config config/tuned_blackbox.yaml
```

## Interpretation limits

These are finite randomized searches, not claims of globally optimal parameters.
Only 37 test defaults make recall estimates uncertain. F2 favors recall but
does not encode lending costs or capacity constraints. More recall can mean
many more false positives. Class weighting changes probability estimation;
inspect Brier scores and calibration curves before using scores as PDs.
The existing baseline test results were already available before this work;
the test partition was excluded from the implemented selection procedure,
but it is not a newly collected external validation cohort.

Reproduce the fresh searches with the `--tune` commands in the README.
Saved classifiers, predictions, runtime versions, full CV results and
resolved effective configurations are in `artifacts/tuned/<model>/`.
