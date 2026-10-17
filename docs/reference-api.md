# Single-split reference API

`main.py` and the shared model classes support the historical baseline and its
parameter-search interface. They do not implement the nested family-selection
protocol used for the [current results](../README.md#results). Their test reporting
concerns the already inspected historical partition and is exploratory.

## CLI

```text
python main.py --help
python main.py --model glassbox --output-dir artifacts/reference --skip-explain
python main.py --model blackbox --output-dir artifacts/reference --skip-explain
```

The CLI trains a model, evaluates the shared train/validation/test split, and
optionally explains a test sample. `--skip-explain` disables explanations only.
`--data` overrides the CSV path; `--config` selects a YAML file; `--output-dir`
sets the artifact root. CLI paths are relative to the working directory; paths
inside YAML are relative to the configuration file. Reusing a reference output
directory replaces its files, so distinct runs should use separate directories.

`--tune` invokes the retained randomized AP search on training folds. Candidate
count, CV folds and parallelism are controlled by `--n-iter`, `--cv-folds` and
`--search-jobs`. Positive-class XGBoost weights are computed separately within
each fit. The selected model fits the training partition; validation selects an
F2 threshold. If calibration is enabled, validation is split into disjoint
calibration and threshold subsets. This interface cannot provide a new final
assessment of the repeatedly inspected historical test.

The [historical selected configurations](historical-reference.md) can be supplied
through `--config` to reconstruct that reference without a fresh parameter search.

## Model interface

```python
from src.config import load_config
from src.data import CreditDataLoader
from src.models import GlassboxModel, BlackboxModel

config = load_config("config/config.yaml")
X, y = CreditDataLoader(config).load()
splits = CreditDataLoader(config).split(X, y)
X_train, y_train = splits["train"]
X_val, y_val = splits["validation"]
X_test, y_test = splits["test"]

model = GlassboxModel(config)  # BlackboxModel has the same interface.
model.fit(X_train, y_train, X_val, y_val)
probabilities = model.predict_proba(X_test)  # columns: non-default, default
labels = model.predict(X_test)
metrics = model.evaluate(X_test, y_test)
explanations = model.explain(X_test, background=X_train, methods=["native", "shap"])
```

The loader normalizes categorical types and converts infinite numerical values
to missing values. Invalid numerical text fails explicitly. Imputation, numerical
scaling for logistic regression, encoding and constant removal fit on training
data. Unseen categories are handled by the fitted encoder. Explanation calls
transform data without refitting preprocessing. Optional sigmoid/isotonic
calibration maps base scores using validation data without refitting the classifier.

## Outputs and metric definitions

The artifact root contains a directory per model with `model.joblib`, effective
`config.yaml`, `environment.json`, `metrics.json`, per-split prediction CSVs and
optional `explanations.json`. A tuning run also records `tuning.json` and
`cv_results.csv`. Load only trusted joblib files.

- AP is average precision, not trapezoidal precision-recall area; Gini is
  `2 × ROC-AUC − 1`.
- KS is the maximum absolute difference between empirical default/non-default
  score distributions, handling ties consistently.
- Brier/log loss and reliability curves assess probabilities separately from
  ranking. Quantile reliability bins and expected calibration error have different
  aggregation rules and are not interchangeable uncertainty estimates.
- Confusion matrices use `[[TN, FP], [FN, TP]]`. Undefined precision/recall use zero.
- PSI uses training-reference quantile bins, collapsed duplicate edges, infinite
  outer boundaries and normalized additive smoothing. Random-split PSI may reflect
  overfitting or sampling variation; it does not establish temporal drift.

Native logistic terms and SHAP contributions describe the uncalibrated log-odds
margin. Numerical logistic coefficients use standardized inputs; missing indicators
are scaled with numerical fields. Tree gain is unsigned training split importance.
LIME targets base probabilities in transformed coordinates and may perturb
one-hot/correlated fields into unrealistic profiles. The reference explanation
budget defaults to 20 observations; the canonical study uses the broader scope
documented in the model card.
