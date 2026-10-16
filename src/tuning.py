"""Fresh hyperparameter searches on training folds; threshold selection on validation."""
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from src.data import build_preprocessor
from src.models import MODELS


def search_spec(name):
    """Explicit fresh estimators and broad spaces, independent of prior model settings."""
    if name == "glassbox":
        baseline = dict(solver="liblinear", max_iter=1000, tol=0.001)
        space = dict(C=np.logspace(-5, 0, 11).tolist(), penalty=["l1", "l2"],
                     class_weight=[None, "balanced", {0: 1, 1: 4}, {0: 1, 1: 8}])
    elif name == "blackbox":
        baseline = dict(objective="binary:logistic", eval_metric="aucpr", tree_method="hist", n_jobs=1,
                        balance_multiplier=None)
        space = dict(n_estimators=[100, 250, 500, 800], max_depth=[2, 3, 4, 6],
                     learning_rate=[0.01, 0.03, 0.06, 0.1], min_child_weight=[1, 5, 10, 20],
                     subsample=[0.6, 0.8, 1.0], colsample_bytree=[0.5, 0.75, 1.0],
                     reg_alpha=[0.0, 0.1, 1.0, 10.0], reg_lambda=[1.0, 5.0, 20.0, 100.0],
                     gamma=[0.0, 0.5, 2.0], balance_multiplier=[None, 0.25, 0.5, 1.0, 2.0])
    else:
        raise ValueError(f"Unknown model: {name}")
    return baseline, {f"classifier__{key}": values for key, values in space.items()}


def tune_model(name, config, X, y, *, n_iter=None, cv=None, n_jobs=None):
    """Accept training data only. All learned transformations and weights fit per fold."""
    settings = config.get("tuning", {})
    n_iter = n_iter if n_iter is not None else settings.get("n_iter", 40)
    cv = cv if cv is not None else settings.get("cv_folds", 5)
    n_jobs = n_jobs if n_jobs is not None else settings.get("n_jobs", 2)
    labels = np.asarray(y)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Tuning requires binary labels containing both classes")
    if not isinstance(cv, int) or cv < 2 or np.bincount(labels.astype(int)).min() < cv:
        raise ValueError("Each class must have at least cv_folds training examples")
    if not isinstance(n_iter, int) or n_iter < 1:
        raise ValueError("Search iterations must be a positive integer")
    if not isinstance(n_jobs, int) or n_jobs == 0:
        raise ValueError("Search workers must be a nonzero integer")
    effective = deepcopy(config)
    effective.setdefault("tuning", {}).update(n_iter=n_iter, cv_folds=cv, n_jobs=n_jobs)
    baseline, space = search_spec(name)
    effective["models"][name] = baseline
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor(effective, X.columns, scale=name == "glassbox")),
        ("classifier", MODELS[name](effective)._estimator()),
    ])
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=config["random_seed"])
    search = RandomizedSearchCV(
        pipeline, space, n_iter=n_iter, scoring={"average_precision": "average_precision", "roc_auc": "roc_auc"},
        refit="average_precision", cv=splitter, random_state=config["random_seed"],
        n_jobs=n_jobs, pre_dispatch=n_jobs if n_jobs > 0 else "2*n_jobs", error_score="raise",
        return_train_score=True, verbose=1,
    ).fit(X, y)
    params = {key.removeprefix("classifier__"): value for key, value in search.best_params_.items()}
    effective["models"][name].update(params)
    results = pd.DataFrame(search.cv_results_).sort_values("rank_test_average_precision")
    counts = np.bincount(labels.astype(int), minlength=2)
    report = dict(
        model=name, scoring="average_precision", cv_folds=cv, candidates=len(results),
        random_seed=config["random_seed"], training_class_counts={"0": int(counts[0]), "1": int(counts[1])},
        no_skill_average_precision=float(labels.mean()), best_params=params,
        best_cv_average_precision=float(search.best_score_),
        best_cv_average_precision_std=float(search.cv_results_["std_test_average_precision"][search.best_index_]),
        best_cv_roc_auc=float(search.cv_results_["mean_test_roc_auc"][search.best_index_]),
        search_space={key.removeprefix("classifier__"): value for key, value in space.items()},
    )
    return effective, report, results


def select_threshold(y, probabilities, beta=2.0):
    """Maximize validation F-beta; break ties toward the higher threshold."""
    y, p = np.asarray(y), np.asarray(probabilities, dtype=float)
    if y.shape != p.shape or set(np.unique(y)) != {0, 1}:
        raise ValueError("Threshold selection requires aligned labels with both classes")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() or not np.isfinite(beta) or beta <= 0:
        raise ValueError("Threshold selection requires probabilities and a positive finite beta")
    precision, recall, thresholds = precision_recall_curve(y, p)
    numerator = (1 + beta ** 2) * precision[:-1] * recall[:-1]
    denominator = beta ** 2 * precision[:-1] + recall[:-1]
    scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    best = int(np.flatnonzero(scores == scores.max())[-1])
    return dict(threshold=float(thresholds[best]), objective=f"F{beta:g}", beta=float(beta),
                score=float(scores[best]), precision=float(precision[best]), recall=float(recall[best]),
                rows=len(y), positives=int(y.sum()))
