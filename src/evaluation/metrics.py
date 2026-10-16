"""Metric definitions with explicit probability and PSI conventions."""
import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, log_loss, precision_score, recall_score, roc_auc_score,
)
from sklearn.calibration import calibration_curve


def _probabilities(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Probabilities must be a finite, nonempty vector")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("Probabilities must be within [0, 1]")
    return values


def population_stability_index(reference, current, bins=10, epsilon=1e-6):
    """Score PSI using reference-only quantiles and normalized additive smoothing.

    Infinite outer edges include out-of-reference-range scores. Constant reference
    scores use fixed probability bins so shifted current populations are detectable.
    """
    reference, current = _probabilities(reference), _probabilities(current)
    if bins < 2 or epsilon <= 0:
        raise ValueError("PSI requires at least two bins and positive smoothing")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        edges = np.linspace(0, 1, bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, bins=edges)[0] / len(reference)
    cur = np.histogram(current, bins=edges)[0] / len(current)
    ref = (ref + epsilon) / (1 + epsilon * len(ref))
    cur = (cur + epsilon) / (1 + epsilon * len(cur))
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def classification_metrics(y_true, probabilities, config, reference=None):
    p = _probabilities(probabilities)
    y = np.asarray(y_true)
    if y.shape != p.shape or set(np.unique(y)) != {0, 1}:
        raise ValueError("Metrics require aligned binary labels containing both classes")
    prediction = (p >= config["threshold"]).astype(int)
    auc = float(roc_auc_score(y, p))
    bucket = np.minimum((p * config["calibration_bins"]).astype(int), config["calibration_bins"] - 1)
    ece = sum(np.mean(bucket == b) * abs(y[bucket == b].mean() - p[bucket == b].mean())
              for b in np.unique(bucket))
    observed, predicted = calibration_curve(y, p, n_bins=config["calibration_bins"], strategy="quantile")
    result = {
        "roc_auc": auc, "pr_auc": float(average_precision_score(y, p)), "gini": 2 * auc - 1,
        "ks_statistic": float(ks_2samp(p[y == 0], p[y == 1]).statistic),
        "accuracy": float(accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])), "ece": float(ece),
        "threshold": config["threshold"], "rows": len(y), "default_rate": float(y.mean()),
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
        "calibration_curve": {"mean_prediction": predicted.tolist(), "default_rate": observed.tolist()},
    }
    if reference is not None:
        result["psi"] = population_stability_index(reference, p, config["psi_bins"], config["psi_epsilon"])
    return result
