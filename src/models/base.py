"""Shared interface and optional calibration on a separate validation set."""
from abc import ABC, abstractmethod
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from src.data import build_preprocessor
from src.evaluation import classification_metrics


class CreditModel(ABC):
    name: str

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def _estimator(self):
        """Build a fresh classifier from central configuration."""

    def fit(self, X, y, X_validation=None, y_validation=None):
        method = self.config["calibration"]["method"]
        if method not in {"none", "sigmoid", "isotonic"}:
            raise ValueError("Calibration method must be none, sigmoid, or isotonic")
        if method != "none" and (X_validation is None or y_validation is None):
            raise ValueError("Calibration requires a separate labeled validation set")
        self.pipeline_ = Pipeline([
            ("preprocessor", build_preprocessor(self.config, X.columns, scale=self.name == "glassbox")),
            ("classifier", self._estimator()),
        ]).fit(X, y)
        self.classes_ = self.pipeline_.classes_
        self.feature_names_in_ = list(X.columns)
        self.calibrator_ = None
        if method != "none":
            p = self.pipeline_.predict_proba(X_validation)[:, 1]
            if method == "isotonic":
                self.calibrator_ = IsotonicRegression(out_of_bounds="clip").fit(p, y_validation)
            else:
                cfg = self.config["calibration"]
                self.calibrator_ = LogisticRegression(C=cfg["sigmoid_C"], max_iter=cfg["max_iter"],
                    random_state=self.config["random_seed"]).fit(self._logits(p), y_validation)
        return self

    @staticmethod
    def _logits(p):
        eps = np.finfo(float).eps
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def predict_proba(self, X):
        check_is_fitted(self, "pipeline_")
        p = self.pipeline_.predict_proba(X)[:, 1]
        if isinstance(self.calibrator_, IsotonicRegression):
            p = self.calibrator_.predict(p)
        elif self.calibrator_ is not None:
            p = self.calibrator_.predict_proba(self._logits(p))[:, 1]
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.config["evaluation"]["threshold"]).astype(int)

    def evaluate(self, X, y, reference=None):
        return classification_metrics(y, self.predict_proba(X)[:, 1], self.config["evaluation"], reference)

    def explain(self, X, background, methods=None):
        check_is_fitted(self, "pipeline_")
        from src.explainability import explain_model
        return explain_model(self, X, background, methods)
