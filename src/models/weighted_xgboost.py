"""Compute the positive-class weight from each fit's labels, including CV folds."""
import numpy as np
from xgboost import XGBClassifier


class FoldWeightedXGBClassifier(XGBClassifier):
    def __init__(self, balance_multiplier=None, **kwargs):
        self.balance_multiplier = balance_multiplier
        super().__init__(**kwargs)

    def get_xgb_params(self):
        params = super().get_xgb_params()
        params.pop("balance_multiplier", None)
        return params

    def fit(self, X, y, **kwargs):
        counts = np.bincount(np.asarray(y, dtype=int), minlength=2)
        if not counts[0] or not counts[1]:
            raise ValueError("Class weighting requires both binary classes")
        weight = 1.0 if self.balance_multiplier is None else float(self.balance_multiplier) * counts[0] / counts[1]
        if weight <= 0 or not np.isfinite(weight):
            raise ValueError("Positive-class weight must be finite and positive")
        self.set_params(scale_pos_weight=weight)
        return super().fit(X, y, **kwargs)
