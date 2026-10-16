"""Gradient boosted trees; import XGBoost only when needed."""
from .base import CreditModel


class BlackboxModel(CreditModel):
    name = "blackbox"

    def _estimator(self):
        if "balance_multiplier" in self.config["models"][self.name]:
            from .weighted_xgboost import FoldWeightedXGBClassifier
            return FoldWeightedXGBClassifier(random_state=self.config["random_seed"],
                                             **self.config["models"][self.name])
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=self.config["random_seed"], **self.config["models"][self.name])
