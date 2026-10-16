"""Regularized logistic regression with scaled numerical features."""
from sklearn.linear_model import LogisticRegression
from .base import CreditModel


class GlassboxModel(CreditModel):
    name = "glassbox"

    def _estimator(self):
        return LogisticRegression(random_state=self.config["random_seed"], **self.config["models"][self.name])
