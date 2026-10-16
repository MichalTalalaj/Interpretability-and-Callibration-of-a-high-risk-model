"""Common interface for glassbox and blackbox estimators."""
from .glassbox import GlassboxModel
from .blackbox import BlackboxModel

MODELS = {"glassbox": GlassboxModel, "blackbox": BlackboxModel}
