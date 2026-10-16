"""Validate raw data and split before learning any transformations."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class CreditDataLoader:
    def __init__(self, config):
        self.config = config

    def load(self):
        cfg = self.config["data"]
        frame = pd.read_csv(cfg["path"], encoding=cfg["encoding"], sep=cfg["separator"])
        target = cfg["target"]
        if target not in frame or frame[target].isna().any() or set(frame[target].unique()) != {0, 1}:
            raise ValueError(f"Target {target!r} must contain both 0 and 1 without missing labels")
        features = cfg["features"]
        if features is None:
            features = [c for c in frame if c != target and c not in cfg["drop_columns"]]
        if not features or len(set(features)) != len(features) or target in features:
            raise ValueError("Features must be unique, nonempty, and exclude the target")
        if set(features) & set(cfg["drop_columns"]):
            raise ValueError("Explicit features include a dropped column")
        missing = set(features) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing configured features: {sorted(missing)}")
        X = frame.loc[:, features].copy()
        for column in X:
            if column in cfg["categorical_features"]:
                X[column] = X[column].map(lambda v: str(v) if pd.notna(v) else np.nan)
            else:
                X[column] = pd.to_numeric(X[column], errors="raise")
        return X.replace([np.inf, -np.inf], np.nan), frame[target].astype(int)

    def split(self, X, y):
        cfg = self.config["data"]
        test, validation = cfg["test_size"], cfg["validation_size"]
        if not (0 < test < 1 and 0 < validation < 1 and test + validation < 1):
            raise ValueError("Test and validation fractions must be positive and sum to less than one")
        seed = self.config["random_seed"]
        X_rest, X_test, y_rest, y_test = train_test_split(X, y, test_size=test, random_state=seed, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(
            X_rest, y_rest, test_size=validation / (1 - test), random_state=seed, stratify=y_rest)
        return {"train": (X_train, y_train), "validation": (X_val, y_val), "test": (X_test, y_test)}
