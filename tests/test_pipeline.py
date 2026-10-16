"""Behavioral regression tests for leakage, persistence, metrics and explanations."""
import copy
from pathlib import Path
import tempfile
import unittest
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from src.config import load_config
from src.data import CreditDataLoader
from src.evaluation import classification_metrics, population_stability_index
from src.models import MODELS


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(Path(__file__).resolve().parents[1] / "config/config.yaml")
        self.config["data"]["categorical_features"] = ["category"]
        self.config["models"]["blackbox"]["n_estimators"] = 8
        self.config["explainability"].update(sample_size=2, background_size=20, lime_num_samples=100)
        rng = np.random.default_rng(71)
        self.X = pd.DataFrame({"amount": rng.normal(size=120), "category": ["a", "b"] * 60,
                               "empty": np.full(120, np.nan)})
        self.y = pd.Series((self.X.amount + rng.normal(size=120) > 0).astype(int))

    def test_splits_are_disjoint_and_reproducible(self):
        loader = CreditDataLoader(self.config)
        splits = loader.split(self.X, self.y)
        again = loader.split(self.X, self.y)
        indices = [set(x.index) for x, _ in splits.values()]
        self.assertEqual(set.union(*indices), set(self.X.index))
        for i, left in enumerate(indices):
            for right in indices[i + 1:]:
                self.assertFalse(left & right)
        for key in splits:
            self.assertTrue(splits[key][0].equals(again[key][0]))

    def test_metric_definitions_and_psi_edges(self):
        cfg = self.config["evaluation"]
        metrics = classification_metrics([0, 0, 1, 1], [0, 0.1, 0.9, 1], cfg)
        for key in ["roc_auc", "pr_auc", "gini", "ks_statistic"]:
            self.assertEqual(metrics[key], 1)
        self.assertEqual(population_stability_index([0.1, 0.2], [0.1, 0.2]), 0)
        self.assertGreater(population_stability_index([0.1] * 10, [0.9] * 10), 1)
        self.assertTrue(np.isfinite(population_stability_index([0.2, 0.4, 0.6], [0, 1])))
        with self.assertRaises(ValueError):
            classification_metrics([0, 1], [0.1, np.nan], cfg)

    def test_both_models_preserve_training_state_and_roundtrip(self):
        for name, cls in MODELS.items():
            with self.subTest(model=name):
                model = cls(copy.deepcopy(self.config)).fit(self.X, self.y)
                probe = self.X.iloc[:3].copy()
                probe["category"] = "unseen"
                probe["amount"] = [np.nan, 1e6, -1e6]
                pre = model.pipeline_.named_steps["preprocessor"]
                before = joblib.hash(pre)
                p = model.predict_proba(probe)
                self.assertEqual(p.shape, (3, 2))
                np.testing.assert_allclose(p.sum(axis=1), 1)
                explanation = model.explain(probe, self.X, ["native"])
                self.assertEqual(joblib.hash(pre), before)
                self.assertEqual(len(explanation["feature_names"]), pre.transform(probe).shape[1])
                if name == "glassbox":
                    native = explanation["native"]
                    reconstructed = expit(np.asarray(native["local_log_odds_contributions"]).sum(axis=1) + native["intercept"])
                    np.testing.assert_allclose(reconstructed, explanation["base_default_probability"])
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / "model.joblib"
                    joblib.dump(model, path)
                    np.testing.assert_allclose(joblib.load(path).predict_proba(probe), p)

    def test_shap_additivity_and_lime_for_both_models(self):
        for name, cls in MODELS.items():
            with self.subTest(model=name):
                model = cls(copy.deepcopy(self.config)).fit(self.X, self.y)
                report = model.explain(self.X.iloc[:3], self.X, ["shap", "lime"])
                reconstructed = expit(np.asarray(report["shap"]["values"]).sum(axis=1) + report["shap"]["base_values"])
                np.testing.assert_allclose(reconstructed, report["base_default_probability"], atol=1e-6)
                self.assertEqual(len(report["lime"]), 2)

    def test_calibration_does_not_refit_preprocessing(self):
        for method in ["sigmoid", "isotonic"]:
            cfg = copy.deepcopy(self.config)
            cfg["calibration"]["method"] = method
            model = MODELS["glassbox"](cfg).fit(self.X.iloc[:80], self.y.iloc[:80], self.X.iloc[80:], self.y.iloc[80:])
            plain = MODELS["glassbox"](self.config).fit(self.X.iloc[:80], self.y.iloc[:80])
            self.assertEqual(joblib.hash(model.pipeline_), joblib.hash(plain.pipeline_))
            self.assertTrue(np.isfinite(model.predict_proba(self.X)).all())

    def test_loader_rejects_target_leakage(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.csv"
            self.X.assign(default=self.y).to_csv(path, index=False)
            cfg = copy.deepcopy(self.config)
            cfg["data"].update(path=str(path), features=["amount", "default"])
            with self.assertRaises(ValueError):
                CreditDataLoader(cfg).load()

    def test_holdout_values_never_change_imputation_statistics(self):
        X_train = self.X.iloc[:80].copy()
        X_train.loc[X_train.index[:10], "amount"] = np.nan
        X_holdout = self.X.iloc[80:].copy()
        X_holdout["amount"] = 1e12
        cfg = copy.deepcopy(self.config)
        cfg["calibration"]["method"] = "sigmoid"
        model = MODELS["glassbox"](cfg).fit(X_train, self.y.iloc[:80], X_holdout, self.y.iloc[80:])
        numeric = model.pipeline_.named_steps["preprocessor"].named_steps["columns"].named_transformers_["numeric"]
        self.assertAlmostEqual(numeric.named_steps["imputer"].statistics_[0], X_train["amount"].median())


if __name__ == "__main__":
    unittest.main()
