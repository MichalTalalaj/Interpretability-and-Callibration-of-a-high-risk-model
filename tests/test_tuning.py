"""Tuning behavior: fresh searches, fold-local weights, and isolated holdouts."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import fbeta_score
import yaml
from main import main
from src.config import load_config
from src.models.weighted_xgboost import FoldWeightedXGBClassifier
from src.tuning import select_threshold, tune_model


class TuningTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(Path(__file__).resolve().parents[1] / "config/config.yaml")
        self.config["data"]["categorical_features"] = ["category"]
        rng = np.random.default_rng(21)
        self.X = pd.DataFrame({"amount": rng.normal(size=160), "category": ["a", "b"] * 80})
        self.y = pd.Series(([0] * 7 + [1]) * 20)

    def test_xgboost_recomputes_weights_on_every_fit_and_clones(self):
        model = FoldWeightedXGBClassifier(balance_multiplier=0.5, n_estimators=2, n_jobs=1)
        model.fit(self.X[["amount"]], self.y)
        self.assertEqual(model.get_params()["scale_pos_weight"], 3.5)
        self.assertNotIn("balance_multiplier", model.get_xgb_params())
        cloned = clone(model)
        labels = pd.Series([0, 0, 0, 1] * 40)
        cloned.fit(self.X[["amount"]], labels)
        self.assertEqual(cloned.get_params()["scale_pos_weight"], 1.5)
        self.assertEqual(model.get_params()["scale_pos_weight"], 3.5)
        unweighted = FoldWeightedXGBClassifier(n_estimators=2, n_jobs=1).fit(self.X[["amount"]], self.y)
        self.assertEqual(unweighted.get_params()["scale_pos_weight"], 1)

    def test_search_ignores_old_parameters_and_is_reproducible(self):
        self.config["models"]["glassbox"] = {"C": -1, "solver": "invalid"}
        before = copy.deepcopy(self.config)
        with contextlib.redirect_stdout(io.StringIO()):
            effective, report, results = tune_model("glassbox", self.config, self.X, self.y, n_iter=2, cv=2, n_jobs=1)
            again, second, _ = tune_model("glassbox", self.config, self.X, self.y, n_iter=2, cv=2, n_jobs=1)
        self.assertEqual(self.config, before)
        self.assertEqual(effective, again)
        self.assertEqual(report, second)
        self.assertEqual(effective["models"]["glassbox"]["solver"], "liblinear")
        self.assertEqual(len(results), 2)
        self.assertEqual(report["no_skill_average_precision"], 0.125)
        self.assertIn("classifier__class_weight", results.iloc[0]["params"])

    def test_threshold_maximizes_f2_including_probability_ties(self):
        y = np.array([0, 1, 0, 1, 0, 0, 1, 0])
        p = np.array([0.01, 0.1, 0.1, 0.2, 0.3, 0.3, 0.4, 0.8])
        result = select_threshold(y, p)
        candidates = [(fbeta_score(y, p >= threshold, beta=2), threshold) for threshold in np.unique(p)]
        expected_score, expected_threshold = max(candidates)
        self.assertAlmostEqual(result["score"], expected_score)
        self.assertEqual(result["threshold"], expected_threshold)
        with self.assertRaises(ValueError):
            select_threshold(y, p, beta=0)
        with self.assertRaises(ValueError):
            tune_model("glassbox", self.config, self.X, self.y, cv=21)

    def test_calibration_threshold_and_test_use_disjoint_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cfg = copy.deepcopy(self.config)
            self.X.assign(default=self.y).to_csv(root / "input.csv", index=False)
            cfg["data"]["path"] = str(root / "input.csv")
            cfg["output"]["directory"] = str(root / "output")
            cfg["calibration"]["method"] = "sigmoid"
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--model", "glassbox", "--config", str(config_path), "--tune", "--n-iter", "2",
                      "--cv-folds", "2", "--search-jobs", "1", "--skip-explain"])
            output = root / "output/glassbox"
            report = json.loads((output / "tuning.json").read_text())
            test_indices = pd.read_csv(output / "predictions_test.csv").row_index.tolist()
            sets = [set(report[key]) for key in ["training_indices", "calibration_indices", "threshold_indices"]]
            sets.append(set(test_indices))
            for i, left in enumerate(sets):
                for right in sets[i + 1:]:
                    self.assertFalse(left & right)
            self.assertEqual(set.union(*sets), set(self.X.index))
            saved = joblib.load(output / "model.joblib")
            self.assertEqual(saved.config["evaluation"]["threshold"], report["threshold_selection"]["threshold"])


if __name__ == "__main__":
    unittest.main()
