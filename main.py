"""Train, evaluate, and explain one credit model with reproducible artifacts."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import joblib
import pandas as pd
import yaml
from src.config import load_config
from src.data import CreditDataLoader
from src.models import MODELS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config/config.yaml")
    parser.add_argument("--data", type=Path, help="Override CSV path (relative to current working directory)")
    parser.add_argument("--output-dir", type=Path, help="Override artifact root")
    parser.add_argument("--explain", nargs="+", choices=["native", "shap", "lime"], help="Override configured methods")
    parser.add_argument("--skip-explain", action="store_true")
    parser.add_argument("--tune", action="store_true", help="Search fresh parameters on training folds and select a validation F-beta threshold")
    parser.add_argument("--n-iter", type=int, help="Candidates per tuning search")
    parser.add_argument("--cv-folds", type=int, help="Stratified training folds for tuning")
    parser.add_argument("--search-jobs", type=int, help="Parallel tuning workers")
    args = parser.parse_args(argv)
    if not args.tune and any(value is not None for value in [args.n_iter, args.cv_folds, args.search_jobs]):
        parser.error("--n-iter, --cv-folds and --search-jobs require --tune")
    config = load_config(args.config)
    if args.data:
        config["data"]["path"] = str(args.data.resolve())
    if args.output_dir:
        config["output"]["directory"] = str(args.output_dir.resolve())
    if args.explain:
        config["explainability"]["methods"] = args.explain
    loader = CreditDataLoader(config)
    X, y = loader.load()
    splits = loader.split(X, y)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["validation"]
    tuning_report = None
    if args.tune:
        from src.tuning import tune_model
        config, tuning_report, cv_results = tune_model(
            args.model, config, X_train, y_train, n_iter=args.n_iter, cv=args.cv_folds, n_jobs=args.search_jobs)
    X_cal, y_cal = X_val, y_val
    X_threshold, y_threshold = X_val, y_val
    if args.tune and config["calibration"]["method"] != "none":
        from sklearn.model_selection import train_test_split
        X_cal, X_threshold, y_cal, y_threshold = train_test_split(
            X_val, y_val, test_size=0.5, stratify=y_val, random_state=config["random_seed"])
    model = MODELS[args.model](config).fit(X_train, y_train, X_cal, y_cal)
    if args.tune:
        from src.tuning import select_threshold
        tuning_report["threshold_selection"] = select_threshold(
            y_threshold, model.predict_proba(X_threshold)[:, 1],
            beta=config.get("tuning", {}).get("threshold_beta", 2.0))
        config["evaluation"]["threshold"] = tuning_report["threshold_selection"]["threshold"]
        tuning_report["calibration_indices"] = X_cal.index.tolist() if config["calibration"]["method"] != "none" else []
        tuning_report["threshold_indices"] = X_threshold.index.tolist()
        tuning_report["training_indices"] = X_train.index.tolist()
    reference = model.predict_proba(X_train)[:, 1]
    metrics = {name: model.evaluate(features, labels, reference) for name, (features, labels) in splits.items()}
    output = Path(config["output"]["directory"]) / args.model
    output.mkdir(parents=True, exist_ok=True)
    if tuning_report is not None:
        (output / "tuning.json").write_text(json.dumps(tuning_report, indent=2, allow_nan=False), encoding="utf-8")
        cv_results.to_csv(output / "cv_results.csv", index=False)
    else:
        for filename in ["tuning.json", "cv_results.csv"]:
            (output / filename).unlink(missing_ok=True)
    joblib.dump(model, output / "model.joblib")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    (output / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    versions = {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "scipy", "joblib", "PyYAML"]}
    if args.model == "blackbox":
        versions["xgboost"] = importlib.metadata.version("xgboost")
    if not args.skip_explain:
        for method in config["explainability"]["methods"]:
            if method in {"shap", "lime"}:
                versions[method] = importlib.metadata.version(method)
    (output / "environment.json").write_text(json.dumps({"python": platform.python_version(), "packages": versions}, indent=2), encoding="utf-8")
    for name, (features, labels) in splits.items():
        pd.DataFrame({"row_index": features.index, "actual": labels.to_numpy(),
                      "probability": model.predict_proba(features)[:, 1],
                      "prediction": model.predict(features)}).to_csv(output / f"predictions_{name}.csv", index=False)
    explanation_path = output / "explanations.json"
    if explanation_path.exists():
        explanation_path.unlink()
    if not args.skip_explain:
        explanations = model.explain(splits["test"][0], X_train)
        explanation_path.write_text(json.dumps(explanations, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"model": args.model, "test_metrics": metrics["test"], "artifacts": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
