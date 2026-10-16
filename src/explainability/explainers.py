"""Native, SHAP, and LIME explanations in the fitted feature space."""
import numpy as np


def explain_model(model, X, background, methods=None):
    cfg = model.config["explainability"]
    methods = cfg["methods"] if methods is None else methods
    if set(methods) - {"native", "shap", "lime"}:
        raise ValueError("Explanation methods must be native, shap, or lime")
    if X.empty or background.empty or cfg["sample_size"] < 1 or cfg["background_size"] < 1:
        raise ValueError("Explanations require nonempty samples and positive sample sizes")
    seed = model.config["random_seed"]
    sample = X.sample(min(len(X), cfg["sample_size"]), random_state=seed)
    baseline = background.sample(min(len(background), cfg["background_size"]), random_state=seed)
    preprocessor = model.pipeline_.named_steps["preprocessor"]
    classifier = model.pipeline_.named_steps["classifier"]
    values, reference = preprocessor.transform(sample), preprocessor.transform(baseline)
    names = preprocessor.get_feature_names_out().tolist()
    result = {
        "feature_names": names, "row_indices": sample.index.tolist(),
        "scope": "base classifier in transformed feature space; calibration is excluded",
        "predicted_default_probability": model.predict_proba(sample)[:, 1].tolist(),
        "base_default_probability": classifier.predict_proba(values)[:, 1].tolist(),
    }
    if "native" in methods:
        if model.name == "glassbox":
            weights = classifier.coef_[0]
            result["native"] = {
                "coefficients": weights.tolist(), "intercept": float(classifier.intercept_[0]),
                "local_log_odds_contributions": (values * weights).tolist(),
                "units": "numerical coefficients per training standard deviation; categorical coefficients per indicator",
            }
        else:
            weights = classifier.feature_importances_
            result["native"] = {"feature_importances": weights.tolist(), "importance_type": "gain"}
        order = np.argsort(np.abs(weights))[::-1][:cfg["top_features"]]
        result["native"]["top_features"] = [{"feature": names[i], "weight": float(weights[i])} for i in order]
    if "shap" in methods:
        import shap
        if model.name == "glassbox":
            explainer = shap.LinearExplainer(classifier, reference)
        else:
            explainer = shap.TreeExplainer(classifier, data=reference,
                feature_perturbation="interventional", model_output="raw")
        explanation = explainer(values)
        result["shap"] = {
            "values": np.asarray(explanation.values).tolist(),
            "base_values": np.asarray(explanation.base_values).tolist(),
            "mean_absolute_importance": np.abs(explanation.values).mean(axis=0).tolist(),
            "units": "base model log odds",
        }
    if "lime" in methods:
        from lime.lime_tabular import LimeTabularExplainer
        explainer = LimeTabularExplainer(reference, feature_names=names,
            class_names=["non-default", "default"], mode="classification",
            random_state=seed, discretize_continuous=False)
        result["lime"] = [
            {"weights": explainer.explain_instance(row, classifier.predict_proba,
                labels=(1,), num_features=cfg["top_features"],
                num_samples=cfg["lime_num_samples"]).as_list(label=1)} for row in values
        ]
    return result
