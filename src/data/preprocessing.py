"""All learned preprocessing stays inside the estimator pipeline."""
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(config, columns, scale):
    cfg = config["preprocessing"]
    categorical = [c for c in columns if c in config["data"]["categorical_features"]]
    numeric = [c for c in columns if c not in categorical]
    steps = [("imputer", SimpleImputer(strategy=cfg["numeric_imputation"],
              add_indicator=cfg["add_missing_indicators"], keep_empty_features=True))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    transformer = ColumnTransformer([
        ("numeric", Pipeline(steps), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ], remainder="drop", verbose_feature_names_out=True)
    return Pipeline([("columns", transformer), ("variance", VarianceThreshold(cfg["variance_threshold"]))])
