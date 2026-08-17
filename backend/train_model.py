
from pathlib import Path
import tarfile
import urllib.request

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from scipy.stats import randint

from app.ml_pipeline import ClusterSimilarity, column_ratio, ratio_name


def load_housing_data():
    housing_tar = Path("datasets/housing.tgz")
    if not housing_tar.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, housing_tar)
        with tarfile.open(housing_tar) as housing_data:
            housing_data.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))


def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler(),
    )


def main():
    housing = load_housing_data()

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    strat_train_set, strat_test_set = train_test_split(
        housing, test_size=0.2, stratify=housing["income_cat"], random_state=42
    )
    for split in (strat_train_set, strat_test_set):
        split.drop("income_cat", axis=1, inplace=True)

    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"),
    )

    log_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(np.log, feature_names_out="one-to-one"),
        StandardScaler(),
    )

    cluster_similarity = ClusterSimilarity(n_clusters=10, gamma=0.1, random_state=42)

    default_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

    preprocessing = ColumnTransformer(
        [
            ("bedrooms_ratio", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
            ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
            ("people_per_house", ratio_pipeline(), ["population", "households"]),
            (
                "log",
                log_pipeline,
                ["total_bedrooms", "total_rooms", "population", "households", "median_income"],
            ),
            ("geo", cluster_similarity, ["latitude", "longitude"]),
            ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
        ],
        remainder=default_pipeline,
    )

    full_pipeline = Pipeline(
        [
            ("preprocessing", preprocessing),
            ("random_forest", RandomForestRegressor(random_state=42)),
        ]
    )

    param_distribs = {
        "preprocessing__geo__n_clusters": randint(low=3, high=50),
        "random_forest__max_features": randint(low=2, high=20),
    }

    print("Running RandomizedSearchCV (this can take a few minutes)...")
    rnd_search = RandomizedSearchCV(
        full_pipeline,
        param_distributions=param_distribs,
        n_iter=10,
        cv=3,
        scoring="neg_root_mean_squared_error",
        random_state=42,
    )
    rnd_search.fit(housing, housing_labels)

    final_model = rnd_search.best_estimator_

    x_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()
    from sklearn.metrics import root_mean_squared_error

    final_predictions = final_model.predict(x_test)
    final_rmse = root_mean_squared_error(y_test, final_predictions)
    print(f"Final test RMSE: {final_rmse:,.0f}")

    out_dir = Path(__file__).parent / "app" / "model"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "house_price_calculator_ml.pkl"
    joblib.dump(final_model, out_path)
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    main()
