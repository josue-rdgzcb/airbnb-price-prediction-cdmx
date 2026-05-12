from sklearn.impute import SimpleImputer

from src.settings.transform_config import (
    MEAN_IMPUTE_FEATURES,
    MEDIAN_IMPUTE_FEATURES,
    MOST_FREQUENT_IMPUTE_FEATURES
)


def apply_imputation(df):

    imputers = {

        "mean": {
            "features": MEAN_IMPUTE_FEATURES,
            "imputer": SimpleImputer(strategy="mean")
        },

        "median": {
            "features": MEDIAN_IMPUTE_FEATURES,
            "imputer": SimpleImputer(strategy="median")
        },

        "most_frequent": {
            "features": MOST_FREQUENT_IMPUTE_FEATURES,
            "imputer": SimpleImputer(strategy="most_frequent")
        }
    }

    for _, config in imputers.items():

        features = config["features"]
        imputer = config["imputer"]

        if not features:
            continue

        df[features] = imputer.fit_transform(df[features])

    return df

