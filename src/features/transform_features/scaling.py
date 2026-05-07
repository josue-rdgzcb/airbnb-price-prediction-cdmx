 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)


def build_scaler_pipeline(
    robust_features=None,
    minmax_features=None,
    standard_features=None
):
    """
    Build ColumnTransformer for scaling.

    Returns
    -------
    ColumnTransformer
    """

    robust_features = robust_features or []
    minmax_features = minmax_features or []
    standard_features = standard_features or []

    scaler = ColumnTransformer(
        transformers=[

            (
                "robust",
                RobustScaler(),
                robust_features
            ),

            (
                "minmax",
                MinMaxScaler(),
                minmax_features
            ),

            (
                "standard",
                StandardScaler(),
                standard_features
            )

        ],
        remainder="passthrough"
    )

    return scaler