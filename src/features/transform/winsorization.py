import pandas as pd

from src.settings.transform_config import (WINSOR_FEATURES)


def apply_winsorization(
    df,
    suffix="_winsor"
):
    """
    Apply upper-tail winsorization.

    Parameters
    ----------
    df : pandas.DataFrame

    suffix : str, default="_winsor"

    Returns
    -------
    pandas.DataFrame
    """

    for col, config in WINSOR_FEATURES.items():

        if not WINSOR_FEATURES:
            return df

        lower_q = config.get("lower", 0.00)
        upper_q = config.get("upper", 0.99)

        lower = df[col].quantile(lower_q)
        upper = df[col].quantile(upper_q)

        df[f"{col}{suffix}"] = df[col].clip(
            lower,
            upper
        )

        # Drop original feature
        df.drop(columns=col, inplace=True)

    return df