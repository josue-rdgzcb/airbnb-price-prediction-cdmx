import pandas as pd

from src.settings.transform_config import (BINNING_FEATURES)


def apply_binning(
    df,
    suffix="_binned"
):
    """
    Apply statistical binning to selected variables.

    Parameters
    ----------
    df : pandas.DataFrame

    suffix : str, default="_binned"

    Returns
    -------
    pandas.DataFrame
    """

    for col, config in BINNING_FEATURES.items():

        if not BINNING_FEATURES:
            return df

        method = config.get("method", "qcut")
        bins = config.get("bins", 4)

        new_col = f"{col}{suffix}"

        # Quantile binning
        if method == "qcut":

            df[new_col] = pd.qcut(
                df[col],
                q=bins,
                duplicates="drop"
            )

        # Equal-width binning
        elif method == "cut":

            df[new_col] = pd.cut(
                df[col],
                bins=bins
            )

        else:

            raise ValueError(
                f"Invalid method '{method}' "
                f"for column '{col}'"
            )
        
        # Drop original feature
        df.drop(columns=col, inplace=True)

    return df