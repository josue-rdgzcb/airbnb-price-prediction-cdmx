 
import pandas as pd


def apply_binning(df, binning_config, suffix="_binned"):
    """
    Apply manual binning to selected variables.

    Parameters
    ----------
    df : pandas.DataFrame
    binning_config : dict

        Example:
        {
            "minimum_nights": {
                "bins": [0, 7, 30, float("inf")],
                "labels": [
                    "short_stay",
                    "medium_stay",
                    "long_term"
                ]
            }
        }

    Returns
    -------
    pandas.DataFrame
    """
    df = df.copy()

    for col, config in binning_config.items():

        bins = config["bins"]
        labels = config["labels"]

        df[f"{col}{suffix}"] = pd.cut(
            df[col],
            bins=bins,
            labels=labels
        )

    return df
