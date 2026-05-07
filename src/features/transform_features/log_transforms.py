 
import numpy as np


def apply_log_transformations(df, columns, suffix="_log"):
    """
    Apply log1p transformation to selected columns.

    Parameters
    ----------
    df : pandas.DataFrame
    columns : list
        Columns to transform.
    suffix : str, default="_log"
        Suffix for transformed columns.

    Returns
    -------
    pandas.DataFrame
    """
    df = df.copy()

    for col in columns:
        df[f"{col}{suffix}"] = np.log1p(df[col])

    return df