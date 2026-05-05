
import pandas as pd
import numpy as np

# ===========================================================
# =================== NUMERIC DIAGNOSTICS ===================
# ===========================================================
def numeric_diagnostics(df, target="log_price"):
    """
    Perform numeric diagnostics on DataFrame columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing numeric features and target.
    target : str, default="log_price"
        Name of the target variable used for correlation, excluded from the final diagnostics.

    Returns
    -------
    pandas.DataFrame
        Table with diagnostics for each numeric feature, including:
        - Percentage of nulls
        - Skewness and kurtosis
        - Standard deviation, min, max
        - Outlier percentage (IQR method)
        - Correlation with target
    """
    results = []

    # Select numeric columns, excluding the target
    num_cols = df.select_dtypes(include=[np.number]).columns.drop(target)

    for col in num_cols:
        series = df[col].dropna()

        # Outlier detection using IQR
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        outliers = ((series < q1 - 1.5*iqr) | (series > q3 + 1.5*iqr)).sum()
        outliers_pct = outliers / len(series)

        # Collect diagnostics for each feature
        results.append({
            "feature": col,
            "nulls_%": df[col].isna().mean(),
            "skew": series.skew(),
            "kurtosis": series.kurt(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "outliers_%": outliers_pct,
            "corr_with_target": df[col].corr(df[target])
        })

    return pd.DataFrame(results).set_index("feature").reset_index()


# =================== HEURISTICS: NULL TREATMENT ===================
def suggest_null_treatment(row):
    """
    Suggest null value treatment based on feature statistics.

    Parameters
    ----------
    row : pandas.Series
        Row from diagnostics table containing feature statistics.

    Returns
    -------
    str
        Suggested null treatment strategy:
        - "no_impute" if no nulls
        - "median" if skewed or with outliers
        - "mean" otherwise
    """
    nulls = row["nulls_%"]
    skew = row["skew"]
    outliers = row["outliers_%"]

    if nulls == 0:
        return "no_impute"

    if outliers > 0.05 or abs(skew) > 1:
        return "median"
    else:
        return "mean"


# =================== HEURISTICS: TRANSFORMATION ===================
def suggest_transform_treatment(row):
    """
    Suggest a transformation treatment for a numeric feature based on skewness and outlier percentage.

    Parameters
    ----------
    row : pandas.Series
        A row from the diagnostics DataFrame containing at least:
        - "skew": skewness of the feature
        - "outliers_%": percentage of outliers detected

    Returns
    -------
    str
        Recommended transformation:
        - "log" for strong skewness (> 2)
        - "sqrt" for moderate skewness (> 1)
        - "winsor_strong" for heavy outliers (> 10%)
        - "winsor" for moderate outliers (> 5%)
        - "no_transform" if no adjustment is needed
    """

    skew = abs(row["skew"])
    outliers = row["outliers_%"]

    # Priority: strong skewness → log transform
    if skew > 2:
        return "log"

    # Moderate skewness → square root transform
    if skew > 1:
        return "sqrt"

    # Heavy outliers (without strong skew) → strong winsorization
    if outliers > 0.10:
        return "winsor_strong"

    # Moderate outliers → winsorization
    if outliers > 0.05:
        return "winsor"

    # Default: no transformation needed
    return "no_transform"


# =================== HEURISTICS: BINNING ===================
def suggest_binning(row):
    """
    Suggest whether a feature could benefit from binning.

    Parameters
    ----------
    row : pandas.Series
        Row from diagnostics table containing feature statistics.

    Returns
    -------
    str
        Binning suggestion:
        - "binning_candidate" with reason
        - "no_binning" otherwise
    """
    skew = abs(row["skew"])
    outliers = row["outliers_%"]
    corr = abs(row["corr_with_target"])
    std = row["std"]

    # Candidates due to non-linearity (low correlation + skew)
    if corr < 0.1 and skew > 1:
        return "binning_candidate (non_linear)"

    # Candidates due to extreme values
    if outliers > 0.10:
        return "binning_candidate (extreme_values)"

    # Candidates due to low variance / discreteness
    if std < 1:
        return "binning_candidate (low_variance/discrete)"

    return "no_binning"

# =================== HEURISTICS: SCALING ===================
def suggest_scaling(row):
    """
    Suggest a scaling method based on outlier percentage and skewness.

    Parameters
    ----------
    row : pandas.Series
        Must contain 'outliers_%' and 'skew'.

    Returns
    -------
    str
        - "robust" if many outliers (> 5%)
        - "standard" if distribution is near normal (skew < 0.5)
        - "minmax" otherwise
    """

    outliers = row["outliers_%"]
    skew = abs(row["skew"])

    # Many outliers → robust scaling
    if outliers > 0.05:
        return "robust"

    # Distribution close to normal → standard scaling
    if skew < 0.5:
        return "standard"

    # Default → min-max scaling
    return "minmax"

# =================== HEURISTICS: SIGNAL ===================
def suggest_signal(row):
    """
    Suggest signal strength based on correlation with target.

    Parameters
    ----------
    row : pandas.Series
        Must contain 'corr_with_target'.

    Returns
    -------
    str
        - "low_signal" if correlation < 0.05
        - "weak_signal" if correlation < 0.15
        - "useful_signal" otherwise
    """

    corr = abs(row["corr_with_target"])

    # Very low correlation → low signal
    if corr < 0.05:
        return "low_signal"

    # Weak correlation → weak signal
    elif corr < 0.15:
        return "weak_signal"

    # Otherwise → useful signal
    else:
        return "useful_signal"



# =================== FINAL PIPELINE ===================
def build_numeric_diagnostics(df, target="log_price"):
    """
    Build complete numeric diagnostics pipeline.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing numeric features and target.
    target : str, default="log_price"
        Name of the target variable used for correlation, excluded from the final diagnostics.

    Returns
    -------
    pandas.DataFrame
        Diagnostics table with:
        - Null treatment suggestion
        - Transformation suggestion
        - Binning suggestion
    """
    diag = numeric_diagnostics(df, target)

    diag["signal_suggestion"] = diag.apply(suggest_signal, axis=1)
    diag["null_treatment_suggestion"] = diag.apply(suggest_null_treatment, axis=1)
    diag["transform_suggestion"] = diag.apply(suggest_transform_treatment, axis=1)
    diag["binning_suggestion"] = diag.apply(suggest_binning, axis=1)
    diag["scaling_suggestion"] = diag.apply(suggest_scaling, axis=1)

    return diag
