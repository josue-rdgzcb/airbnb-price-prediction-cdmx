import pandas as pd
import numpy as np
from IPython.display import display


# ============= BASIC PROFILE =======================
def basic_profile(df, features_list):
    """
    Computes basic descriptive statistics for a list of features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    features_list : list
        List of feature names to analyze.

    Returns
    -------
    pd.DataFrame
        Summary table with data type, non-null count, unique values,
        null count and null percentage.
    """

    # Validate features
    missing_features = set(features_list) - set(df.columns)
    if missing_features:
        raise ValueError(f"Features not found in dataframe: {missing_features}")

    data = df[features_list]

    summary = pd.DataFrame({
        "data_type": data.dtypes,
        "non_null": data.notnull().sum(),
        "unique": data.nunique(dropna=True),
        "null": data.isnull().sum(),
        "null_ratio_%": (data.isnull().mean() * 100).round(2)
    })

    print("======== Basic Profile ========\n")
    print(summary.sort_index())

    #return summary.sort_index()

# ============= ADVANCED PROFILE =======================
def advanced_profile(df, vars_to_check):
    """
    Generate descriptive statistics tables for categorical and numerical variables.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    vars_to_check : list
        List of column names to analyze.
    
    Returns
    -------
    cat_summary : pandas.DataFrame or None
        Summary table for categorical variables (statistics as rows, variables as columns).
        Returns None if no categorical variables are found.
    num_summary : pandas.DataFrame or None
        Summary table for numerical variables (statistics as rows, variables as columns).
        Returns None if no numerical variables are found.
    
    Notes
    -----
    - Categorical statistics include: dtype, count, non_null, unique, null count, null ratio (%),
      is_binary, top category, top frequency, top ratio, balance ratio, relative cardinality.
    - Numerical statistics include: dtype, count, non_null, null count, null ratio (%), unique,
      mean, std, coefficient of variation, IQR, range, min, p1, p25, p50, p75, p90, p99, max,
      number of outliers, percentage of outliers, skewness, kurtosis, percentage of zeros,
      percentage of negatives, is_binary.
    """
    
    cat_stats = {}
    num_stats = {}

    for var in vars_to_check:
        series = df[var]
        total = series.shape[0]
        nulls = series.isnull().sum()
        non_nulls = total - nulls
        unique = series.nunique()
        
        # Handle categorical variables
        if series.dtype == "object" or series.dtype.name == "category" or series.dtype == "str":
            mode_val = series.mode().iloc[0] if not series.mode().empty else None
            mode_freq = series.value_counts().iloc[0]
            mode_ratio = mode_freq / total
            vc = series.value_counts()
            balance_ratio = vc.iloc[0] / vc.iloc[-1] if unique > 1 else None
            relative_cardinality = unique / total
            
            cat_stats[var] = {
                "dtype": series.dtype,
                "count": total,
                "non_null": non_nulls,
                "unique": unique,
                "null": nulls,
                "null_ratio_%": round(nulls/total*100, 2),
                "is_binary": unique == 2,
                "top_category": mode_val,
                "top_freq": mode_freq,
                "top_ratio": round(mode_ratio, 3),
                "balance_ratio": round(balance_ratio, 3) if balance_ratio else None,
                "relative_cardinality": round(relative_cardinality, 3)
            }
        
        # Handle numerical variables
        else:
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            cv = series.std() / series.mean() if series.mean() != 0 else np.nan
            
            # Outliers defined by Tukey's rule (1.5 * IQR)
            outliers = ((series < (q1 - 1.5*iqr)) | (series > (q3 + 1.5*iqr))).sum()
            outlier_ratio = outliers / non_nulls if non_nulls > 0 else np.nan
            
            num_stats[var] = {
                "dtype": series.dtype,
                "count": total,
                "non_null": non_nulls,
                "null": nulls,
                "null_ratio_%": round(nulls/total*100, 2),
                "unique": unique,
                "mean": round(series.mean(), 3),
                "std": round(series.std(), 3),
                "coef_var": round(cv, 3),
                "IQR": round(iqr, 3),
                "range": series.max() - series.min(),
                "min": series.min(),
                "p1": series.quantile(0.01),
                "p25": q1,
                "p50": series.median(),
                "p75": q3,
                "p90": series.quantile(0.90),
                "p99": series.quantile(0.99),
                "max": series.max(),
                "outliers": outliers,
                "outliers_%": round(outlier_ratio*100, 2),
                "skew": round(series.skew(), 3),
                "kurtosis": round(series.kurtosis(), 3),
                "zeros_%": round((series == 0).sum()/total*100, 2),
                "negatives_%": round((series < 0).sum()/total*100, 2),
                "is_binary": unique == 2
            }

    # Convert to DataFrames (variables as columns, statistics as rows)
    cat_summary = pd.DataFrame(cat_stats) if cat_stats else None
    num_summary = pd.DataFrame(num_stats) if num_stats else None
    
    # Print with titles and display formatting
    if cat_summary is not None:
        print("===== Categorical Statistics =====\n")
        print(cat_summary)
    if num_summary is not None:
        print("\n===== Numerical Statistics =====\n")
        print(num_summary)
