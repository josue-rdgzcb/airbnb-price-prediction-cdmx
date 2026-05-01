import pandas as pd
import numpy as np

def validate_features(df, features_list, target):
    """
    Validate and profile multiple features in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the features and target variable.
    features_list : list of str
        List of feature column names to validate.
    target : str
        Target column name used for correlation (numeric) or median by category (categorical).

    Behavior
    --------
    - Checks if all requested features exist in the DataFrame.
    - Prints a basic profile for each feature (dtype, counts, nulls, unique values).
    - For numeric features:
        * Prints descriptive statistics (mean, median, mode, std, percentiles, skew, kurtosis).
        * Detects outliers using the 1.5*IQR rule.
        * Computes Pearson correlation with the target if numeric.
    - For categorical features:
        * Prints frequency distribution (counts and percentages).
        * Prints median target value by category if target is numeric.
    - Includes try/except blocks to handle errors gracefully without stopping execution.
    """

    # =============== Validate existence of columns ===============
    missing_features = set(features_list) - set(df.columns)
    if missing_features:
        raise ValueError(f"Features not found in dataframe: {missing_features}")

    # =============== Iterate through each feature ===============
    for col in features_list:
        print(f"\n========== Validating: {col} ==========")

        # =============== Basic Profile ===============
        try:
            profile = pd.DataFrame({
                "data_type": [df[col].dtype],
                "count": [df[col].count()],
                "non_null": [df[col].notnull().sum()],
                "unique": [df[col].nunique(dropna=True)],
                "null": [df[col].isnull().sum()],
                "null_ratio_%": [(df[col].isnull().mean() * 100).round(2)]
            })
            print("\n--- Basic Profile ---")
            print(profile)
        except Exception as e:
            print(f"Error computing basic profile for {col}: {e}")
            continue

        # =============== Numeric Features ===============
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            try:
                stats_dict = {
                    "mean": df[col].mean(),
                    "median": df[col].median(),
                    "mode": df[col].mode().iloc[0] if not df[col].mode().empty else np.nan,
                    "std": df[col].std(),
                    "min": df[col].min(),
                    "p25": df[col].quantile(0.25),
                    "p50": df[col].quantile(0.50),
                    "p75": df[col].quantile(0.75),
                    "max": df[col].max(),
                    "skew": df[col].skew(),
                    "kurt": df[col].kurt()
                }

                # Outlier detection using 1.5*IQR rule
                try:
                    q1, q3 = df[col].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
                    outliers = ((df[col] < lower) | (df[col] > upper)).sum()
                    stats_dict["outliers_count"] = outliers
                    stats_dict["outliers_pct"] = round(outliers / df[col].count() * 100, 2)
                except Exception as e:
                    print(f"Error computing outliers for {col}: {e}")

                # Pearson correlation with target
                try:
                    if target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
                        stats_dict["pearson_r"] = df[[col, target]].corr().iloc[0,1]
                except Exception as e:
                    print(f"Error computing correlation for {col}: {e}")

                stats_df = pd.DataFrame(stats_dict, index=[0]).T
                print("\n--- Statistical Summary ---")
                print(stats_df)
            except Exception as e:
                print(f"Error computing numeric summary for {col}: {e}")

        # =============== Categorical Features ===============
        else:
            try:
                freq = df[col].value_counts(dropna=False)
                freq_pct = df[col].value_counts(normalize=True, dropna=False) * 100
                freq_table = pd.DataFrame({"count": freq, "pct": freq_pct.round(2)})
                print("\n--- Frequency distribution ---")
                print(freq_table.head(10))
            except Exception as e:
                print(f"Error computing frequency distribution for {col}: {e}")

            try:
                if target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
                    target_median = df.groupby(col)[target].median().sort_values(ascending=False)
                    target_median_df = target_median.reset_index()
                    target_median_df.columns = [col, "median_target"]

                    print("\n--- Median target by category ---")
                    print(target_median_df)
            except Exception as e:
                print(f"Error computing median target by category for {col}: {e}")

