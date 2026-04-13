import numpy as np
import pandas as pd
from scipy import stats

# ========================= CATEGORICAL VARIABLES ==================================

def analyze_categorical_vars(df, vars, target, rows=None):
    """
    Perform a basic profiling and statistical analysis of categorical variables
    against a target variable.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to analyze.
    vars : list
        List of categorical column names to analyze.
    target : str
        Name of the target variable (e.g., 'log_price').
    rows : int, optional (default=None)
        Number of rows to display in summary tables.

    Returns
    -------
    None
        Prints to console the results of the basic profile and analysis
        (frequency distribution, medians, differences, spread statistics,
        and correlation ratio).
    """


    # ========= Build a basic profile table for all variables ============
    summary_vars = []

    for col in vars:
        summary_vars.append({
            "variable": col,
            "dtype": df[col].dtype,                      # data type
            "non_null": df[col].notnull().sum(),         # non-null count
            "unique": df[col].nunique(dropna=True),      # unique values (excluding NaN)
            "null": df[col].isnull().sum(),              # null count
            "null_ratio (%)": df[col].isnull().mean() * 100,  # null ratio in %
        })

    basic_profile = pd.DataFrame(summary_vars)

    print(f"\n{'='*65}")
    print(f"BASIC PROFILE OF VARIABLE(S)")
    print(f"{'='*65}")
    print(basic_profile)

    # ==================== Analysis by variable =========================
    # Iterate over each categorical variable
    for col in vars:
        print(f"\n{'='*65}")
        print(f"ANALYSIS OF {col}")
        print(f"{'='*65}\n")
        
        # ============= Frequency distribution ==========================
        counts = df[col].value_counts(dropna=False)
        proportions = df[col].value_counts(normalize=True, dropna=False) * 100
        freq_dist = pd.DataFrame({
            "count": counts,
            "proportion (%)": proportions
        })
        print(f"{'='*12} Frequency Distribution {'='*12}\n")
        print(freq_dist.head(rows))

        # ============= Median target by categorical variable ============
        median_values = df.groupby(col)[target].median().sort_values(ascending=False)
        print(f"\n{'='*10} Median target by category {'='*10}\n")
        print(median_values.head(rows))

        # ============= Median differences ===============================
        try:
            n_categories = len(median_values)

            if n_categories == 2:
                values = median_values.values
                median_diff = abs(values[0] - values[1])
                real_diff = np.exp(median_diff) - 1
                print(f"\n={'='*12} Median Difference {'='*12}")
                print(f"\nMedian Difference (binary): {median_diff:.4f}")
                print(f"Equivalent real difference: {real_diff*100:.2f}%")

            elif 3 <= n_categories <= 5:
                baseline = df[col].value_counts(dropna=False).idxmax()
                baseline_median = median_values.loc[baseline]

                print(f"\n{'='*15} Median Difference {'='*15}")
                print(f"\nBaseline category (most frequent): {baseline}")
                diffs = median_values - baseline_median
                real_diffs = np.exp(diffs) - 1
                print("\nDifferences vs baseline (log scale):")
                print(diffs)
                print("\nDifferences vs baseline (real %):")
                print((real_diffs*100).round(2))

            else:
                print("\nMedian Difference: Exceeds number of categories (not computed)")

        except Exception as e:
            print(f"\nMedian Difference: Error computing ({e})")

        # =============== Target mean/median + spread =======================
        mean_values = df.groupby(col)[target].mean()
        std_values = df.groupby(col)[target].std()
        iqr_values = df.groupby(col)[target].apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))

        print(f"\n{'='*10} Target mean/median + spread {'='*10}\n")
        summary_stats = pd.DataFrame({
            "mean": mean_values,
            "median": median_values,
            "std": std_values,
            "IQR": iqr_values
        })
        print(summary_stats.sort_values(by="median", ascending=False).head(rows))

        # ============== Correlation ratio (η²) ===============================
        try:
            cat = pd.Categorical(df[col])
            y = df[target].values
            y_avg = np.mean(y)
            ss_between = sum([
                len(y[cat == level]) * (np.mean(y[cat == level]) - y_avg) ** 2
                for level in cat.categories
            ])
            ss_total = sum((y - y_avg) ** 2)
            eta_sq = ss_between / ss_total if ss_total > 0 else np.nan
            print(f"\n{'='*13} Correlation ratio (η²) {'='*13}")
            print(f"η²: {eta_sq:.4f}")
        except Exception as e:
            print(f"\nCorrelation ratio: Error ({e})")

# ========================= NUMERICAL VARIABLES ==================================

def analyze_numeric_vars(df, vars, target):
    """
    Perform a basic profiling and statistical analysis of numeric variables
    against a target variable.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to analyze.
    vars : list
        List of numeric column names to analyze.
    target : str
        Name of the target variable (e.g., 'log_price').

    Returns
    -------
    None
        Prints to console the results of the basic profile and analysis
        (central tendency, dispersion, outliers via IQR, skewness, kurtosis,
        correlations with target).
    """

    # ========= Build a basic profile table for all variables ============
    summary_vars = []
    for col in vars:
        summary_vars.append({
            "variable": col,
            "dtype": df[col].dtype,
            "non_null": df[col].notnull().sum(),
            "unique": df[col].nunique(dropna=True),
            "null": df[col].isnull().sum(),
            "null_ratio (%)": df[col].isnull().mean() * 100,
        })

    basic_profile = pd.DataFrame(summary_vars)

    print(f"\n{'='*65}")
    print(f"BASIC PROFILE OF NUMERIC VARIABLE(S)")
    print(f"{'='*65}")
    print(basic_profile)

    # ==================== Analysis by variable =========================
    for col in vars:
        print(f"\n{'='*65}")
        print(f"ANALYSIS OF {col}")
        print(f"{'='*65}\n")

        series = df[col].dropna()

        # ============= Central Tendency ==========================
        mean_val = series.mean()
        median_val = series.median()
        mode_val = series.mode().iloc[0] if not series.mode().empty else np.nan
        print(f"{'='*12} Central Tendency {'='*12}")
        print(f"Mean:   {mean_val:.4f}")
        print(f"Median: {median_val:.4f}")
        print(f"Mode:   {mode_val:.4f}")

        # ============= Dispersion ==========================
        std_val = series.std()
        percentiles = np.percentile(series, [5,25,50,75,90,95])
        min_val, max_val = series.min(), series.max()
        range_val = max_val - min_val
        var_val = series.var()
        cv_val = std_val / mean_val if mean_val != 0 else np.nan

        dispersion_dict = {
            "std": std_val,
            "min": min_val,
            "p5": percentiles[0],
            "p25": percentiles[1],
            "p50": percentiles[2],
            "p75": percentiles[3],
            "p90": percentiles[4],
            "p95": percentiles[5],
            "max": max_val,
            "range": range_val,
            "variance": var_val,
            "coef_var": cv_val
        }

        dispersion_table = pd.DataFrame.from_dict(dispersion_dict, orient="index", columns=["value"])
        print(f"\n{'='*15} Dispersion {'='*15}")
        print(dispersion_table)

        # ============= Outliers (IQR method) ==========================
        q1, q3 = np.percentile(series, [25,75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5*iqr
        upper_bound = q3 + 1.5*iqr
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        outlier_pct = len(outliers)/len(series)*100

        outliers_dict = {
            "IQR": iqr,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outliers_count": len(outliers),
            "outliers_%": outlier_pct
        }

        outliers_table = pd.DataFrame.from_dict(outliers_dict, orient="index", columns=["value"])
        print(f"\n{'='*14} Outliers (IQR) {'='*14}")
        print(outliers_table)



        # ============= Skewness & Kurtosis ==================
        try:
            skew_val = stats.skew(series)
            kurt_val = stats.kurtosis(series, fisher=False)
            print(f"\n{'='*12} Distribution Shape {'='*12}")
            print(f"Skewness: {skew_val:.4f}")
            print(f"Kurtosis: {kurt_val:.4f}")
        except Exception as e:
            print(f"\nSkew/Kurtosis: Error computing ({e})")

        # ============= Correlation with target ==========================
        try:
            # Filtrar filas válidas en ambas columnas a la vez
            valid = df[[col, target]].dropna()
            x = valid[col]
            y = valid[target]

            pearson_r, pearson_p = stats.pearsonr(x, y)

            print(f"\n{'='*10} Correlation with target {'='*10}")
            print(f"Pearson r: {pearson_r:.4f} (p={pearson_p:.4e})")
        except Exception as e:
            print(f"\nCorrelation: Error computing ({e})")


