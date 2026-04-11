import numpy as np
import pandas as pd
from scipy import stats

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

