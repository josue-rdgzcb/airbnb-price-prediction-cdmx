 
import numpy as np
import pandas as pd

def transform_feature(df, col, target="log_price"):
    """
    Apply transformations (winsor moderate/strong, log, sqrt) to a single feature
    and return a comparison DataFrame with 'transformation' as the first column.
    """
    # Winsorization (moderate: 1% upper tail)
    lower = df[col].quantile(0.00)
    upper = df[col].quantile(0.99)
    winsor_moderate = df[col].clip(lower, upper)

    # Winsorization (strong: 5% upper tail)
    upper_strong = df[col].quantile(0.95)
    winsor_strong = df[col].clip(lower, upper_strong)

    # Log transform
    log_trans = np.log1p(df[col])

    # Square root transform
    sqrt_trans = np.sqrt(df[col])

    # Collect results with clean transformation names
    features = {
        "original": df[col],
        "log": log_trans,
        "sqrt": sqrt_trans,
        "winsor_moderate": winsor_moderate,
        "winsor_strong": winsor_strong
    }

    results = []
    for name, series in features.items():
        results.append({
            "transformation": name,
            "mean": round(series.mean(), 4),
            "std": round(series.std(), 4),
            "skew": round(series.skew(), 4),
            "corr_with_target": round(series.corr(df[target]), 4)
        })

    return pd.DataFrame(results)


def compare_transformations(df, cols, target="log_price"):
    """
    Generate separate comparison tables for each feature in cols.
    Prints each table with headers.
    """
    cols = [c for c in cols if c != target]  # exclude target from output

    for col in cols:
        table = transform_feature(df, col, target)
        print(f"\n========== VARIABLE: {col} ==========\n")
        print(table)
        