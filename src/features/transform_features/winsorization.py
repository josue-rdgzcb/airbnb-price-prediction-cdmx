 
def apply_winsorization(df, winsor_config, suffix="_winsor"):
    """
    Apply upper-tail winsorization.

    Parameters
    ----------
    df : pandas.DataFrame
    winsor_config : dict
        Example:
        {
            "beds": 0.99,
            "host_total_listings_count": 0.95
        }

    Returns
    -------
    pandas.DataFrame
    """
    df = df.copy()

    for col, upper_q in winsor_config.items():

        lower = df[col].quantile(0.00)
        upper = df[col].quantile(upper_q)

        df[f"{col}{suffix}"] = df[col].clip(lower, upper)

    return df