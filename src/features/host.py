# Import utility to parse stringified lists into Python lists
from src.features.preprocess import parse_column


# Create host_verifications_list column
def host_verifications_list(df):
    """
    Parse the 'host_verifications' column into Python lists and
    add a new column 'host_verifications_list' if not already present.
    """
    if "host_verifications_list" not in df.columns:
        df["host_verifications_list"] = df["host_verifications"].apply(parse_column)
    return df


# Group host verifications into categories
def group_host_verifications(verif_list):
    """
    Categorize a list of host verifications into groups:
    - unknown: not a list
    - none: empty list
    - extended: contains 'work_email'
    - basic: contains only 'email' and 'phone'
    - low: single verification
    - other: all other cases
    """
    if not isinstance(verif_list, list):
        return "unknown"

    if len(verif_list) == 0:
        return "none"

    if "work_email" in verif_list:
        return "extended"

    if set(verif_list) == {"email", "phone"}:
        return "basic"

    if len(verif_list) == 1:
        return "low"

    return "other"


# Feature: add grouped host verifications
def add_host_verifications_grouped(df):
    """
    Create a new column 'host_verifications_grouped' by applying group_host_verifications.
    """
    df = df.copy()

    if "host_verifications_list" not in df.columns:
        df = host_verifications_list(df)

    df["host_verifications_grouped"] = df["host_verifications_list"].apply(
        group_host_verifications
    )

    return df

