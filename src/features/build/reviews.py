from src.settings.features_params import REVIEW_SCORE_COLUMNS

# Feature: compute mean of review score columns
def add_review_scores_mean(df):
    """
    Compute the mean of all review score columns defined in REVIEW_SCORE_COLUMNS
    and add a new column 'review_scores_mean' to the DataFrame.
    """
    df = df.copy()
    df["review_scores_mean"] = df[REVIEW_SCORE_COLUMNS].mean(axis=1)
    return df


# Feature: add binary indicator if listing has any review
def add_has_review(df):
    """
    Add a binary feature 'has_review' to the DataFrame.
    The column is True if any of the review score columns
    defined in REVIEW_SCORE_COLUMNS are non-null, otherwise False.
    """
    df = df.copy()
    df["has_review"] = df[REVIEW_SCORE_COLUMNS].notna().any(axis=1)
    return df


