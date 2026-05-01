import pandas as pd
from src.settings.features_params import MINIMUM_NIGHTS_BINS

# Feature: segment listings by minimum nights (short, medium, long stay)
def add_minimum_nights_segment(df):
    """
    Segment listings by minimum nights using fixed bins from settings.MINIMUM_NIGHTS_BINS.
    Labels are defined directly in the function.
    """
    
    df["minimum_nights_segment"] = pd.cut(
        df["minimum_nights"],
        bins=MINIMUM_NIGHTS_BINS,
        labels=["short_stay", "medium_stay", "long_term"]
    )
    return df


