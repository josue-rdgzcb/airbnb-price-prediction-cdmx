
from src.features.geo import compute_min_distance_balltree
from src.settings.settings import ATTRACTION_POINTS


def add_distance_to_attractions(df):

    df["dist_to_nearest_attraction"] = compute_min_distance_balltree(
        df[["latitude", "longitude"]],
        ATTRACTION_POINTS
    )

    return df
