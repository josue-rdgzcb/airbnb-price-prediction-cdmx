from src.features.geo import compute_min_distance_balltree
from functools import lru_cache
from src.data.external_data import get_pois_attractions
from src.features.geo import compute_points_within_radius
from src.data.external_data import get_pois_commercial



@lru_cache(maxsize=1)
def load_attraction_points():
    return get_pois_attractions()

@lru_cache(maxsize=1)
def load_commercial_points():
    return get_pois_commercial()

# =============
def add_distance_to_attractions(df):

    pois_attractions = load_attraction_points()

    df["dist_to_nearest_attraction"] = compute_min_distance_balltree(
        df[["latitude", "longitude"]],
        pois_attractions
    )

    return df

# ============
def add_attractions_density(df):

    pois_attractions = load_attraction_points()

    df["attractions_within_radius"] = compute_points_within_radius(
        df[["latitude", "longitude"]],
        pois_attractions,
        radius_km=1.0
    )

    return df

# 0====
def add_commercial_density(df):

    pois_commercial = load_commercial_points()

    df["commercial_within_radius"] = compute_points_within_radius(
        df[["latitude", "longitude"]],
        pois_commercial,
        radius_km=1.0
    )

    return df