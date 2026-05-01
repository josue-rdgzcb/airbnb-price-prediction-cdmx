import numpy as np
from sklearn.neighbors import BallTree

# Import external data loaders for POIs (points of interest)
from src.data.external_data import get_pois_attractions, get_pois_commercial

# Import caching utility to avoid reloading data multiple times
from functools import lru_cache

# ...
def compute_min_distance_balltree(coords_df, points_array):

    # Convert to radians
    coords_rad = np.radians(coords_df.values)
    points_rad = np.radians(points_array)

    # Build tree
    tree = BallTree(points_rad, metric="haversine")

    # Query nearest
    distances, _ = tree.query(coords_rad, k=1)

    # Convert to km
    earth_radius_km = 6371
    distances_km = distances * earth_radius_km

    return distances_km.flatten()

# ...
def compute_points_within_radius(coords_df, points_array, radius_km=1.0):

    # Convert to radians
    coords_rad = np.radians(coords_df.values)
    points_rad = np.radians(points_array)

    # Build BallTree
    tree = BallTree(points_rad, metric="haversine")

    # Convert radius from km to radians
    earth_radius_km = 6371
    radius_rad = radius_km / earth_radius_km

    # Query neighbors within radius
    indices = tree.query_radius(coords_rad, r=radius_rad)

    # Count number of neighbors per point
    counts = np.array([len(idx) for idx in indices])

    return counts

# Cached loader for attraction POIs
@lru_cache(maxsize=1)
def load_attraction_points():
    return get_pois_attractions()

# Cached loader for commercial POIs
@lru_cache(maxsize=1)
def load_commercial_points():
    return get_pois_commercial()

# Feature: distance to nearest attraction
def add_distance_to_attractions(df):

    pois_attractions = load_attraction_points()

    df["dist_to_nearest_attraction"] = compute_min_distance_balltree(
        df[["latitude", "longitude"]],
        pois_attractions
    )

    return df

# Feature: density of attractions within 1 km radius
def add_attractions_density(df):

    pois_attractions = load_attraction_points()

    df["attractions_within_radius"] = compute_points_within_radius(
        df[["latitude", "longitude"]],
        pois_attractions,
        radius_km=1.0
    )

    return df

# Feature: density of commercial POIs within 1 km radius
def add_commercial_density(df):

    pois_commercial = load_commercial_points()

    df["commercial_within_radius"] = compute_points_within_radius(
        df[["latitude", "longitude"]],
        pois_commercial,
        radius_km=1.0
    )

    return df