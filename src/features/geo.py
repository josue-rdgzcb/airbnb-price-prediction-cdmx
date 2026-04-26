 
import numpy as np
from sklearn.neighbors import BallTree


def compute_min_distance_balltree(coords_df, points_array):
    """
    coords_df: DataFrame with columns latitude, longitude
    points_array: np.array (N, 2) with POI coordinates
    """

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