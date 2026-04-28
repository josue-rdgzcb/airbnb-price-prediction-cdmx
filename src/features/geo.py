 
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


def compute_points_within_radius(coords_df, points_array, radius_km=1.0):
    """
    Compute the number of points (POIs) within a given radius for each coordinate.

    Parameters
    ----------
    coords_df : pd.DataFrame
        DataFrame with columns ['latitude', 'longitude']
    points_array : np.array
        Array of shape (N, 2) with POI coordinates
    radius_km : float
        Radius in kilometers

    Returns
    -------
    np.array
        Array with counts of POIs within the radius for each row
    """

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