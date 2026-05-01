# ================================= MODULE IMPORTS =================================

# Import external data loaders for POIs (points of interest)
from src.data.external_data import get_pois_attractions, get_pois_commercial

# Import geographic feature functions
from src.features.geo import compute_min_distance_balltree, compute_points_within_radius

# Import caching utility to avoid reloading data multiple times
from functools import lru_cache

# Import preprocessing function for string normalization
from src.features.preprocess import normalize_string_column

# Import map_property_type function for property type classification
from src.features.property import map_property_type

# Import pandas for DataFrame operations
import pandas as pd

# ================= MAIN PIPELINE TO BUILD FEATURES FOR THE DATASET =================
def build_features(df):

    df = df.copy()  # Work on a copy to avoid modifying original DataFrame

    # GEO features
    df = add_distance_to_attractions(df)   # Minimum distance to nearest attraction
    df = add_attractions_density(df)       # Count of attractions within radius
    df = add_commercial_density(df)        # Count of commercial POIs within radius

    # PROPERTY features
    df = add_property_group(df)            # Property type grouping
    df = add_property_group_room(df)       # Room type grouping

    # BOOKINK RESTRICTIONS feaures
    df = add_minimum_nights_segment(df)    # Minimun nights segment

    return df

# ========================== FEATURE ENGINEERING UTILITIES ==========================

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

# Feature: group property types into broader categories
def add_property_group(df):
    df["property_group"] = df["property_type"].apply(map_property_type)

    return df

# Feature: combine property group with normalized room type
def add_property_group_room(df):
    df["room_type"] = normalize_string_column(df["room_type"])
    df["property_group_room"] = df["property_group"] + "_" + df["room_type"]
    
    return df

# Feature: segment listings by minimum nights (short, medium, long stay)
def add_minimum_nights_segment(df):
    df["minimum_nights_segment"] = pd.cut(
        df["minimum_nights"],
        bins=[0, 3, 29, float("inf")],
        labels=["short_stay", "medium_stay", "long_term"]
    )

    return df




