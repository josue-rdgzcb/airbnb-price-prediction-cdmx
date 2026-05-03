from src.features.geo import(
    add_distance_to_attractions,
    add_attractions_density,
    add_commercial_density
)

from src.features.property import(
    add_property_group,
    add_property_group_room
)

from src.features.booking_restrictions import(
    add_minimum_nights_segment
)

from src.features.amenities import(
    add_amenity_count,
    add_amenity_count_binned,
    add_has_amenity_features,
    add_amenity_score
)

from src.features.host import add_host_verifications_grouped

from src.features.reviews import(
    add_review_scores_mean,
    add_has_review
)


# ================= MAIN PIPELINE TO BUILD FEATURES FOR THE DATASET =================
def build_features(df):

    """
    Build engineered features for the dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing raw listing data.

    Returns
    -------
    pandas.DataFrame
        A copy of the input DataFrame with additional engineered features:
        - GEO features: distance to nearest attraction, attractions density, commercial density
        - PROPERTY features: property type grouping, room type grouping
        - BOOKING RESTRICTIONS features: minimum nights segment
        - AMENITIES features: amenities count, binned count, binary indicators, weighted score
        - HOST features: grouped host verifications
        - REVIEWS features: mean of review score columns, binary indicator for presence of reviews
    """

    df = df.copy()  # Work on a copy to avoid modifying original DataFrame

    # GEO features
    df = add_distance_to_attractions(df)        # Minimum distance to nearest attraction
    df = add_attractions_density(df)            # Count of attractions within radius
    df = add_commercial_density(df)             # Count of commercial POIs within radius

    # PROPERTY features
    df = add_property_group(df)                 # Property type grouping
    df = add_property_group_room(df)            # Room type grouping

    # BOOKINK RESTRICTIONS feaures
    df = add_minimum_nights_segment(df)         # Minimun nights segment

    # AMENITIES features
    df = add_amenity_count(df)                  # Count number of amenities per listing
    df = add_amenity_count_binned(df)           # Bin amenities count into low, medium, high categories
    df, _ = add_has_amenity_features(df)        # Add binary features (has_amenity)
    df = add_amenity_score(df)                  # Compute weighted amenity score

    # HOST features
    df = add_host_verifications_grouped(df)     # Group host verifications into categories

    # REVIEWS features
    df = add_review_scores_mean(df)             # Compute mean of review score columns
    df = add_has_review(df)                     # Add binary indicator if listing has any review

    return df




