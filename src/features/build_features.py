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
    df = add_host_verifications_grouped(df)     #

    # REVIEWS features
    df = add

    return df




