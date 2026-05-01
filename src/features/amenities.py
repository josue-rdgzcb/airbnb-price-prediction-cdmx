 
import ast
import re
import pandas as pd

# Import amenity parameters from settings
from src.settings.features_params import (
    SELECTED_AMENITIES,
    AMENITIES_WEIGHTS
)

# Parse amenities column into Python lists
def _parse_amenities(x):
    """
    Parse a stringified amenities value into a Python list.
    - Return [] if value is NaN or cannot be parsed.
    """
    if pd.isna(x):
        return []
    
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return []
    
# Create amenities_list column by parsing stringified lists
def add_amenities_list(df):
    """
    Parse the 'amenities' column into Python lists and
    add a new column 'amenities_list' if not already present.
    """
    if "amenities_list" not in df.columns:
        df["amenities_list"] = df["amenities"].apply(_parse_amenities)
    return df

# Clean individual amenity string
def _clean_amenity(a: str) -> str:
    """
    Clean individual amenity string:
    - Lowercase
    - Remove non-alphanumeric characters except spaces and hyphens
    - Strip leading/trailing spaces
    """
    return re.sub(r'[^a-z0-9\s+\-]', '', a.lower()).strip()


# Feature: count number of amenities per listing
def add_amenity_count(df):
    """
    Add a feature column 'amenity_count' with the number of amenities per listing.
    """
    df = add_amenities_list(df)
    df["amenity_count"] = df["amenities_list"].apply(len)
    return df


# Feature: binning amenities count into categories
def add_amenity_count_binned(df):
    """
    Bin 'amenity_count' into categories: low, medium, high.
    """
    df = add_amenity_count(df)
    df["amenity_count_binned"] = pd.qcut(
        df["amenity_count"],
        q=3,
        labels=["low", "medium", "high"]
    )
    return df

# Cleaning and normalizing amenities list
def add_amenities_list_clean(df):
    """
    Clean and normalize amenities list:
    - Ensure 'amenities_list' exists
    - Create 'amenities_list_clean' with normalized strings
    - Create 'amenities_set' for set-based operations
    """
    if "amenities_list" not in df.columns:
        raise ValueError("amenities_list not found. Run add_amenities_list first.")

    if "amenities_list_clean" not in df.columns:
        df["amenities_list_clean"] = df["amenities_list"].apply(
            lambda lst: [_clean_amenity(a) for a in lst]
        )

    if "amenities_set" not in df.columns:
        df["amenities_set"] = df["amenities_list_clean"].apply(set)

    return df


# Keyword matching with exclusions
def _has_keyword(s, include, exclude=None):
    """
    Check if any keyword in 'include' is present in set 's',
    excluding matches with keywords in 'exclude'.
    """
    exclude = exclude or []
    return any(
        (inc in a) and not any(exc in a for exc in exclude)
        for a in s for inc in include
    )


# Adding special amenities features
def add_special_amenities(df):
    """
    Add binary features for special amenities:
    - Washer (excluding dishwasher)
    - Pool (excluding pool table, whirlpool variants)
    - Streaming platforms (Netflix, HBO, etc.)
    """
    df = add_amenities_list_clean(df)

    # Washer (excluding dishwasher)
    df["has_washer"] = df["amenities_set"].apply(
        lambda s: _has_keyword(s, ["washer"], ["dishwasher"])
    )

    # Pool (excluding noise)
    df["has_pool"] = df["amenities_set"].apply(
        lambda s: _has_keyword(
            s,
            ["pool"],
            ["pool table", "whirlpool", "wirpool", "whirpool"]
        )
    )

    # Streaming
    streaming_keywords = ['netflix','hbo','prime video','disney+','apple tv','hulu']
    df["has_streaming_platform"] = df["amenities_set"].apply(
        lambda s: any(any(k in a for k in streaming_keywords) for a in s)
    )

    return df


# Adding simple amenities features
def add_simple_amenities(df):
    """
    Add binary features for a predefined list of simple amenities.
    Returns DataFrame and list of created column names.
    """
    df = add_amenities_list_clean(df)

    simple_amenities = [
        'wifi','kitchen','hot water','essentials','bed linens','microwave',
        'refrigerator','air conditioning','heating','cooking basics',
        'dishes and silverware','iron','hair dryer','dedicated workspace',
        'dining table','dishwasher','freezer','coffee maker','blender',
        'self check-in','private entrance','elevator','free parking',
        'pets allowed','cleaning available during stay','tv','pool table',
        'piano','game console','ping pong table','patio or balcony',
        'backyard','sauna','city skyline view','outdoor furniture',
        'outdoor dining area','smoke alarm','carbon monoxide alarm',
        'fire extinguisher','exterior security cameras on property',
        'first aid kit'
    ]

    created_cols = []

    for amenity in simple_amenities:
        col = "has_" + amenity.replace(" ", "_").replace("+", "plus").replace("-", "_")
        df[col] = df["amenities_set"].apply(
            lambda s: any(amenity in a for a in s)
        )
        created_cols.append(col)

    return df, created_cols


# Feature: orchestrator -> combine special and simple amenities
def add_has_amenity_features(df):
    """
    Orchestrator: add both special and simple amenity features.
    Returns DataFrame and list of created 'has_' columns.
    """
    df = add_special_amenities(df)
    df, has_amenity_cols = add_simple_amenities(df)
    return df, has_amenity_cols


# Feature: compute weighted amenity score
def add_amenity_score(df):
    """
    Compute weighted amenity score based on SELECTED_AMENITIES and AMENITIES_WEIGHTS.
    """
    df["amenity_score"] = 0.0
    for col in SELECTED_AMENITIES:
        if col in df.columns:
            df["amenity_score"] += df[col] * AMENITIES_WEIGHTS[col]
    return df
