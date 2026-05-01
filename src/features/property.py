# Import preprocessing function for string normalization
from src.features.preprocess import normalize_string_column

# Map raw property_type strings into broader property groups
def map_property_type(pt: str) -> str:
    """
    Map a raw property_type string into a broader category.
    Categories: apartment, house, guesthouse, hotel, unique/nature, other.
    """
    pt = str(pt).lower()

    if any(x in pt for x in ["apartment", "condo", "loft", "rental unit", "aparthotel"]):
        return "apartment"
    elif any(x in pt for x in ["home", "townhouse", "villa", "cottage", "vacation home", "casa", "tiny home", "earthen home"]):
        return "house"
    elif any(x in pt for x in ["guesthouse", "guest suite"]):
        return "guesthouse"
    elif any(x in pt for x in ["hotel", "hostel", "bed and breakfast"]):
        return "hotel"
    elif any(x in pt for x in [
        "dome","hut","yurt","tent","barn","camper","farm stay","cabin","chalet",
        "bungalow","lighthouse","nature lodge","holiday park","resort","minsu",
        "shipping container","castle","tower"
    ]):
        return "unique/nature"
    else:
        return "other"

# Feature: group property types into broader categories
def add_property_group(df):
    df["property_group"] = df["property_type"].apply(map_property_type)

    return df

# Feature: combine property group with normalized room type
def add_property_group_room(df):
    df["room_type"] = normalize_string_column(df["room_type"])
    df["property_group_room"] = df["property_group"] + "_" + df["room_type"]
    
    return df

