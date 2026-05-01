

# Booking restrictions / minimum nights bins
MINIMUM_NIGHTS_BINS = [0, 3, 29, float("inf")]

# List of selected amenity features to include in scoring
SELECTED_AMENITIES = [
    "has_washer",
    "has_pool",
    "has_kitchen",
    "has_hot_water",
    "has_essentials",
    "has_bed_linens",
    "has_microwave",
    "has_refrigerator",
    "has_air_conditioning",
    "has_heating",
    "has_cooking_basics",
    "has_dishes_and_silverware",
    "has_iron",
    "has_hair_dryer",
    "has_dining_table",
    "has_dishwasher",
    "has_freezer",
    "has_coffee_maker",
    "has_blender",
    "has_self_check_in",
    "has_elevator",
    "has_free_parking",
    "has_tv",
    "has_pool_table",
    "has_patio_or_balcony",
    "has_city_skyline_view",
    "has_outdoor_furniture",
    "has_smoke_alarm",
]

# Dictionary of amenity weights used to compute weighted amenity score
AMENITIES_WEIGHTS = {
    "has_washer": 0.06599917673494388,
    "has_pool": 0.07640906279034472,
    "has_kitchen": 0.027166176041232254,
    "has_hot_water": 0.02663999550811914,
    "has_essentials": 0.035348594450278215,
    "has_bed_linens": 0.04603033287517955,
    "has_microwave": 0.05304858080136863,
    "has_refrigerator": 0.050057379487391356,
    "has_air_conditioning": 0.09124062435123476,
    "has_heating": 0.06296818697894219,
    "has_cooking_basics": 0.041828701717067075,
    "has_dishes_and_silverware": 0.0324592006768626,
    "has_iron": 0.06967579727711887,
    "has_hair_dryer": 0.1423545708706952,
    "has_dining_table": 0.05634592955516059,
    "has_dishwasher": 0.10788055327674656,
    "has_freezer": 0.03487550991418362,
    "has_coffee_maker": 0.10775204793498554,
    "has_blender": 0.03209001647923982,
    "has_self_check_in": 0.0712053644781975,
    "has_elevator": 0.12361748161624177,
    "has_free_parking": 0.10254429781414127,
    "has_tv": 0.20062146306358053,
    "has_pool_table": 0.03318245084454518,
    "has_patio_or_balcony": 0.05735485862965591,
    "has_city_skyline_view": 0.03192048994648707,
    "has_outdoor_furniture": 0.06844712668063994,
    "has_smoke_alarm": 0.05828566971124684,
}

# Fixed normalization bounds for amenity score (training set reference)
AMENITIES_SCORE_MIN = 0.0
AMENITIES_SCORE_MAX = 1.0

