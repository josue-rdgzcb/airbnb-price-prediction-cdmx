from settings.transform_config import *
from log_transforms import apply_log_transformations
from .winsorization import apply_winsorization
from .scaling import apply_scaling
from .binning import apply_binning


def transform_features(df):

    # log
    df = apply_log_transformations(df, LOG_FEATURES)

    # winsor
    df = apply_winsorization(df, WINSOR_FEATURES)

    # binning
    df = apply_binning(df, BINNING_FEATURES)

    # scaling
    df = apply_scaling(
        df,
        robust_features=ROBUST_SCALE_FEATURES,
        minmax_features=MINMAX_SCALE_FEATURES,
        standard_features=STANDARD_SCALE_FEATURES
    )

    return df