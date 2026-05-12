from src.features.transform.log_transforms import apply_log_transformations
from src.features.transform.winsorization import apply_winsorization
from src.features.transform.binning import apply_binning
from src.features.transform.imputation import apply_imputation
#from src.features.transform_features.scaling import apply_scaling


def apply_transformations(df):

    df = df.copy()  # Work on a copy to avoid modifying original DataFrame

    # imputation
    df = apply_imputation(df)

    # log
    df = apply_log_transformations(df)

    # winsor
    df = apply_winsorization(df)

    # binning
    df = apply_binning(df)

    # scaling
    #df = apply_scaling(df)

    return df