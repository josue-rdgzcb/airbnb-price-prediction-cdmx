 
# Normalize categorical string values (lowercase + replace spaces with underscores)
def normalize_string_column(series):
    """
    Normalize a pandas Series of strings:
    - Convert to lowercase
    - Replace spaces with underscores
    """
    return series.str.lower().str.replace(" ", "_")