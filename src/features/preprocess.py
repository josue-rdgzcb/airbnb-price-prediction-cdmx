import pandas as pd 
import ast

# Normalize categorical string values (lowercase + replace spaces with underscores)
def normalize_string_column(series):
    """
    Normalize a pandas Series of strings:
    - Convert to lowercase
    - Replace spaces with underscores
    """
    return series.str.lower().str.replace(" ", "_")

# Parse column into Python lists
def parse_column(x):
    """
    Parse a stringified host verifications value into a Python list.
    - Return [] if value is NaN or cannot be parsed.
    """
    if pd.isna(x):
        return []
    
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return []
    