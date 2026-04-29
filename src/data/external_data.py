from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

def get_pois_attractions():
    path = BASE_DIR / "data/external/pois_attractions.csv"

    df = pd.read_csv(path)

    df = df.dropna(subset=["latitude", "longitude"])

    return df[["latitude", "longitude"]].values 

def get_pois_commercial():
    path = BASE_DIR / "data/external/pois_commercial.csv"

    df = pd.read_csv(path)

    df = df.dropna(subset=["latitude", "longitude"])

    return df[["latitude", "longitude"]].values 
