import pandas as pd 

ATTRACTION_POINTS = pd.read_csv("../data/external/pois_attractions.csv")[["latitude", "longitude"]].values
