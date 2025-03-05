import pandas as pd

def main(data: pd.DataFrame) -> pd.DataFrame:
    latLowerBound = 39
    latUpperBound = 42
    lngLowerBound = 115
    lngUpperBound  = 118
    filtered_data = data[(data["lat"].between(latLowerBound, latUpperBound)) & (data["lng"].between(lngLowerBound, lngUpperBound))]
    return filtered_data