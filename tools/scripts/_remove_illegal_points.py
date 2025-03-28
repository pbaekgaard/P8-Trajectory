import pandas as pd

def main(data: pd.DataFrame) -> pd.DataFrame:
    # WHOLE WORLD BITCHES
    max_latitude = 180
    max_longitude = 180
    min_latitude = -180
    min_longitude = -180

    # BEIJING ONLY
    # min_longitude = 115
    # max_longitude = 118
    # min_latitude = 39
    # max_latitude = 42

    data = data[((data["longitude"] <= max_longitude) & (data["longitude"] >= min_longitude) & (data["latitude"] <= max_latitude) & (data["latitude"] >= min_latitude))]
    return data