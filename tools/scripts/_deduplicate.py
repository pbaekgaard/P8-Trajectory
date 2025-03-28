import pandas as pd


def main(data: pd.DataFrame) -> pd.DataFrame:
    data.drop_duplicates(subset=['trajectory_id', 'timestamp'], keep='first', inplace=True)
    data = data[~(data.duplicated(subset=['trajectory_id', 'longitude', 'latitude'], keep='first') & data.duplicated(subset=['trajectory_id', 'longitude', 'latitude'], keep='last'))]
    return data

