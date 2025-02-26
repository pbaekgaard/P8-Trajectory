import pandas as pd


def main(data: pd.DataFrame) -> pd.DataFrame:
    data.drop_duplicates(subset=['trajectory_id', 'timestamp'], keep='first', inplace=True)
    return data

