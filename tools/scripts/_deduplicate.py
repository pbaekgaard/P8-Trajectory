import pandas as pd


def main(data: pd.DataFrame) -> pd.DataFrame:
    data.drop_duplicates(subset=['agent_id', 'time'], keep='first', inplace=True)
    return data

