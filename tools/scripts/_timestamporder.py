import pandas as pd


def main(data: pd.DataFrame) -> pd.DataFrame:
    data.sort_values(by=["trajectory_id", "timestamp"], ascending=[True, True], inplace=True)
    return data
