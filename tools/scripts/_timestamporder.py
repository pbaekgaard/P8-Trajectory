import pandas as pd


def main(data: pd.DataFrame) -> pd.DataFrame:
    data.sort_values(by=["agent_id", "time"], ascending=[True, True], inplace=True)
    return data
