import pandas as pd
def main(data, length):
    data.groupby("trajectory_id").apply(
        lambda x: x.iloc[:length]
    ).reset_index(drop=True)
    return data