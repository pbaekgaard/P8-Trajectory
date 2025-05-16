import pandas as pd

def main(df):
    return df[df["trajectory_id"].isin(df["trajectory_id"].drop_duplicates().iloc[:1000])]
