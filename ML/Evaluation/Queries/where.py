import pandas as pd
import numpy as np
from datetime import datetime

def where_query_processing(where_query, group_by_df):
    temp_list = []
    for trajectory_id, group_df in group_by_df:
        less_than = group_df[group_df["timestamp"] <= where_query]
        if less_than.empty: continue
        first_less_than = less_than.iloc[-1][["timestamp", "longitude", "latitude"]].to_frame().T.reset_index(drop=True)

        greater_than = group_df[group_df["timestamp"] >= where_query]
        if greater_than.empty: continue
        first_greater_than = greater_than.iloc[0][["timestamp", "longitude", "latitude"]].to_frame().T.reset_index(
            drop=True)

        concatenated_df = None
        concatenated_df = pd.concat([first_less_than, first_greater_than], axis=1)

        new_row = pd.DataFrame([[trajectory_id] + concatenated_df.values.flatten().tolist()],
                               columns=["trajectory_id", "less_than_time", "less_than_longitude",
                                                       "less_than_latitude", "greater_than_time",
                                                       "greater_than_longitude", "greater_than_latitude"])

        temp_list.append(new_row)


    where_query_df = pd.concat(temp_list, ignore_index=True)
    denominator = (pd.to_datetime(where_query_df["greater_than_time"]) - pd.to_datetime(where_query_df["less_than_time"])).dt.total_seconds()
    denominator = denominator.replace(0, np.nan)

    percentage_distances_to_less_than = (datetime.strptime(where_query, "%Y-%m-%d %H:%M:%S") - pd.to_datetime(where_query_df["less_than_time"])).dt.total_seconds() / denominator
    percentage_distances_to_less_than = percentage_distances_to_less_than.fillna(0)

    diff_longitude = where_query_df["greater_than_longitude"] - where_query_df["less_than_longitude"]
    diff_latitude = where_query_df["greater_than_latitude"] - where_query_df["less_than_latitude"]

    where_longitude = where_query_df["less_than_longitude"] + (percentage_distances_to_less_than * diff_longitude)
    where_latitude = where_query_df["less_than_latitude"] + (percentage_distances_to_less_than * diff_latitude)


    where_query_result = pd.concat([where_query_df["trajectory_id"], where_longitude, where_latitude], axis=1)
    where_query_result.columns = ["trajectory_id", "longitude", "latitude"]
    return where_query_result