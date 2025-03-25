from tools.scripts._load_data import main as _load_data, load_compressed_data as _load_compressed_data
import numpy as np
import pandas as pd
from datetime import datetime

def query_accuracy_evaluation(queries):
    # compressed_data, original_dataset = _load_compressed_data(), _load_data()
    original_dataset = _load_data()
    query_original_dataset(original_dataset, queries)

def query_original_dataset(dataset, queries):
    group_by_df = dataset.groupby("trajectory_id")

    where_query = queries["where"]
    distance_query = queries["distance"]

    where_query_results = where_query_processing(where_query, group_by_df)
    distance_query_results = distance_query_processing(group_by_df, distance_query)

    return where_query_results

def where_query_processing(where_query, group_by_df):
    where_query_df = pd.DataFrame([], columns=["trajectory_id", "less_than_time", "less_than_longitude",
                                                       "less_than_latitude", "greater_than_time",
                                                       "greater_than_longitude", "greater_than_latitude"])

    for trajectory_id, group_df in group_by_df:
        less_than = group_df[group_df["timestamp"] <= where_query]
        if less_than.empty: continue
        first_less_than = less_than.iloc[-1][["timestamp", "longitude", "latitude"]].to_frame().T.reset_index(drop=True)

        greater_than = group_df[group_df["timestamp"] >= where_query]
        first_greater_than = greater_than.iloc[0][["timestamp", "longitude", "latitude"]].to_frame().T.reset_index(
            drop=True) if not greater_than.empty else pd.DataFrame([], columns=["timestamp", "longitude", "latitude"])

        concatenated_df = pd.concat([first_less_than, first_greater_than], axis=1)

        new_row = pd.DataFrame([[trajectory_id] + concatenated_df.values.flatten().tolist()],
                               columns=where_query_df.columns)

        where_query_df = pd.concat([where_query_df, new_row], ignore_index=True)
        concatenated_df = None

    denominator = (pd.to_datetime(where_query_df["greater_than_time"]) - pd.to_datetime(where_query_df["less_than_time"])).dt.total_seconds()
    denominator = denominator.replace(0, np.nan)

    percentage_distances_to_less_than = (datetime.strptime(where_query, "%Y-%m-%d %H:%M:%S") - pd.to_datetime(where_query_df["less_than_time"])).dt.total_seconds() / denominator
    percentage_distances_to_less_than = percentage_distances_to_less_than.fillna(0)

    diff_longitude = where_query_df["greater_than_longitude"] - where_query_df["less_than_longitude"]
    diff_latitude = where_query_df["greater_than_latitude"] - where_query_df["less_than_latitude"]

    where_longitude = where_query_df["less_than_longitude"] + (percentage_distances_to_less_than * diff_longitude)
    where_latitude = where_query_df["less_than_latitude"] + (percentage_distances_to_less_than * diff_latitude)


    where_query_result = pd.concat([where_query_df["trajectory_id"], where_longitude, where_latitude], axis=1)
    return where_query_result

def distance_query_processing(distance_query, group_by_df):
    first_positions = where_query_processing(distance_query["time_first"], group_by_df)
    last_positions = where_query_processing(distance_query["time_last"], group_by_df)

    # position_df = pd.DataFrame(columns=["trajectory_id", "longitude", "latitude"])
    middle_positions = group_by_df[distance_query["time_first"] < group_by_df["timestamp"] < distance_query["time_last"]]

    #TODO: check of fix concat
    position_df = pd.concat([first_positions, middle_positions ,last_positions], axis=1)

    distance_query_result = calculate_distances(position_df)
    return distance_query_result

# TODO: Calculate the distance correctly using something like the "Haversine" formula
#  for each position pair in "first_positions" and "second_positions"
def calculate_distances(position_df: pd.DataFrame) -> pd.DataFrame:
    return position_df

def when_query_processing(when_query, group_by_df):
    # when_query_df = pd.DataFrame([], columns=["trajectory_id", "less_than_time", "less_than_longitude",
    #                                                    "less_than_latitude", "greater_than_time",
    #                                                    "greater_than_longitude", "greater_than_latitude"])

    pos_before = None

    pos_after = None


def query_compressed_data():
    pass

def create_queries():
    return {
        "where": "2008-02-02 15:38:08",
        "distance": {
            "time_first": "2008-02-02 15:38:08",
            "time_last": "2008-02-02 15:48:08"
        }
    }

if __name__ == '__main__':
    query_accuracy_evaluation(create_queries())