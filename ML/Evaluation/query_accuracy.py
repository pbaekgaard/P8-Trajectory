from rich import columns

from tools.scripts._load_data import main as _load_data, load_compressed_data as _load_compressed_data
import numpy as np
import pandas as pd
from datetime import datetime
from haversine import haversine, Unit

def query_accuracy_evaluation(queries):
    # compressed_data, original_dataset = _load_compressed_data(), _load_data()
    # original_dataset = _load_data()
    data = [
        # Beijing Trajectories
        [0, "2008-02-02 15:36:08", 116.51172, 39.92123],  # Trajectory 1
        [0, "2008-02-02 15:40:10", 116.51222, 39.92173],
        [0, "2008-02-02 16:00:00", 116.51372, 39.92323],

        [1, "2008-02-02 14:00:00", 116.50000, 39.90000],  # Trajectory 2
        [1, "2008-02-02 14:15:00", 116.51000, 39.91000],

        [2, "2008-02-02 16:10:00", 116.55000, 39.95000],  # Trajectory 3
        [2, "2008-02-02 16:12:00", 116.55200, 39.95200],

        [3, "2008-02-02 13:30:00", 116.50050, 39.91050],  # Trajectory 4
        [3, "2008-02-02 13:45:00", 116.52050, 39.93050],
        [3, "2008-02-02 14:00:00", 116.54050, 39.95050],

        [4, "2008-02-02 17:10:00", 116.57000, 39.97000],  # Trajectory 5
        [4, "2008-02-02 17:15:00", 116.58000, 39.98000],

        [5, "2008-02-02 18:00:00", 116.59000, 39.99000],  # Trajectory 6
        [5, "2008-02-02 18:05:00", 116.60000, 39.99200],
        [5, "2008-02-02 18:10:00", 116.61000, 39.99300]
    ]

    original_dataset = pd.DataFrame(data, columns=["trajectory_id", "timestamp", "longitude", "latitude"])

    query_original_dataset(original_dataset, queries)

def query_original_dataset(dataset, queries):
    group_by_df = dataset.groupby("trajectory_id")

    where_queries = queries["where"]
    where_queries_results = []
    for where_query in where_queries:
        where_queries_results.append(where_query_processing(where_query, group_by_df))

    distance_queries = queries["distance"]
    distance_queries_results = []
    for distance_query in distance_queries:
        distance_queries_results.append(distance_query_processing(distance_query, group_by_df))

    when_queries = queries["when"]
    when_queries_results = []
    for when_query in when_queries:
        when_queries_results.append(when_query_processing(when_query, group_by_df))

    return where_queries_results, distance_queries_results, when_queries_results

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


def distance_query_processing(distance_query, group_by_df):
    first_positions = where_query_processing(distance_query["time_first"], group_by_df)
    last_positions = where_query_processing(distance_query["time_last"], group_by_df)

    # position_df = pd.DataFrame(columns=["trajectory_id", "longitude", "latitude"])
    distance_query_results = []
    for trajectory_id, group_df in group_by_df:
        if trajectory_id not in first_positions["trajectory_id"].values or trajectory_id not in last_positions["trajectory_id"].values: continue # ignore because we need at least two points to calculate distance
        is_point_between = (distance_query["time_first"] < group_df["timestamp"]) & (group_df["timestamp"] < distance_query["time_last"])
        middle_positions = group_df[is_point_between]

        position_df = pd.concat([first_positions.loc[first_positions["trajectory_id"] == trajectory_id, ["longitude", "latitude"]], middle_positions[["longitude", "latitude"]], last_positions.loc[last_positions["trajectory_id"] == trajectory_id, ["longitude", "latitude"]]])

        distance_query_results.append(pd.DataFrame({"trajectory_id": [trajectory_id], "distance": [calculate_distance(position_df)]}))

    distance_query_result = pd.concat(distance_query_results, ignore_index=True)
    return distance_query_result


def calculate_distance(position_df: pd.DataFrame) -> int:
    distances = [
        haversine((position_df.iloc[i]['latitude'], position_df.iloc[i]['longitude']),
                  (position_df.iloc[i + 1]['latitude'], position_df.iloc[i + 1]['longitude']),
                  unit=Unit.METERS)
        for i in range(len(position_df) - 1)
    ]
    total_distance = sum(distances)

    return total_distance

def when_query_processing(when_query, group_by_df):
    temp_list = []
    for trajectory_id, group_df in group_by_df:
        less_than = group_df[group_df["timestamp"] <= when_query]
        if less_than.empty: continue
        first_less_than = less_than.iloc[-1][["timestamp", "longitude", "latitude"]].to_frame().T.reset_index(drop=True)

        greater_than = group_df[group_df["timestamp"] >= when_query]
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

    when_query_df = pd.concat(temp_list, ignore_index=True)



    # when_query_df = pd.DataFrame([], columns=["trajectory_id", "less_than_time", "less_than_longitude",
    #                                                    "less_than_latitude", "greater_than_time",
    #                                                    "greater_than_longitude", "greater_than_latitude"])

    pos_before = None

    pos_after = None


def query_compressed_data():
    pass

def create_queries():
    return {
        "where": ["2008-02-02 15:38:08"],
        "distance": [{
            "time_first": "2008-02-02 15:38:08",
            "time_last": "2008-02-02 15:58:08"
        },
        {
            "time_first": "2008-02-02 13:31:08",
            "time_last": "2008-02-02 13:31:08"
        }],
        "when": [
            {    # 15:38
                "longitude": 116.51230,
                "latitude": 39.92180
            },
            {   # 18:02:30
                "longitude": 116.595000,
                "latitude": 39.991000
            }
        ]
    }
if __name__ == '__main__':
    query_accuracy_evaluation(create_queries())