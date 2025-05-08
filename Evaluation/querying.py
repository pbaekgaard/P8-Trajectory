import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from Queries.count import count_query_processing
from Queries.distance import distance_query_processing
from Queries.how_long import how_long_query_processing
from Queries.knn import knn_query_processing
from Queries.when import when_query_processing
from Queries.where import where_query_processing
from Queries.window import window_query_processing


def query_original_dataset(dataset, queries):
    #TODO: MAYBE: maybe make universal query_dataset_function with query_functions as argument
    group_by_df = dataset.groupby("trajectory_id")

    where_queries = queries["where"]
    where_queries_results = []
    for where_query in where_queries:
        where_queries_results.append(where_query_processing(where_query, group_by_df))
    print("where done")

    distance_queries = queries["distance"]
    distance_queries_results = []
    for distance_query in distance_queries:
        distance_queries_results.append(distance_query_processing(distance_query, group_by_df))
    print("distance done")

    when_queries = queries["when"]
    when_queries_results = []
    for when_query in when_queries:
        when_queries_results.append(when_query_processing(when_query, group_by_df))
    print("when done")

    how_long_queries = queries["how_long"]
    how_long_queries_results = []
    for how_long_query in how_long_queries:
        how_long_queries_results.append(how_long_query_processing(how_long_query, group_by_df))
    print("how long done")

    count_queries = queries["count"]
    count_queries_results = []
    for count_query in count_queries:
        count_queries_results.append(count_query_processing(count_query, group_by_df))
    print("count done")

    knn_queries = queries["knn"]
    knn_queries_results = []
    for knn_query in knn_queries:
        knn_queries_results.append(knn_query_processing(knn_query, dataset))
    print("knn done")

    window_queries = queries["window"]
    window_queries_results = []
    for window_query in window_queries:
        window_queries_results.append(window_query_processing(window_query, dataset))
    print("window done")

    result = where_queries_results, distance_queries_results, when_queries_results, how_long_queries_results, count_queries_results, knn_queries_results, window_queries_results
    return result


def query_compressed_dataset(compressed_dataset, merged_df, queries):
    df = reconstruct_trajectories(compressed_dataset, merged_df)
    result = query_original_dataset(df, queries)
    return result

def reconstruct_trajectories(compressed_dataset, merged_df):
    reconstructed_trajectories = []
    for new_id, compressed_trajectory in compressed_dataset.items():
        reconstructed_points = []
        for (trajectory_id, start_index, end_index) in compressed_trajectory:
            trajectory : pd.DataFrame = merged_df[merged_df["trajectory_id"] == trajectory_id].iloc[start_index:end_index + 1].reset_index(drop=True)
            trajectory = trajectory.sort_values(by=["timestamp"]).reset_index(drop=True)
            reference_trajectory = trajectory.copy()
            trajectory["trajectory_id"] = new_id
            for i, row in trajectory.iterrows():
                trajectory.at[i, "timestamp"] = get_correct_timestamp(row, trajectory, reference_trajectory, new_id)
            reconstructed_points.append(trajectory)
        reconstructed_trajectories.append(pd.concat(reconstructed_points, ignore_index=True))

    return pd.concat(reconstructed_trajectories, ignore_index=True).drop(columns=["timestamp_corrected"])


def get_correct_timestamp(t, trajectory, reference_trajectory, new_id):
    if t["timestamp_corrected"] and new_id in t["timestamp_corrected"]:
        return t["timestamp_corrected"][new_id]
    else:
        t_row = t.name
        if t_row == 0:
            return t["timestamp"]
        return trajectory.iloc[t_row - 1]["timestamp"] + (reference_trajectory.iloc[t_row]["timestamp"] - reference_trajectory.iloc[t_row - 1]["timestamp"])
