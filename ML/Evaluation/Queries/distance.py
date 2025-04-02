import pandas as pd

from ML.Evaluation.Queries.where import where_query_processing
from ML.Evaluation.Queries._helper_functions_and_classes import calculate_distance

def distance_query_processing(distance_query, group_by_df):
    first_positions = where_query_processing(distance_query["time_first"], group_by_df)
    last_positions = where_query_processing(distance_query["time_last"], group_by_df)

    distance_query_results = []
    for trajectory_id, group_df in group_by_df:
        if trajectory_id not in first_positions["trajectory_id"].values or trajectory_id not in last_positions["trajectory_id"].values: continue # ignore because we need at least two points to calculate distance
        is_point_between = (distance_query["time_first"] < group_df["timestamp"]) & (group_df["timestamp"] < distance_query["time_last"])
        middle_positions = group_df[is_point_between]

        position_df = pd.concat([first_positions.loc[first_positions["trajectory_id"] == trajectory_id, ["longitude", "latitude"]], middle_positions[["longitude", "latitude"]], last_positions.loc[last_positions["trajectory_id"] == trajectory_id, ["longitude", "latitude"]]])

        distance_query_results.append(pd.DataFrame({"trajectory_id": [trajectory_id], "distance": [calculate_distance(position_df)]}))

    distance_query_result = pd.concat(distance_query_results, ignore_index=True)
    return distance_query_result