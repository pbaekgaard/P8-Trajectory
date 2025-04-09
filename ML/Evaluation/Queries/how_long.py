import pandas as pd

from ML.Evaluation.Queries.when import when_query_processing


def how_long_query_processing(how_long_query, group_by_df):
    first_times = when_query_processing(how_long_query["first_point"], group_by_df)
    last_times = when_query_processing(how_long_query["last_point"], group_by_df)

    how_long_query_results = []
    for trajectory_id, group_df in group_by_df:
        if trajectory_id not in first_times["trajectory_id"].values or trajectory_id not in last_times["trajectory_id"].values: continue # Because we need to have to points on the trajectory to calculate the time it took to travel from A to B

        time_difference = last_times[last_times["trajectory_id"] == trajectory_id]["timestamp"] - first_times[first_times["trajectory_id"] == trajectory_id]["timestamp"]

        how_long_query_results.append(pd.DataFrame({"trajectory_id": [trajectory_id], "time_difference": [time_difference.iloc[0]]}))

    if not how_long_query_results:
        return pd.DataFrame(columns=["trajectory_id", "time_difference"])

    how_long_query_result = pd.concat(how_long_query_results, ignore_index=True)
    return how_long_query_result
