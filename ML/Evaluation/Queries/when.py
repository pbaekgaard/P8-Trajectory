from haversine import haversine, Unit
import pandas as pd
from shapely.geometry import Point, LineString
from pyproj import Transformer

from ML.Evaluation.Queries._helper_functions_and_classes import calculate_distance, closest_endpoints_on_trajectory_if_within_threshold, trajectory_df_to_linestring


transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def when_query_processing(when_query, group_by_df):
    #TODO: CHECK DET VIRKER
    when_query_results = []
    for trajectory_id, group_df in group_by_df:
        nearby_points_df = closest_endpoints_on_trajectory_if_within_threshold(Point(transformer.transform(when_query["longitude"], when_query["latitude"])), group_df)
        if nearby_points_df is None or nearby_points_df.size < 2:
            continue
        point_before = nearby_points_df.iloc[[0]]
        point_after = nearby_points_df.iloc[[1]]
        when_query_as_df = pd.DataFrame({"latitude": [when_query["latitude"]], "longitude": when_query["longitude"]})
        total_distance = calculate_distance(pd.concat([point_before, when_query_as_df, point_after]).reset_index(drop=True))
        distance_to_point_before = calculate_distance(pd.concat([point_before, when_query_as_df]).reset_index(drop=True))
        percentage_distance = distance_to_point_before / total_distance if total_distance != 0 else 0

        diff_timestamp = point_after["timestamp"].reset_index(drop=True) - point_before["timestamp"].reset_index(drop=True)
        time_in_when_query_point = point_before["timestamp"].reset_index(drop=True) + (percentage_distance * diff_timestamp)

        when_query_results.append(pd.DataFrame({"trajectory_id": [trajectory_id], "timestamp": [time_in_when_query_point.iloc[0]]}))
    if when_query_results:
        when_query_result = pd.concat(when_query_results, ignore_index=True)
    else:
        when_query_result = pd.DataFrame(columns=["trajectory_id", "timestamp"])
    return when_query_result