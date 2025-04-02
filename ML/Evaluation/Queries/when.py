from haversine import haversine, Unit
import pandas as pd
from shapely.geometry import Point, LineString
from pyproj import Transformer

from ML.Evaluation.Queries._helper_functions_and_classes import calculate_distance, is_point_on_trajectory, trajectory_df_to_linestring


transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def when_query_processing(when_query, group_by_df):
    when_query_results = []
    for trajectory_id, group_df in group_by_df:
        nearby_points = is_point_on_trajectory(Point(transformer.transform(when_query["longitude"], when_query["latitude"])), trajectory_df_to_linestring(group_df))
        if nearby_points.empty: continue
        when_query_as_df = pd.DataFrame({"latitude": [when_query["latitude"]], "longitude": when_query["longitude"]})
        point_before = group_df[((group_df["latitude"] == nearby_points.iloc[:1]["latitude"].iloc[0]) & (group_df["longitude"] == nearby_points.iloc[:1]["longitude"].iloc[0]))]
        point_after = group_df[((group_df["longitude"] == nearby_points.iloc[1:]["longitude"].iloc[0]) & (group_df["latitude"] == nearby_points.iloc[1:]["latitude"].iloc[0]))]

        total_distance = calculate_distance(pd.concat([nearby_points.iloc[:1], when_query_as_df, nearby_points.iloc[1:]]).reset_index(drop=True))
        distance_to_point_before = calculate_distance(pd.concat([nearby_points.iloc[:1], when_query_as_df]).reset_index(drop=True))
        percentage_distance = distance_to_point_before / total_distance if total_distance != 0 else 0

        diff_timestamp = pd.to_datetime(point_after["timestamp"]).reset_index(drop=True) - pd.to_datetime(point_before["timestamp"]).reset_index(drop=True)
        time_in_when_query_point = pd.to_datetime(point_before["timestamp"]).reset_index(drop=True) + (percentage_distance * diff_timestamp)

        when_query_results.append(pd.DataFrame({"trajectory_id": [trajectory_id], "timestamp": [time_in_when_query_point.iloc[0]]}))

    if len(when_query_results) == 0:
        return pd.DataFrame(columns=["trajectory_id", "timestamp"])
    when_query_result = pd.concat(when_query_results, ignore_index=True)
    return when_query_result