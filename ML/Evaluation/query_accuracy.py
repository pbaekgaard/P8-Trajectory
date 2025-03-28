from rich import columns

from tools.scripts._preprocess import main as _load_data
from tools.scripts._load_data import load_compressed_data as _load_compressed_data
import numpy as np
import pandas as pd
from datetime import datetime
from haversine import haversine, Unit
from shapely.geometry import Point, LineString
import math

def query_accuracy_evaluation(queries):
    # compressed_data, original_dataset = _load_compressed_data(), _load_data()
    original_dataset = _load_data()
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

    # original_dataset = pd.DataFrame(data, columns=["trajectory_id", "timestamp", "longitude", "latitude"])

    query_original_dataset(original_dataset, queries)

def query_original_dataset(dataset, queries):
    group_by_df = dataset.groupby("trajectory_id")

    # where_queries = queries["where"]
    # where_queries_results = []
    # for where_query in where_queries:
    #     where_queries_results.append(where_query_processing(where_query, group_by_df))
    #
    # distance_queries = queries["distance"]
    # distance_queries_results = []
    # for distance_query in distance_queries:
    #     distance_queries_results.append(distance_query_processing(distance_query, group_by_df))
    #
    # when_queries = queries["when"]
    # when_queries_results = []
    # for when_query in when_queries:
    #     when_queries_results.append(when_query_processing(when_query, group_by_df))
    #
    # how_long_queries = queries["how_long"]
    # how_long_queries_results = []
    # for how_long_query in how_long_queries:
    #     how_long_queries_results.append(how_long_query_processing(how_long_query, group_by_df))

    count_queries = queries["count"]
    count_queries_results = []
    for count_query in count_queries:
        count_queries_results.append(count_query_processing(count_query, group_by_df))

    # return where_queries_results, distance_queries_results, when_queries_results, how_long_queries_results, count_queries_results

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
    when_query_results = []
    for trajectory_id, group_df in group_by_df:
        nearby_points = is_point_on_trajectory(Point(when_query["longitude"], when_query["latitude"]), trajectory_to_linestring(group_df))
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

def trajectory_to_linestring(traj_df):
    """Convert a DataFrame of points into a Shapely LineString.
       Assumes points are in order by timestamp."""
    points = [Point(lon, lat) for lon, lat in zip(traj_df["longitude"], traj_df["latitude"])]
    return LineString(points)


def is_point_on_trajectory(query_point, trajectory_line, threshold: float = 0.001):
    """
    Check if query_point is within a threshold distance to trajectory_line.
    Returns a tuple:
      (True/False, distance, (pt1, pt2))
    where (pt1, pt2) are the endpoints of the segment that the query point projects onto.
    """
    distance = query_point.distance(trajectory_line)
    on_traj = distance < threshold
    seg_endpoints = None
    if on_traj:
        seg_endpoints = find_segment_endpoints(trajectory_line, query_point)
    return pd.DataFrame(seg_endpoints, columns=["longitude", "latitude"])

def find_segment_endpoints(line, query_point):
    """
    Given a LineString and its coordinate list, compute the distance along the line where
    query_point projects and return the two consecutive coordinate pairs (segment endpoints)
    that contain this projection.
    """
    proj_distance = line.project(query_point)
    coords = list(line.coords)
    cumulative = 0.0
    for i in range(len(coords) - 1):
        seg = LineString([coords[i], coords[i+1]])
        seg_length = seg.length
        if cumulative + seg_length >= proj_distance:
            return coords[i], coords[i+1]
        cumulative += seg_length
    # Fallback: return the last segment if projection is at the very end
    return coords[-2], coords[-1]

def how_long_query_processing(how_long_query, group_by_df):
    first_times = when_query_processing(how_long_query["first_point"], group_by_df)
    last_times = when_query_processing(how_long_query["last_point"], group_by_df)

    how_long_query_results = []
    for trajectory_id, group_df in group_by_df:
        if trajectory_id not in first_times["trajectory_id"].values or trajectory_id not in last_times["trajectory_id"].values: continue # Because we need to have to points on the trajectory to calculate the time it took to travel from A to B

        time_difference = last_times[last_times["trajectory_id"] == trajectory_id]["timestamp"] - first_times[first_times["trajectory_id"] == trajectory_id]["timestamp"]

        how_long_query_results.append(pd.DataFrame({"trajectory_id": [trajectory_id], "distance": [time_difference.iloc[0]]}))

    how_long_query_result = pd.concat(how_long_query_results, ignore_index=True)
    return how_long_query_result


def count_query_processing(count_query, group_by_df):
    count = 0
    bounding_box = get_bounding_box(count_query["latitude"], count_query["longitude"], count_query["radius"])

    for trajectory_id, group_df in group_by_df:
        # are we inside the bounding box? otherwise, ignore that trajectory.
        if ((group_df["latitude"].max() < bounding_box["min_latitude"]) | (group_df["latitude"].min() > bounding_box["max_latitude"])): continue
        if ((group_df["longitude"].max() < bounding_box["min_longitude"]) | (group_df["longitude"].min() > bounding_box["max_longitude"])): continue

        # we are now inside the bounding box.
        count += len(group_df[group_df.apply(lambda row: haversine((count_query["latitude"], count_query["longitude"]), (row['latitude'], row['longitude']), unit=Unit.METERS) <= count_query["radius"], axis=1)])

    print(count)
    return count


def get_bounding_box(lat, lon, distance):
    """
    Calculate the bounding box coordinates given a central point (latitude, longitude)
    and a distance in meters.

    Parameters:
    lat (float): Latitude of the central point.
    lon (float): Longitude of the central point.
    distance (float): Distance in meters to calculate bounding box.

    Returns:
    pd.DataFrame: A DataFrame containing min/max latitude and longitude.
    """
    # Earth's radius in meters
    R = 6378137.0  # WGS-84 approximate radius

    # Convert latitude and longitude from degrees to radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)

    # Angular distance in radians
    angular_distance = distance / R

    # Latitude boundaries
    min_lat = lat_rad - angular_distance
    max_lat = lat_rad + angular_distance

    # Longitude boundaries (adjusted by latitude to account for Earth's curvature)
    min_lon = lon_rad - angular_distance / math.cos(lat_rad)
    max_lon = lon_rad + angular_distance / math.cos(lat_rad)

    # Convert back to degrees
    min_lat = math.degrees(min_lat)
    max_lat = math.degrees(max_lat)
    min_lon = math.degrees(min_lon)
    max_lon = math.degrees(max_lon)

    return {
        "min_latitude": min_lat,
        "max_latitude": max_lat,
        "min_longitude": min_lon,
        "max_longitude": max_lon
    }


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
            {    # 13:38, 15:41
                "longitude": 116.51230,
                "latitude": 39.92180
            },
            {   # 18:02:30
                "longitude": 116.595000,
                "latitude": 39.991000
            },
            {
                "longitude": 1160.51230,
                "latitude": 390.991000
            }
        ],
        "how_long": [
            # 00:19, 00:01
            {
                "first_point": {
                    "longitude": 116.51230,
                    "latitude": 39.92180
                },
                "last_point": {
                    "longitude": 116.51372,
                    "latitude": 39.92323
                }
            }
        ],
        "count": [
            {
                "longitude": 116.51230,
                "latitude": 39.92180,
                "radius": 10
            },
            {
                "longitude": 116.244311,
                "latitude": 39.911225,
                "radius": 10
            }
        ]
    }
if __name__ == '__main__':
    queries = create_queries()
    query_accuracy_evaluation(queries)