import pyproj
from haversine import haversine, Unit
import pandas as pd
from shapely.geometry import Point, LineString
from pyproj import Transformer, enums
import math
import heapq

# CLASSES:
class MaxHeap:
    def __init__(self, max_size):
        self.max_size = max_size
        self.heap = []  # Stores (-distance, obj)

    def _push(self, obj):
        """Pushes an object while ensuring only the smallest distances are kept"""
        # Negate the distance to simulate max heap behavior
        obj["distance"] = -obj["distance"]

        if len(self.heap) < self.max_size:
            heapq.heappush(self.heap, (obj["distance"], obj))
        else:
            # If the new distance is lower than the max in the heap, replace the max element
            if obj["distance"] > self.heap[0][0]:
                heapq.heappushpop(self.heap, (obj["distance"], obj))

    def _pop(self):
        """Removes and returns the smallest distance element"""
        if self.heap:
            _, obj = heapq.heappop(self.heap)
            obj["distance"] = -obj["distance"]  # Restore original value
            return obj
        return None

    def _peek(self):
        """Returns the largest distance element without removing it"""
        if self.heap:
            return -self.heap[0][0], self.heap[0][1]  # Return max element
        return None

    def update(self, trajectory_id, df_within, df_before, df_after, t1, t2, query_point):
        query_point = Point(transformer.transform(query_point["longitude"], query_point["latitude"]))
        trajectory_points_within_timeframe = list(zip(df_within["longitude"], df_within["latitude"]))

        if df_within.empty:
            if df_before.empty or df_after.empty:
                return
            else:
                before_point = df_before.iloc[0][["longitude", "latitude"]].to_list()
                after_point = df_after.iloc[0][["longitude", "latitude"]].to_list()
                before_time = df_before.iloc[0]["timestamp"]
                after_time = df_after.iloc[0]["timestamp"]
                trajectory_line = get_adjusted_trajectory_segment(before_point, after_point, before_time, after_time, t1, t2)

        else:
            if not df_before.empty:
                before_point = df_before.iloc[0][["longitude", "latitude"]].to_list()
                before_time = df_before.iloc[0]["timestamp"]
                first_point_within = trajectory_points_within_timeframe[0]
                timestamp_of_first_point_within = df_within.iloc[0]["timestamp"]
                interpolated_before_point = get_interpolated_point(before_point, first_point_within, before_time, timestamp_of_first_point_within, t1)
                trajectory_points_within_timeframe.insert(0, interpolated_before_point)

            if not df_after.empty:
                after_point = df_after.iloc[0][["longitude", "latitude"]].to_list()
                after_time = df_after.iloc[0]["timestamp"]
                last_point_within = trajectory_points_within_timeframe[-1]
                timestamp_of_last_time_within = df_within.iloc[-1]["timestamp"]
                interpolated_after_point = get_interpolated_point(last_point_within, after_point, timestamp_of_last_time_within, after_time, t2)
                trajectory_points_within_timeframe.append(interpolated_after_point)

            trajectory_line = LineString(trajectory_points_within_timeframe)
        # TODO: distance er ikke i meter. Det er ikke et problem, men kan være svært at debugge.
        distance = query_point.distance(trajectory_line)

        self._push(({"trajectory_id": trajectory_id, "distance": distance}))

    def get_elements(self):
        """Returns elements sorted by smallest distance"""
        return [obj["trajectory_id"] for _, obj in sorted(self.heap, key=lambda x: x[0], reverse=True)]


# FUNCTIONS:

transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

def calculate_distance(position_df: pd.DataFrame) -> int:
    distances = [
        haversine((position_df.iloc[i]['latitude'], position_df.iloc[i]['longitude']),
                  (position_df.iloc[i + 1]['latitude'], position_df.iloc[i + 1]['longitude']),
                  unit=Unit.METERS)
        for i in range(len(position_df) - 1)
    ]
    total_distance = sum(distances)

    return total_distance

def trajectory_df_to_linestring(traj_df):
    """Convert a DataFrame of points into a Shapely LineString.
       Assumes points are in order by timestamp."""
    if traj_df.empty:
        return None
    points = [Point(transformer.transform(lon, lat)) for lon, lat in zip(traj_df["longitude"], traj_df["latitude"])]
    if len(points) == 1:
        return points[0]
    return LineString(points)


def closest_endpoints_on_trajectory_if_within_threshold(query_point, group_df, threshold: float = 100):
    #TODO: RENAME!!!
    """
    Check if query_point is within a threshold distance to trajectory_line.
    Returns:
      (pt1, pt2)
    where (pt1, pt2) are the endpoints of the segment that the query point projects onto.
    """
    trajectory_line = trajectory_df_to_linestring(group_df)
    if trajectory_line is None:
        return None
    distance = query_point.distance(trajectory_line)
    on_traj = distance < threshold
    seg_endpoints = pd.DataFrame()
    if on_traj:
        seg_endpoints = find_segment_endpoints(trajectory_line, group_df,  query_point)
    return seg_endpoints

def find_segment_endpoints(line, df, query_point):
    """
    Given a LineString and its coordinate list, compute the distance along the line where
    query_point projects and return the two consecutive coordinate pairs (segment endpoints)
    that contain this projection.
    """
    if type(line) == Point:
        return df.iloc[[0]]
    proj_distance = line.project(query_point)
    coords = list(line.coords)
    cumulative = 0.0
    for i in range(len(coords) - 1):
        seg = LineString([coords[i], coords[i+1]])
        seg_length = seg.length
        if cumulative + seg_length >= proj_distance:
            return df.iloc[[i, i+1]]
        cumulative += seg_length
    # Fallback: return the last segment if projection is at the very end
    return df.iloc[[-2, -1]]


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


def get_interpolated_point(point1, point2, time1, time2, target_time):
    """
    Given two points (longitude, latitude) with corresponding timestamps,
    interpolate to find the exact position at `target_time`, assuming constant speed.
    """
    if time1 == time2:  # Avoid division by zero
        return point1

    ratio = (target_time - time1) / (time2 - time1)  # Linear interpolation factor
    interpolated_longitude = point1[0] + ratio * (point2[0] - point1[0])
    interpolated_latitude = point1[1] + ratio * (point2[1] - point1[1])

    return interpolated_longitude, interpolated_latitude


def get_adjusted_trajectory_segment(point1, point2, time1, time2, query_t1, query_t2):
    """
    Returns the adjusted LineString segment that falls within the timeframe.
    If the entire segment is outside the timeframe, return None.
    """
    # If the entire segment is outside the timeframe, return None
    if time2 < query_t1 or time1 > query_t2:
        return None

    # Determine the actual segment that falls within the timeframe
    new_start = point1 if time1 >= query_t1 else get_interpolated_point(point1, point2, time1, time2, query_t1)
    new_end = point2 if time2 <= query_t2 else get_interpolated_point(point1, point2, time1, time2, query_t2)

    # Create and return the adjusted trajectory segment
    return LineString([new_start, new_end])
