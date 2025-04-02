from haversine import haversine, Unit
import pandas as pd
from shapely.geometry import Point, LineString
from pyproj import Transformer
import math
import heapq

# CLASSES:
class MaxHeap:
    def __init__(self, max_size):
        self.max_size = max_size
        self.heap = []  # Stores (-distance, obj)

    def push(self, obj):
        """Pushes an object while ensuring only the smallest distances are kept"""
        # Negate the distance to simulate max heap behavior
        obj["distance"] = -obj["distance"]

        if len(self.heap) < self.max_size:
            heapq.heappush(self.heap, (obj["distance"], obj))
        else:
            # If the new distance is lower than the max in the heap, replace the max element
            if obj["distance"] > self.heap[0][0]:
                heapq.heappushpop(self.heap, (obj["distance"], obj))

    def pop(self):
        """Removes and returns the smallest distance element"""
        if self.heap:
            _, obj = heapq.heappop(self.heap)
            obj["distance"] = -obj["distance"]  # Restore original value
            return obj
        return None

    def peek(self):
        """Returns the largest distance element without removing it"""
        if self.heap:
            return -self.heap[0][0], self.heap[0][1]  # Return max element
        return None

    def get_elements(self):
        """Returns elements sorted by smallest distance"""
        return [obj["trajectory_id"] for _, obj in sorted(self.heap, key=lambda x: x[0], reverse=True)]


# FUNCTION:

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
    points = [Point(transformer.transform(lon, lat)) for lon, lat in zip(traj_df["longitude"], traj_df["latitude"])]
    return LineString(points)


def is_point_on_trajectory(query_point, trajectory_line, threshold: float = 0.001):
    #TODO: RENAME!!!
    """
    Check if query_point is within a threshold distance to trajectory_line.
    Returns:
      (pt1, pt2)
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

