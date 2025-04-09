from pyproj import Transformer
from mappymatch.matchers.lcss.lcss import MatchResult
from mappymatch.constructs.match import Match
from pandas import DataFrame
from shapely.geometry.linestring import LineString
import pandas as pd


def list_of_id_traj_tuples_to_dfs(data: list[(int, MatchResult)]):
    trajectories = []
    i = 0
    for traj_id, traj in data:
        trajectory = pd.DataFrame(columns=["traj_id", "lng", "lat", "timestamp"])
        prev_time = 0.0
        for match in traj.matches:
            match : Match
            if(match.road is None):
                continue
            linestr : LineString = match.road.geom
            lng, lat = linestr.xy
            trans_lng, trans_lat = transform_to_wgs84(lng, lat)
            rows=[]
            prev_time = prev_time + match.road.metadata.get("travel_time", None)
            for plip, plop in zip(trans_lng, trans_lat):
                rows.append({
                    "traj_id": traj_id,
                    "lng": plip,
                    "lat": plop,
                    "timestamp": prev_time
                })
            new_traj = pd.DataFrame(rows)
            trajectory = pd.concat([trajectory, new_traj], ignore_index=True)
        i = i+1
        print(f"Processed trajectory {i} out of {len(data)}")
        trajectories.append(trajectory)
    return trajectories

def transform_to_wgs84(x, y, from_epsg="EPSG:3857"):
    transformer = Transformer.from_crs(from_epsg, "EPSG:4326", always_xy=True)
    return transformer.transform(x, y)