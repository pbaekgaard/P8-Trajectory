import os
import pickle
from typing import List, Tuple

import osmnx as ox
import pandas as pd
from mappymatch.constructs.match import Match
from mappymatch.constructs.trace import Trace
from mappymatch.maps.nx.nx_map import NxMap
from mappymatch.maps.nx.readers.osm_readers import (NetworkType,
                                                    parse_osmnx_graph)
from mappymatch.matchers.lcss.lcss import LCSSMatcher, MatchResult
from mappymatch.utils.plot import plot_matches, plot_trace
from pyproj import Transformer
from shapely.geometry.linestring import LineString
from tqdm import tqdm  # For progress bars


def getMap():
    # Define the path to the OSM file (Beijing road network)
    # ROADMAP = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Beijing.osm"))
    SAVED_MAP = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../map.json"))
    if os.path.exists(SAVED_MAP):
        print(f"Found existing Map!")
        nx_map = NxMap.from_file(SAVED_MAP)
        return nx_map
    print(f"Creating Graph")
    G = ox.graph.graph_from_place("Beijing, China")
    #
    # print(f"Creating NxMap")
    nx_map = NxMap(parse_osmnx_graph(G, network_type=NetworkType.DRIVE))
    #
    #
    # Load road network
    print(f"Saving to map to: {SAVED_MAP}")
    nx_map.to_file(SAVED_MAP)
    return nx_map


def save_match_results(match_results, filename="match_results.pkl"):
    """Save match results to a file using pickle."""
    with open(filename, "wb") as f:
        pickle.dump(match_results, f)

def load_match_results(filename="match_results.pkl"):
    """Load match results from a file if it exists."""
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    return None

# Ensure the road network has a CRS (Coordinate Reference System)

def mapmatch(data, map : NxMap):
    """
    Perform map matching on trajectory data.
    
    Parameters:
        data (pd.DataFrame): DataFrame with columns ['agent_id', 'lng', 'lat', 'time']
    
    Returns:
        dict: Mapping of agent_id to a dict with trajectory and matched road segments
    """

    print(f"Loading Map")
    matcher = LCSSMatcher(map)
    match_results = {}

    # Convert road network to a GeoDataFrame if not already

    traces = []
    # Process only the first taxi's trajectory
    i : int = 0
    for taxi_id, group in tqdm(data.groupby("agent_id"), desc="Creating Traces"):
        i += 1
        
        # Convert the group of GPS points to a GeoDataFrame
        trace = Trace.from_dataframe(
            group,
            lat_column= "lat",
            lon_column= "lng",
            xy= True
        )
        traces.append((taxi_id, trace))
        if i > 10:
            break


    matches = [(id, matcher.match_trace(t)) for id, t in tqdm(traces, desc="Processing traces")]
    for (id, mat) in matches:
        for match in mat:
            match : MatchResult
            match.crs
        
    print(matches[0][1].matches[1].coordinate)

        # Store both trajectory and matches for this taxi

    return matches


def list_of_id_traj_tuples_to_dfs(data: List[Tuple[int, MatchResult]]):
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


