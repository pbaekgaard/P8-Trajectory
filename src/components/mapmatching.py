import os
import pickle

import folium
import osmnx as ox
from mappymatch.constructs.match import Match
from mappymatch.constructs.trace import Trace
from mappymatch.maps.nx.nx_map import NxMap
from mappymatch.maps.nx.readers.osm_readers import (NetworkType,
                                                    parse_osmnx_graph)
from mappymatch.matchers.lcss.lcss import LCSSMatcher, MatchResult
from mappymatch.utils.plot import plot_matches, plot_trace
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
    for taxi_id, group in tqdm(data.groupby("agent_id"), desc="Map-matching trajectories"):
        print(f"Mapmatching taxi_{taxi_id}")
        
        # Convert the group of GPS points to a GeoDataFrame
        trace = Trace.from_dataframe(
            group,
            lat_column= "lat",
            lon_column= "lng",
            xy= True
        )
        traces.append((taxi_id, trace))


    matches = [(id, matcher.match_trace(t)) for id, t in traces]

        # Store both trajectory and matches for this taxi

    return matches




def plot_mapmatched_trajectory(data, match_results):
    """
    Plot original and map-matched trajectories on a Folium map.
    
    Parameters:
        data (pd.DataFrame): Original trajectory data with ['agent_id', 'lng', 'lat', 'time']
        match_results (dict): Map-matched results from `mapmatch()` function
    """
    map_center = [data.iloc[0]['lat'], data.iloc[0]['lng']]
    folium_map = folium.Map(location=map_center, zoom_start=14)
    print(match_results)
    for taxi_id, group in tqdm(data.groupby("agent_id"), desc="Map-matching trajectories"):
        tr = Trace.from_dataframe(
            group,
            lat_column= "lat",
            lon_column= "lng",
            xy= True
        )
        folium_map = plot_matches(matches=match_results[taxi_id])
        plot_trace(trace=tr, m=folium_map)
        break

    return folium_map
