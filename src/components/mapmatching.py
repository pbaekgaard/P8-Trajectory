import os
import pandas as pd
from tqdm import tqdm
from mappymatch.constructs.trace import Trace
from mappymatch.constructs.geofence import Geofence
from mappymatch.maps.nx.nx_map import NxMap
from mappymatch.matchers.lcss.lcss import LCSSMatcher

ROADMAP_SHP = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/roadnet/gis_osm_roads_free_1.shp"))

def mapmatch(data : pd.DataFrame):
    traces = []
    match_dfs = {}
    for taxi_id, group in data.groupby("agent_id"):
        trace = Trace.from_dataframe(
            group,
            lat_column= "lat",
            lon_column= "lng",
            xy= True
        )
        traces.append((taxi_id,trace))
    for taxi_id, trace in tqdm(traces, "Matching traces"):
        geofence = Geofence.from_trace(trace, padding = 1e3)
        nx_map = NxMap.from_geofence(geofence=geofence)
        print("MapMatching with LCSSMatcher now")
        matcher = LCSSMatcher(nx_map)
        print("Done map matching")
        matches = matcher.match_trace(trace=trace)
        df = matches.matches_to_dataframe()
        match_dfs[taxi_id] = df
        
    
    return match_dfs
