import logging
import os
import sys
import time
from functools import partial
from multiprocessing import Pool

import geopandas
import networkx as nx
import osmnx as ox
import pandas as pd
import pyproj
from mappymatch.constructs.geofence import Geofence
from mappymatch.constructs.trace import Trace
from mappymatch.maps.nx.nx_map import NxMap
from mappymatch.matchers.lcss.lcss import LCSSMatcher
from shapely.geometry import LineString

logging.getLogger("mappymatch").setLevel(logging.ERROR)
logging.getLogger("osmnx").setLevel(logging.ERROR)

ROADMAP_SHP = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/roadnet/gis_osm_roads_free_1.shp"))
gdf = geopandas.read_file(ROADMAP_SHP)

# Initialize MultiDiGraph
G = nx.MultiDiGraph()

# Iterate through road segments
print("Creating graph")
for idx, row in gdf.iterrows():
    geom = row.geometry
    if isinstance(geom, LineString):  # Ensure it's a valid road segment
        coords = list(geom.coords)  # Get list of (x, y) coordinates
        
        # Add nodes and edges
        for i in range(len(coords) - 1):
            u, v = coords[i], coords[i + 1]  # Start and end points
            
            # Add nodes
            G.add_node(u, pos=u)
            G.add_node(v, pos=v)
            
            # Add edge (with attributes if needed)
            G.add_edge(u, v, key=idx, length=geom.length, **row.to_dict())

# Check if the GeoDataFrame has a CRS
if gdf.crs is not None:
    G.graph["crs"] = gdf.crs
else:
    # If no CRS is found, assume WGS84 (lat/lon) as a fallback
    G.graph["crs"] = pyproj.CRS("EPSG:4326")
print("Creating NxMap")
nx_map = NxMap(G)

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def process_trace(item):
    """Process a single trace and return the taxi_id and matched dataframe"""
    start = time.time()
    taxi_id, trace = item
    print(f"[{taxi_id}]: Processing!")
    try:
        geofence = Geofence.from_trace(trace, padding=1e3)
        print("creating nxmap from geofence")
        # with HiddenPrints():
        #     nx_map = NxMap.from_geofence(geofence=geofence)
        matcher = LCSSMatcher(nx_map)
        print(f"matching trace to map")
        matches = matcher.match_trace(trace=trace)
        df = matches.matches_to_dataframe()
        print(f"Taxi {taxi_id} took {time.time() - start:.2f} seconds")
        print(f"[{taxi_id}]: DONE!")
        return taxi_id, df
    except Exception as e:
        print(f"Failed to download map for taxi {taxi_id}: {e}")
        return taxi_id, []

def mapmatch(data: pd.DataFrame):
    traces = []
    match_dfs = {}
    
    # Create traces (this part remains sequential)
    for taxi_id, group in data.groupby("agent_id"):
        trace = Trace.from_dataframe(
            group,
            lat_column="lat",
            lon_column="lng",
            xy=True
        )
        traces.append((taxi_id, trace))
    # Parallel processing of trace


    results = []
    results.append(process_trace(traces[0]))
    results.append(process_trace(traces[1]))
    # with Pool(processes=6) as pool:  # Uses all available CPU cores by default
    #     # You can specify number of processes with Pool(processes=4) if you want to limit it
    #     results = pool.map(process_trace, traces)
    
    # Collect results into match_dfs dictionary
    for taxi_id, df in results:
        match_dfs[taxi_id] = df
    print(match_dfs)
        
    return match_dfs

if __name__ == '__main__':
    # Your main code here if running as a script
    pass
