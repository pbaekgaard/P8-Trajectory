import logging
import os
import sys
import time
from functools import partial
from multiprocessing import Pool

import pandas as pd
from mappymatch.constructs.geofence import Geofence
from mappymatch.constructs.trace import Trace
from mappymatch.maps.nx.nx_map import NxMap
from mappymatch.matchers.lcss.lcss import LCSSMatcher

logging.getLogger("mappymatch").setLevel(logging.ERROR)
logging.getLogger("osmnx").setLevel(logging.ERROR)

ROADMAP_SHP = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/roadnet/gis_osm_roads_free_1.shp"))


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
        with HiddenPrints():
            nx_map = NxMap.from_geofence(geofence=geofence)
        matcher = LCSSMatcher(nx_map)
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
    
    # Parallel processing of traces
    with Pool(processes=6) as pool:  # Uses all available CPU cores by default
        # You can specify number of processes with Pool(processes=4) if you want to limit it
        results = pool.map(process_trace, traces)
    
    # Collect results into match_dfs dictionary
    for taxi_id, df in results:
        match_dfs[taxi_id] = df
        
    return match_dfs

if __name__ == '__main__':
    # Your main code here if running as a script
    pass
