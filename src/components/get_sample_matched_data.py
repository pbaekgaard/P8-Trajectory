from mappymatch import package_root
import pandas as pd
from mappymatch.constructs.trace import Trace
from mappymatch.constructs.geofence import Geofence
from mappymatch.maps.nx.nx_map import NxMap, NetworkType
from mappymatch.matchers.lcss.lcss import LCSSMatcher


def get_sample_matched_data():
    trace = Trace.from_csv(package_root() / "resources/traces/sample_trace_3.csv", lat_column="latitude", lon_column="longitude", xy=True)
    geofence = Geofence.from_trace(trace)
    nx_map = NxMap.from_geofence(geofence)
    matcher = LCSSMatcher(nx_map)
    match_result = matcher.match_trace(trace)
    return match_result