import os

import geopandas as gpd
import pandas as pd

ROADMAP_SHP = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/roadnet/gis_osm_roads_free_1.shp"))

def mapmatch(data : pd.DataFrame):
    roadmap = gpd.read_file(ROADMAP_SHP)

    return
