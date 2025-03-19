import os

import geopandas as gpd
import networkx as nx
import osmnx as ox

ROADMAP = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Beijing.osm.geojson"))

print(f"Creating GeoDataFrame")
gdf = gpd.read_file(ROADMAP)

print(f"Creating NX Graph")
G = ox.utils.
print(f"G: {G}")
