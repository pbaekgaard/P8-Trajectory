import os

import folium
import geopandas as gpd
import igraph
import networkx as nx
import osmnx as ox
import pandas as pd
import pyproj
from mappymatch.maps.nx.nx_map import NxMap
from mappymatch.matchers.lcss.lcss import LCSSMatcher
from pyrosm import OSM
from shapely.geometry import Point
from tqdm import tqdm  # For progress bars

# Define the path to the OSM file (Beijing road network)
ROADMAP = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../beijing-latest.osm.pbf"))

# Load road network
osm : OSM= OSM(ROADMAP)

# Get the road network
(nodes, edges) = osm.get_network(nodes=True, network_type="driving")  # Change to "walking" or "all" if needed

# Convert to NetworkX graph
G : nx.MultiDiGraph = osm.to_graph(edges=edges, nodes=nodes, graph_type="networkx")
G.graph['crs'] = pyproj.CRS('EPSG:4326')
map = NxMap(G)


# Ensure the road network has a CRS (Coordinate Reference System)

def mapmatch(data):
    """
    Perform map matching on trajectory data.
    
    Parameters:
        data (pd.DataFrame): DataFrame with columns ['agent_id', 'lng', 'lat', 'time']
    
    Returns:
        dict: Mapping of agent_id to a dict with trajectory and matched road segments
    """
    match_results = {}

    # Convert road network to a GeoDataFrame if not already
    if not isinstance(roads, gpd.GeoDataFrame):
        roads_gdf = gpd.GeoDataFrame(roads)
    else:
        roads_gdf = roads

    # Process only the first taxi's trajectory
    for taxi_id, group in tqdm(data.groupby("agent_id"), desc="Map-matching trajectories"):
        print(f"Headers for taxi_id {taxi_id}: {list(group.columns)}")
        
        # Convert the group of GPS points to a GeoDataFrame
        geometry = [Point(xy) for xy in zip(group["lng"], group["lat"])]
        traj_gdf = gpd.GeoDataFrame(group, geometry=geometry, crs="EPSG:4326")

        # List to store matched road IDs or coordinates for this taxi
        matched_segments = []


        # Store both trajectory and matches for this taxi
        match_results[taxi_id] = {
            "trajectory": traj_gdf,
            "matches": matched_segments
        }
        break  # Only process one taxi

    return match_results

def plot_mapmatch_results_folium(results, taxi_id, output_file="mapmatch_results.html"):
    # Get the trajectory to center the map
    trajectory = results[taxi_id]["trajectory"]
    center_lat = trajectory["lat"].mean()
    center_lon = trajectory["lng"].mean()

    # Create a Folium map centered on the trajectory
    m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="OpenStreetMap")

    # Define a small bounding box around the trajectory for context
    buffer = 0.01  # ~1 km in Beijing
    bounds = [
        trajectory.total_bounds[0] - buffer,  # minx
        trajectory.total_bounds[1] - buffer,  # miny
        trajectory.total_bounds[2] + buffer,  # maxx
        trajectory.total_bounds[3] + buffer   # maxy
    ]
    roads_subset = roads.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]

    # Add road network (gray lines)
    folium.GeoJson(
        roads_subset,
        style_function=lambda x: {"color": "gray", "weight": 1, "opacity": 0.5},
        name="Road Network"
    ).add_to(m)

    # Add matched road segments (blue lines)
    matched_geometries = [match["geometry"] for match in results[taxi_id]["matches"]]
    matched_gdf = gpd.GeoDataFrame(geometry=matched_geometries, crs="EPSG:4326")
    folium.GeoJson(
        matched_gdf,
        style_function=lambda x: {"color": "blue", "weight": 3, "opacity": 0.8},
        name="Matched Roads"
    ).add_to(m)

    # Add original GPS points (red markers)
    for idx, row in trajectory.iterrows():
        folium.Marker(
            location=[row["lat"], row["lng"]],
            popup=f"Time: {row['time']}",
            icon=folium.Icon(color="red", icon="circle")
        ).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Save the map
    m.save(output_file)
    print(f"Map saved as {output_file}. Open it in a web browser to view.")

# Example usage
if __name__ == "__main__":
    print('henlo')
    # # Sample data for one taxi
    # sample_data = pd.DataFrame({
    #     "agent_id": [1, 1, 1],
    #     "lng": [116.305, 116.306, 116.307],
    #     "lat": [39.965, 39.966, 39.967],
    #     "time": ["2008-02-02 13:30:00", "2008-02-02 13:30:10", "2008-02-02 13:30:20"]
    # })
    #
    # # Run map matching
    # results = mapmatch(sample_data)
    #
    # # Plot results for the first taxi
    # taxi_id = list(results.keys())[0]
    # plot_mapmatch_results_folium(results, taxi_id)
