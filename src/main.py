#!/usr/bin/env python3
import os
import pickle

from components.loaddata import main as load_data
from components.mapmatching import (getMap, load_match_results, mapmatch,
                                    plot_mapmatched_trajectory)

MATCHED_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "../mapmatched.pkl"))

def main():
    print("Loading data")
    data = load_data(dataType='parquet')

    results = None
    print("Performing Map Matching")
    if os.path.exists(MATCHED_DATA):
        with open(MATCHED_DATA, "rb") as file:
            results = pickle.load(file)
    else:
        map = getMap()


        results = mapmatch(data, map)
        # Save to a pickle file
        with open("../my_data.pkl", "wb") as file:
            pickle.dump(results, file)
        # results = load_match_results('../match_results.pkl')

    # Plot results for the first taxi
    print("Plotting")
    # folium_map = plot_mapmatched_trajectory(data, results)
    # folium_map.show_in_browser(
    # )
    

if __name__ == "__main__":
    main()
