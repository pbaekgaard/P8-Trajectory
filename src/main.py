#!/usr/bin/env python3
import os

from components.loaddata import main as load_data
from components.mapmatching import (load_match_results, mapmatch,
                                    plot_mapmatched_trajectory)


def main():
    print("Loading data")
    data = load_data(dataType='parquet')

    print("Performing Map Matching")
    results = mapmatch(data)
    # results = load_match_results('../match_results.pkl')

    # Plot results for the first taxi
    print("Plotting")
    # folium_map = plot_mapmatched_trajectory(data, results)
    # folium_map.show_in_browser(
    # )
    

if __name__ == "__main__":
    main()
