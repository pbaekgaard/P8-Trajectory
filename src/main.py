#!/usr/bin/env python3
import os

from components.loaddata import main as load_data
from components.mapmatching import mapmatch, plot_mapmatch_results_folium


def main():
    data = load_data(dataType='parquet')

    results = mapmatch(data)

    # Plot results for the first taxi
    taxi_id = list(results.keys())[0]
    plot_mapmatch_results_folium(results, taxi_id)
    

if __name__ == "__main__":
    main()
