import argparse
import os
import sys
import time
from typing import List
import warnings

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/components")))

import ostc
from _file_access_helper_functions import (find_newest_version,
                                           get_best_params,
                                           load_data_from_file, save_to_file)
from compression_ratio import compression_ratio
from ML.reference_set_construction import generate_reference_set
from query_accuracy import query_accuracy_evaluation
from query_creation import create_queries
from querying import query_compressed_dataset, query_original_dataset

from tools.scripts._convert_timestamp_to_unix import \
    main as _timestamp_conversion
from tools.scripts._load_data import count_trajectories
from tools.scripts._preprocess import main as _load_data

warnings.filterwarnings("ignore", category=RuntimeWarning)

data = [        # Beijing Trajectories
    [0, 1201956968, 116.51172, 39.92123],  # Trajectory 1
    [0, 1201958410, 116.51222, 39.92173],
    [0, 1201965600, 116.51372, 39.92323],

    [1, 1201951200, 116.50000, 39.90000],  # Trajectory 2
    [1, 1201952100, 116.51000, 39.91000],

    [2, 1201966200, 116.55000, 39.95000],  # Trajectory 3
    [2, 1201966320, 116.55200, 39.95200],

    [3, 1201949400, 116.50050, 39.91050],  # Trajectory 4
    [3, 1201950300, 116.52050, 39.93050],
    [3, 1201951200, 116.54050, 39.95050],

    [4, 1201969800, 116.57000, 39.97000],  # Trajectory 5
    [4, 1201970100, 116.58000, 39.98000],

    [5, 1201972800, 116.59000, 39.99000],  # Trajectory 6
    [5, 1201973100, 116.60000, 39.99200],
    [5, 1201973400, 116.61000, 39.99300]
]


def mock_compressed_data(df_lol, reference_set_lol):
    reference_set = [
        [0, 1201956968, 116.51172, 39.92123, {1: 1201951200}],  # Trajectory 1
        [0, 1201957210, 116.51222, 39.92173, {1: 1201952100, 2: 1201961520}],
        [0, 1201958400, 116.51372, 39.92323, None],

        [4, 1201962600, 116.57000, 39.97000, {5: 1201965600}],  # Trajectory 5
        [4, 1201962900, 116.58000, 39.98000, {5: 1201965750}]
    ]

    reference_set_df = pd.DataFrame(reference_set, columns=["trajectory_id", "timestamp", "longitude", "latitude",
                                                            "timestamp_corrected"])

    compressed_data = {
        0: [(0, 0, 2)],
        1: [(0, 0, 1)],
        2: [(0, 0, 2)],
        3: [(3, 0, 2)],
        4: [(4, 0, 1)],
        5: [(4, 0, 1), (5, 0, 1)],
    }

    original_data_trimmed = [
        [3, 1201949400, 116.50050, 39.91050],  # Trajectory 4
        [3, 1201950300, 116.52050, 39.93050],
        [3, 1201951200, 116.54050, 39.95050],

        [5, 1201973100, 116.60000, 39.99200],
        [5, 1201973400, 116.61000, 39.99300]
    ]

    original_df_trimmed = pd.DataFrame(original_data_trimmed,
                                       columns=["trajectory_id", "timestamp", "longitude", "latitude"])

    merged_df = pd.concat([reference_set_df, original_df_trimmed], ignore_index=True).sort_values(
        ["trajectory_id", "timestamp"])

    merged_df["timestamp_corrected"] = merged_df["timestamp_corrected"].fillna(False)

    return compressed_data, merged_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-v', '--version', help='Create version x of evaluation files')
    parser.add_argument('-q', '--query', action="store_true", help='Create queries and query results')
    parser.add_argument('-e', '--evaluation', action="store_true", help='Run evaluation')


    args = parser.parse_args()
    dataset = None


    if args.query:
        if args.version:
            version_number = args.version
        else:
            version_number = find_newest_version() + 1

        # create all that does not exist
        if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "files", f"{version_number}-queries_for_evaluation.pkl")):
            create_queries(amount_of_individual_queries=1, version=version_number)
        queries = load_data_from_file({
            "filename": "queries_for_evaluation",
            "version": version_number
        })
        #queries = dummy_create_queries()

        if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "files", f"{version_number}-original_query_results.pkl")):
            dataset = _load_data()
            #dataset = _timestamp_conversion(pd.DataFrame(data, columns=["trajectory_id", "timestamp", "longitude", "latitude"]))


            query_original_dataset_time_start = time.perf_counter_ns()

            query_result = query_original_dataset(dataset, queries)

            query_original_dataset_time_end = time.perf_counter_ns()
            query_original_dataset_time = query_original_dataset_time_end - query_original_dataset_time_start


            result = {
                "data": query_result,
                "times": {
                    "querying_time": query_original_dataset_time
                }
            }

            save_to_file({
                "filename": "original_query_results",
                "version": version_number
            }, result)

        if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "files", f"{version_number}-compressed_query_results.pkl")):
            #dataset = pd.DataFrame(data, columns=["trajectory_id", "timestamp", "longitude", "latitude"])
            dataset = dataset if dataset is not None else _load_data()

            compression_time_start = time.perf_counter_ns()
            clustering_method, clustering_param, batch_size, d_model, num_heads, clustering_metric, num_layers = get_best_params()

            df, reference_set,_,_,_, ref_ids = generate_reference_set(
                df=dataset, clustering_method=clustering_method, clustering_param=clustering_param,
                batch_size=batch_size, d_model=d_model, num_heads=num_heads, clustering_metric=clustering_metric,
                num_layers=num_layers
            )
            compression_time_ml_end = time.perf_counter_ns()
            ml_time = compression_time_ml_end - compression_time_start

            # compressed_dataset, merged_df = mock_compressed_data(df, reference_set)
            numpy_df = df.to_records(index=False)
            numpy_ref_set = reference_set.to_records(index=False)
            compressed_dataset, merged_df, duration_MRTSearch, duration_OSTC = ostc.compress(numpy_df, numpy_ref_set)

            compression_time_end = time.perf_counter_ns()
            compression_time = compression_time_end - compression_time_ml_end

            query_compressed_dataset_time_start = time.perf_counter_ns()

            query_result = query_compressed_dataset(compressed_dataset, merged_df, queries)

            query_compressed_dataset_time_end = time.perf_counter_ns()
            query_compressed_dataset_time = query_compressed_dataset_time_end - query_compressed_dataset_time_start

            result = {
                "data": query_result,
                "times": {
                    "ml_time": ml_time,
                    "compression_time": compression_time,
                    "Total_MRT_time": duration_MRTSearch,
                    "Total_OSTC_time": duration_OSTC,
                    "querying_time": query_compressed_dataset_time
                },
                "compressed_dataset": compressed_dataset,
                "merged_dataset": merged_df,
            }

            save_to_file({
                "filename": "compressed_query_results",
                "version": version_number
            }, result)


    if args.evaluation:
        if args.version:
            version_number = args.version
        else:
            version_number = find_newest_version()

        original_results = load_data_from_file({
            "filename": "original_query_results",
            "version": version_number
        })["data"]

        compressed_results = load_data_from_file({
            "filename": "compressed_query_results",
            "version": version_number
        })["data"]

        dataset = dataset if dataset is not None else _load_data()
        # dataset = pd.DataFrame(data, columns=["trajectory_id", "timestamp", "longitude", "latitude"])

        accuracy, individual_accuracy_results = query_accuracy_evaluation(original_results, compressed_results, dataset)
        accuracy : float
        individual_accuracy_results : List[float]

        comp_ratio : float = compression_ratio(dataset) # COMPRESSION

        evaluation_results = {"accuracy": accuracy, "compression_ratio": comp_ratio, "accuracy_individual_results": individual_accuracy_results}
        save_to_file({
            "filename": "evaluation",
            "version": version_number
        }, evaluation_results)



