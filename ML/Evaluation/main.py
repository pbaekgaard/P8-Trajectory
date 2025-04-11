import os
import argparse
import pandas as pd
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__ + "/../../")))

from tools.scripts._preprocess import main as _load_data
from tools.scripts._load_data import load_compressed_data as _load_compressed_data
from ML.Evaluation.query_creation import create_queries, dummy_create_queries
from ML.Evaluation._file_access_helper_functions import load_data_from_file
from ML.Evaluation.querying import query_original_dataset, query_compressed_dataset
from ML.Evaluation.query_accuracy import query_accuracy_evaluation


data = [
            # Beijing Trajectories
            [0, "2008-02-02 15:36:08", 116.51172, 39.92123],  # Trajectory 1
            [0, "2008-02-02 15:40:10", 116.51222, 39.92173],
            [0, "2008-02-02 16:00:00", 116.51372, 39.92323],

            [1, "2008-02-02 14:00:00", 116.50000, 39.90000],  # Trajectory 2
            [1, "2008-02-02 14:15:00", 116.51000, 39.91000],

            [2, "2008-02-02 16:10:00", 116.55000, 39.95000],  # Trajectory 3
            [2, "2008-02-02 16:12:00", 116.55200, 39.95200],

            [3, "2008-02-02 13:30:00", 116.50050, 39.91050],  # Trajectory 4
            [3, "2008-02-02 13:45:00", 116.52050, 39.93050],
            [3, "2008-02-02 14:00:00", 116.54050, 39.95050],

            [4, "2008-02-02 17:10:00", 116.57000, 39.97000],  # Trajectory 5
            [4, "2008-02-02 17:15:00", 116.58000, 39.98000],

            [5, "2008-02-02 18:00:00", 116.59000, 39.99000],  # Trajectory 6
            [5, "2008-02-02 18:05:00", 116.60000, 39.99200],
            [5, "2008-02-02 18:10:00", 116.61000, 39.99300]
        ]


def mock_compressed_data():
    reference_set = [
        [0, "2008-02-02 15:36:08", 116.51172, 39.92123, {1: "2008-02-02 14:00:00"}],  # Trajectory 1
        [0, "2008-02-02 15:40:10", 116.51222, 39.92173, {1: "2008-02-02 14:15:00", 2: "2008-02-02 16:12:00"}],
        [0, "2008-02-02 16:00:00", 116.51372, 39.92323, None],

        [4, "2008-02-02 17:10:00", 116.57000, 39.97000, {5: "2008-02-02 18:00:00"}],  # Trajectory 5
        [4, "2008-02-02 17:15:00", 116.58000, 39.98000, None]
    ]

    reference_set_df = pd.DataFrame(reference_set, columns=["trajectory_id", "timestamp", "longitude", "latitude",
                                                            "timestamp_corrected"])

    compressed_data = {
        0: [(0, 0, 2)],
        1: [(0, 0, 1)],
        2: [(0, 0, 1)],
        3: [(3, 0, 2)],
        4: [(4, 0, 1)],
        5: [(4, 0, 1), (5, 0, 1)],
    }

    original_data_trimmed = [
        [3, "2008-02-02 13:30:00", 116.50050, 39.91050],  # Trajectory 4
        [3, "2008-02-02 13:45:00", 116.52050, 39.93050],
        [3, "2008-02-02 14:00:00", 116.54050, 39.95050],

        [5, "2008-02-02 20:05:00", 116.60000, 39.99200],
        [5, "2008-02-02 20:10:00", 116.61000, 39.99300]
    ]

    original_df_trimmed = pd.DataFrame(original_data_trimmed,
                                       columns=["trajectory_id", "timestamp", "longitude", "latitude"])

    merged_df = pd.concat([reference_set_df, original_df_trimmed], ignore_index=True).sort_values(
        ["trajectory_id", "timestamp"])

    merged_df["timestamp_corrected"] = merged_df["timestamp_corrected"].fillna(False)

    return compressed_data, merged_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--force', action='store_true', help='Force creation/overwrite of evaluation files')
    args = parser.parse_args()

    # create all that does not exist
    if not os.path.exists(os.path.join(os.path.abspath(__file__), "..", "files", "queries_for_evaluation.pkl")) or args.force:
        create_queries(amount_of_individual_queries=15)
    queries = load_data_from_file({
        "filename": "queries_for_evaluation",
    })
    #queries = dummy_create_queries()
    if not os.path.exists(os.path.join(os.path.abspath(__file__), "..", "files", "original_query_results.pkl")) or args.force:
        dataset = _load_data()
        #dataset = pd.DataFrame(data, columns=["trajectory_id", "timestamp", "longitude", "latitude"])
        query_original_dataset(dataset, queries)
    if not os.path.exists(os.path.join(os.path.abspath(__file__), "..", "files", "compressed_query_results.pkl")) or args.force:
        compressed_dataset, merged_df = mock_compressed_data()
        # compressed_dataset, merged_df = _load_compressed_data()
        query_compressed_dataset(compressed_dataset, merged_df, queries)

    compressed_results = load_data_from_file({
        "filename": "compressed_query_results",
    })

    print("Querying done")
