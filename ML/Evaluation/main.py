import os
import argparse
import pandas as pd
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__ + "/../../")))

from tools.scripts._preprocess import main as _load_data
from tools.scripts._load_data import load_compressed_data as _load_compressed_data
from ML.Evaluation.query_creation import create_queries, dummy_create_queries
from ML.Evaluation._file_access_helper_functions import load_data_from_file
from ML.Evaluation.querying import query_original_dataset


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--force', action='store_true', help='Force creation/overwrite of evaluation files')
    args = parser.parse_args()

    # create all that does not exist
    if not os.path.exists(os.path.abspath(__file__ + "./files/queries_for_evaluation.pkl")) or args.force:
        create_queries(amount_of_individual_queries=7)
    queries = load_data_from_file({
        "filename": "queries_for_evaluation",
    })
    #queries = dummy_create_queries()
    if not os.path.exists(os.path.abspath(__file__ + "./files/original_query_results.pkl")) or args.force:
        dataset = _load_data()
        #dataset = pd.DataFrame(data, columns=["trajectory_id", "timestamp", "longitude", "latitude"])
        query_original_dataset(dataset, queries)
    if not os.path.exists(os.path.abspath(__file__ + "./files/compressed_query_results.pkl")) or args.force:
        pass

    query_results = load_data_from_file({
        "filename": "original_query_results",
    })

    print(query_results)




