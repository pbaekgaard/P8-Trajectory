import pandas as pd

from tools.scripts._preprocess import main as _load_data
from tools.scripts._load_data import load_compressed_data as _load_compressed_data
from ML.Evaluation.Queries.where import where_query_processing
from ML.Evaluation.Queries.distance import distance_query_processing
from ML.Evaluation.Queries.when import when_query_processing
from ML.Evaluation.Queries.how_long import how_long_query_processing
from ML.Evaluation.Queries.count import count_query_processing
from ML.Evaluation.Queries.knn import knn_query_processing
from ML.Evaluation.Queries.window import window_query_processing


def query_accuracy_evaluation(queries):
    # compressed_data, original_dataset = _load_compressed_data(), _load_data()
    # original_dataset = _load_data()
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

    original_dataset = pd.DataFrame(data, columns=["trajectory_id", "timestamp", "longitude", "latitude"])

    query_original_dataset(original_dataset, queries)

def query_original_dataset(dataset, queries):
    group_by_df = dataset.groupby("trajectory_id")
    # TODO: Det ligner der er noget galt med de første to. CHECK!

    where_queries = queries["where"]
    where_queries_results = []
    for where_query in where_queries:
        where_queries_results.append(where_query_processing(where_query, group_by_df))
    print("where", where_queries_results)
    # distance_queries = queries["distance"]
    # distance_queries_results = []
    # for distance_query in distance_queries:
    #     distance_queries_results.append(distance_query_processing(distance_query, group_by_df))
    #
    # when_queries = queries["when"]
    # when_queries_results = []
    # for when_query in when_queries:
    #     when_queries_results.append(when_query_processing(when_query, group_by_df))
    #
    # how_long_queries = queries["how_long"]
    # how_long_queries_results = []
    # for how_long_query in how_long_queries:
    #     how_long_queries_results.append(how_long_query_processing(how_long_query, group_by_df))
    #
    # count_queries = queries["count"]
    # count_queries_results = []
    # for count_query in count_queries:
    #     count_queries_results.append(count_query_processing(count_query, group_by_df))
    #
    knn_queries = queries["knn"]
    knn_queries_results = []
    for knn_query in knn_queries:
        knn_queries_results.append(knn_query_processing(knn_query, dataset))
    print("knn ", knn_queries_results)

    window_queries = queries["window"]
    window_queries_results = []
    for window_query in window_queries:
        window_queries_results.append(window_query_processing(window_query, dataset))
    print("window ", window_queries_results)
    #
    # return where_queries_results, distance_queries_results, when_queries_results, count_queries_results, knn_queries_results, window_queries_results


def query_compressed_data():
    pass


def create_queries():
    return {
        "where": ["2008-02-02 15:38:08"],
        "distance": [
            {
                "time_first": "2008-02-02 15:38:08",
                "time_last": "2008-02-02 15:58:08"
            },
            {
                "time_first": "2008-02-02 13:31:08",
                "time_last": "2008-02-02 13:31:08"
            }
        ],
        "when": [
            {    # 13:38, 15:41
                "longitude": 116.51230,
                "latitude": 39.92180
            },
            {   # 18:02:30
                "longitude": 116.595000,
                "latitude": 39.991000
            },
            {
                "longitude": 1160.51230,
                "latitude": 390.991000
            }
        ],
        "how_long": [
            # 00:19, 00:01
            {
                "first_point": {
                    "longitude": 116.51230,
                    "latitude": 39.92180
                },
                "last_point": {
                    "longitude": 116.51372,
                    "latitude": 39.92323
                }
            }
        ],
        "count": [
            {
                "longitude": 116.51230,
                "latitude": 39.92180,
                "radius": 10
            },
            {
                "longitude": 116.244311,
                "latitude": 39.911225,
                "radius": 10
            }
        ],
        # TODO: Der kommer nogengange en runtime warning på .distance. sæt query to t1 og t2 til 13:31:08 for at se. Tror det er fordi tiden er den samme. Der kommer også warnings i de andre queries
        "knn": [
            {
                "longitude": 116.51230,
                "latitude": 39.92180,
                "time_first": "2008-02-02 12:38:08",
                "time_last": "2008-02-02 17:58:08",
                "k": 3,
            },
            {
                "longitude": 116.244311,
                "latitude": 39.911225,
                "time_first": "2008-02-02 15:45:08",
                "time_last": "2008-02-02 16:31:08",
                "k": 5,
             },
        ],
        "window": [
            {
                "first_point":
                    {"longitude": 116.244311, "latitude": 39.911225},
                "last_point":
                    {"longitude": 116.51230, "latitude": 39.92180},
                "t1": "2008-02-02 12:38:00",
                "t2": "2008-02-02 17:58:09",
            },
            {
                "first_point":
                    {"longitude": 116.60000, "latitude": 39.99200},
                "last_point":
                    {"longitude": 116.60500, "latitude": 39.992500},
                "t1": "2008-02-02 12:38:00",
                "t2": "2008-02-02 19:00:00",
            }
        ]
    }

if __name__ == '__main__':
    queries = create_queries()
    query_accuracy_evaluation(queries)