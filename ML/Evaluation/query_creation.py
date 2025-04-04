import pandas as pd
import random
from datetime import datetime, timedelta

from ML.Evaluation._file_access_helper_functions import save_to_file

def dummy_create_queries():
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
            {  # 13:38, 15:41
                "longitude": 116.51230,
                "latitude": 39.92180
            },
            {  # 18:02:30
                "longitude": 116.595000,
                "latitude": 39.991000
            },
            {
                "longitude": 116.51230,
                "latitude": 39.991000
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
                "radius": 11
            },
            {
                "longitude": 116.244311,
                "latitude": 39.911225,
                "radius": 1000
            }
        ],
        # time_first < time_last
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
                "time_first": "2008-02-02 13:31:08",
                "time_last": "2008-02-02 13:31:09",
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


def create_queries(amount_of_individual_queries):
    queries = {}
    queries["where"] = create_where_queries(from_date="2008-02-02 13:30:44", to_date="2008-02-08 17:39:19",
                                            times=amount_of_individual_queries)
    queries["distance"] = create_distance_queries(from_date="2008-02-02 13:30:44", to_date="2008-02-08 17:39:19",
                                                  times=amount_of_individual_queries)
    queries["when"] = create_when_queries(from_lon=116.0, to_lon=117.0, from_lat=39.0, to_lat=40.5,
                                          times=amount_of_individual_queries)
    queries["how_long"] = create_how_long_queries(from_lon=116, to_lon=117, from_lat=39, to_lat=40.5,
                                                  times=amount_of_individual_queries)
    queries["count"] = create_count_queries(from_lon=116, to_lon=117, from_lat=39, to_lat=40.5, from_radius=900,
                                            to_radius=1000, times=amount_of_individual_queries)
    queries["knn"] = create_knn_queries(from_lon=116, to_lon=117, from_lat=39, to_lat=40.5,
                                        from_date="2008-02-02 13:30:44", to_date="2008-02-08 17:39:19", from_k=50,
                                        to_k=150, times=amount_of_individual_queries)
    queries["window"] = create_window_queries(from_lon=116, to_lon=117, from_lat=39, to_lat=40.5,
                                              from_date="2008-02-02 13:30:44", to_date="2008-02-08 17:39:19",
                                              times=amount_of_individual_queries)

    save_to_file({
        "filename": "queries_for_evaluation"
    }, queries)


def create_where_queries(from_date, to_date, times):
    where_queries = []
    from_dt = datetime.strptime(from_date, "%Y-%m-%d %H:%M:%S")
    to_dt = datetime.strptime(to_date, "%Y-%m-%d %H:%M:%S")
    total_seconds = int((to_dt - from_dt).total_seconds())

    for _ in range(times):
        random_seconds = random.randint(0, total_seconds)
        random_dt = from_dt + timedelta(seconds=random_seconds)
        where_queries.append(random_dt.strftime("%Y-%m-%d %H:%M:%S"))

    return where_queries


def create_distance_queries(from_date, to_date, times):
    distance_queries = []
    from_dt = datetime.strptime(from_date, "%Y-%m-%d %H:%M:%S")
    to_dt = datetime.strptime(to_date, "%Y-%m-%d %H:%M:%S")
    total_seconds = int((to_dt - from_dt).total_seconds())

    for _ in range(times):
        rand_sec_1 = random.randint(0, total_seconds)
        rand_sec_2 = random.randint(0, total_seconds)

        while rand_sec_1 == rand_sec_2:
            rand_sec_2 = random.randint(0, total_seconds)

        dt1 = from_dt + timedelta(seconds=rand_sec_1)
        dt2 = from_dt + timedelta(seconds=rand_sec_2)

        time_first = min(dt1, dt2).strftime("%Y-%m-%d %H:%M:%S")
        time_last = max(dt1, dt2).strftime("%Y-%m-%d %H:%M:%S")

        distance_queries.append({
            "time_first": time_first,
            "time_last": time_last
        })

    return distance_queries


def create_when_queries(from_lon, to_lon, from_lat, to_lat, times):
    when_queries = []
    for _ in range(times):
        when_queries.append(
            {"longitude": random.uniform(from_lon, to_lon), "latitude": random.uniform(from_lat, to_lat)})
    return when_queries


def create_how_long_queries(from_lon, to_lon, from_lat, to_lat, times):
    how_long_queries = []
    for _ in range(times):
        how_long_queries.append({"first_point": {"longitude": random.uniform(from_lon, to_lon),
                                                 "latitude": random.uniform(from_lat, to_lat)},
                                 "last_point": {"longitude": random.uniform(from_lon, to_lon),
                                                "latitude": random.uniform(from_lat, to_lat)}})
    return how_long_queries


def create_count_queries(from_lon, to_lon, from_lat, to_lat, from_radius, to_radius, times):
    count_queries = []
    for _ in range(times):
        count_queries.append(
            {"longitude": random.uniform(from_lon, to_lon), "latitude": random.uniform(from_lat, to_lat),
             "radius": random.randint(from_radius, to_radius)})
    return count_queries


def create_knn_queries(from_lon, to_lon, from_lat, to_lat, from_date, to_date, from_k, to_k, times):
    from_dt = datetime.strptime(from_date, "%Y-%m-%d %H:%M:%S")
    to_dt = datetime.strptime(to_date, "%Y-%m-%d %H:%M:%S")
    total_seconds = int((to_dt - from_dt).total_seconds())
    knn_queries = []
    for _ in range(times):
        rand_sec_1 = random.randint(0, total_seconds)
        rand_sec_2 = random.randint(0, total_seconds)

        while rand_sec_1 == rand_sec_2:
            rand_sec_2 = random.randint(0, total_seconds)

        dt1 = from_dt + timedelta(seconds=rand_sec_1)
        dt2 = from_dt + timedelta(seconds=rand_sec_2)

        time_first = min(dt1, dt2).strftime("%Y-%m-%d %H:%M:%S")
        time_last = max(dt1, dt2).strftime("%Y-%m-%d %H:%M:%S")

        knn_queries.append({"longitude": random.uniform(from_lon, to_lon), "latitude": random.uniform(from_lat, to_lat),
                            "time_first": time_first, "time_last": time_last, "k": random.randint(from_k, to_k)})

    return knn_queries


def create_window_queries(from_lon, to_lon, from_lat, to_lat, from_date, to_date, times):
    window_queries = []
    from_dt = datetime.strptime(from_date, "%Y-%m-%d %H:%M:%S")
    to_dt = datetime.strptime(to_date, "%Y-%m-%d %H:%M:%S")
    total_seconds = int((to_dt - from_dt).total_seconds())

    for _ in range(times):
        first_point = {"longitude": random.uniform(from_lon, to_lon), "latitude": random.uniform(from_lat, to_lat)}
        last_point = {"longitude": random.uniform(from_lon, to_lon), "latitude": random.uniform(from_lat, to_lat)}
        rand_sec_1 = random.randint(0, total_seconds)
        rand_sec_2 = random.randint(0, total_seconds)

        while rand_sec_1 == rand_sec_2:
            rand_sec_2 = random.randint(0, total_seconds)

        dt1 = from_dt + timedelta(seconds=rand_sec_1)
        dt2 = from_dt + timedelta(seconds=rand_sec_2)

        t1 = min(dt1, dt2).strftime("%Y-%m-%d %H:%M:%S")
        t2 = max(dt1, dt2).strftime("%Y-%m-%d %H:%M:%S")
        window_queries.append({"first_point": first_point, "last_point": last_point, "t1": t1, "t2": t2})

    return window_queries
