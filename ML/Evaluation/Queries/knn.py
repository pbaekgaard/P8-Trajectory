import pandas as pd
from pyproj import Transformer

from ML.Evaluation.Queries._helper_functions_and_classes import MaxHeap

transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def knn_query_processing(knn_query, df):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    t1, t2 = pd.to_datetime(knn_query["time_first"]), pd.to_datetime(knn_query["time_last"])

    heap = MaxHeap(knn_query["k"])
    query_point = {"longitude": knn_query["longitude"], "latitude": knn_query["latitude"]}
    group_by_df = df.groupby("trajectory_id")
    for trajectory_id, group_df in group_by_df:
        df_within = group_df[(group_df["timestamp"] >= t1) & (group_df["timestamp"] <= t2)]
        df_before = group_df[group_df["timestamp"] < t1].tail(1)
        df_after = group_df[group_df["timestamp"] > t2].head(1)

        heap.update(trajectory_id, df_within, df_before, df_after, t1, t2, query_point)


    result = heap.get_elements()

    return result

