import pandas as pd
from shapely.geometry import Point, LineString
from pyproj import Transformer

from ML.Evaluation.Queries._helper_functions_and_classes import MaxHeap, calculate_distance, trajectory_df_to_linestring

transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def knn_query_processing(knn_query, df):
    #TODO: Fix timestamp til pd.datetime. lav func til dette. det bruges også i window_qury_processing
    df_filtered = df[(df["timestamp"] > knn_query["time_first"]) & (df["timestamp"] < knn_query["time_last"])]

    before_point = df[df["timestamp"] <= knn_query["time_first"]].groupby("trajectory_id").tail(1)
    after_point = df[df["timestamp"] >= knn_query["time_last"]].groupby("trajectory_id").head(1)

    df_result = pd.concat([before_point, df_filtered, after_point])
    df_result = df_result.groupby("trajectory_id").filter(lambda x: len(x) > 1)

    heap = MaxHeap(knn_query["k"])
    point = {"longitude": knn_query["longitude"], "latitude": knn_query["latitude"]}
    df_result.groupby("trajectory_id").apply(lambda x: update_heap(x, heap, point))

    result = heap.get_elements()
    return result

def update_heap(trajectory_df, heap, point):
    #TODO: Fix implementation. Closest point er ligegyldigt! bare lav trajectory line til hele trajectorien.
    trajectory_df['distance'] = trajectory_df.apply(lambda row: calculate_distance(pd.DataFrame({
        "longitude": [row["longitude"], point["longitude"]],
        "latitude": [row["latitude"], point["latitude"]]
    })), axis=1)

    closest_point = trajectory_df.loc[trajectory_df['distance'].idxmin()]
    closest_point_index = closest_point.name
    closest_point = closest_point[['trajectory_id', 'longitude', 'latitude', 'distance']].to_frame().T
    before_point = trajectory_df[(trajectory_df.index < closest_point_index)].iloc[-1:]
    after_point = trajectory_df[(trajectory_df.index > closest_point_index)].iloc[:1]

    trajectory_line = trajectory_df_to_linestring(pd.concat([before_point, closest_point, after_point]))
    distance = Point(transformer.transform(point["longitude"], point["latitude"])).distance(trajectory_line)

    heap.push(({"trajectory_id": trajectory_df["trajectory_id"].iloc[0], "distance": distance}))