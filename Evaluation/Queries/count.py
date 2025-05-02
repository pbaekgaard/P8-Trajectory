from haversine import Unit, haversine
from Queries._helper_functions_and_classes import get_bounding_box


def count_query_processing(count_query, group_by_df):
    count = 0
    bounding_box = get_bounding_box(count_query["latitude"], count_query["longitude"], count_query["radius"])

    for trajectory_id, group_df in group_by_df:
        # are we inside the bounding box? otherwise, ignore that trajectory.
        if ((group_df["latitude"].max() < bounding_box["min_latitude"]) | (group_df["latitude"].min() > bounding_box["max_latitude"])): continue
        if ((group_df["longitude"].max() < bounding_box["min_longitude"]) | (group_df["longitude"].min() > bounding_box["max_longitude"])): continue

        # we are now inside the bounding box.
        count += len(group_df[group_df.apply(lambda row: haversine((count_query["latitude"], count_query["longitude"]), (row['latitude'], row['longitude']), unit=Unit.METERS) <= count_query["radius"], axis=1)])

    return count
