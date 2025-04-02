import pandas as pd
from shapely.geometry import LineString


def window_query_processing(window_query, df):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    t1, t2 = pd.to_datetime(window_query["t1"]), pd.to_datetime(window_query["t2"])

    resulting_trajectory_ids = []
    group_by_df = df.groupby("trajectory_id")
    for trajectory_id, group_df in group_by_df:
        df_within = group_df[(group_df["timestamp"] >= t1) & (group_df["timestamp"] <= t2)]
        df_before = group_df[group_df["timestamp"] < t1].tail(1)
        df_after = group_df[group_df["timestamp"] > t2].head(1)

        if does_overlap(df_within, df_before, df_after, window_query):
            resulting_trajectory_ids.append(trajectory_id)

    return resulting_trajectory_ids


def does_overlap(df_within, df_before, df_after, window_query, overlap_threshold: float = 0.001) -> bool:
    """
    :param df_within: dataframe containing all points within window_query's timeframe
    :param df_before: dataframe containing the point immediately before window_query's timeframe
    :param df_after: dataframe containing the point immediately after window_query's' timeframe
    :param window_query: the query.
    :param overlap_threshold: the error. The amount a line between to trajectories is allowed to defer from the LineString drawnfrom the window_queries points
    :return: True if the trajectories overlap with the points from the window_query with the window_queries time window, False otherwise

    logic overview:
    if we have an overlap between query_line and trajectory_line_within_timeframe -> return True
    if both before and after are empty -> return False
    if within_timeframe points exist:
        if before exists: if before point and first point within timeframe overlap -> return True
        if after exists: if last point within timeframe and after point overlap -> return True
    elif point before and point after exist: If they overlap within timeframe: return True
    return False
    """

    # query LineString
    first_point = window_query["first_point"]
    last_point = window_query["last_point"]
    query_line = LineString([(first_point["longitude"], first_point["latitude"]),
                             (last_point["longitude"], last_point["latitude"])])

    trajectory_points_within_timeframe = list(zip(df_within["longitude"], df_within["latitude"]))
    if len(trajectory_points_within_timeframe) >= 2:
        trajectory_line_within_timeframe = LineString(trajectory_points_within_timeframe)
        if trajectory_line_within_timeframe.distance(query_line) <= overlap_threshold:
            return True

    elif len(trajectory_points_within_timeframe) == 1:
        #TODO: Check om dette ene punkt er i windowet, jah? er det overhovedet nødvendigt? Kun relevant for trajectories med et punkt, jah?
        pass

    if df_before.empty and df_after.empty:
        return False

    if trajectory_points_within_timeframe:
        first_point_within = trajectory_points_within_timeframe[0]
        last_point_within = trajectory_points_within_timeframe[-1]

        if not df_before.empty:
            before_point = df_before.iloc[-1][["longitude", "latitude"]].to_list()

            trajectory_line_before_timeframe = LineString([before_point, first_point_within])

            if trajectory_line_before_timeframe.distance(query_line) <= overlap_threshold:
                before_time = df_before.iloc[0]["timestamp"]
                timestamp_of_first_point_within = df_within.iloc[0]["timestamp"]
                adjusted_segment_before = get_adjusted_trajectory_segment(before_point, first_point_within, before_time,
                                                                          timestamp_of_first_point_within, window_query["t1"], window_query["t2"])

                if adjusted_segment_before and adjusted_segment_before.distance(query_line) <= overlap_threshold:
                    return True

        if not df_after.empty:
            after_point = df_after.iloc[0][["longitude", "latitude"]].to_list()

            trajectory_line_after_timeframe = LineString([last_point_within, after_point])

            if trajectory_line_after_timeframe.distance(query_line) <= overlap_threshold:
                after_time = df_after.iloc[0]["timestamp"]
                timestamp_of_last_time_within = df_within.iloc[-1]["timestamp"]

                adjusted_segment_after = get_adjusted_trajectory_segment(last_point_within, after_point,
                                                                         timestamp_of_last_time_within,
                                                                         after_time, window_query["t1"],
                                                                         window_query["t2"])
                if adjusted_segment_after and adjusted_segment_after.distance(query_line) <= overlap_threshold:
                    return True

    elif not df_before.empty and not df_after.empty:
        before_point = df_before.iloc[-1][["longitude", "latitude"]].to_list()
        after_point = df_after.iloc[0][["longitude", "latitude"]].to_list()
        trajectory_line_outside_timeframe = LineString([before_point, after_point])

        if trajectory_line_outside_timeframe.distance(query_line) <= overlap_threshold:
            before_time = df_before.iloc[-1]["timestamp"]
            after_time = df_after.iloc[0]["timestamp"]
            adjusted_segment = get_adjusted_trajectory_segment(before_point, after_point, before_time, after_time, window_query["t1"], window_query["t2"])
            if adjusted_segment and adjusted_segment.distance(query_line) <= overlap_threshold:
                return True

    return False


def get_interpolated_point(point1, point2, time1, time2, target_time):
    """
    Given two points (longitude, latitude) with corresponding timestamps,
    interpolate to find the exact position at `target_time`, assuming constant speed.
    """
    if time1 == time2:  # Avoid division by zero
        return point1

    ratio = (target_time - time1) / (time2 - time1)  # Linear interpolation factor
    interpolated_longitude = point1[0] + ratio * (point2[0] - point1[0])
    interpolated_latitude = point1[1] + ratio * (point2[1] - point1[1])

    return (interpolated_longitude, interpolated_latitude)


def get_adjusted_trajectory_segment(point1, point2, time1, time2, query_t1, query_t2):
    """
    Returns the adjusted LineString segment that falls within the timeframe.
    If the entire segment is outside the timeframe, return None.
    """
    # If the entire segment is outside the timeframe, return None
    if time2 < query_t1 or time1 > query_t2:
        return None

    # Determine the actual segment that falls within the timeframe
    new_start = point1 if time1 >= query_t1 else get_interpolated_point(point1, point2, time1, time2, query_t1)
    new_end = point2 if time2 <= query_t2 else get_interpolated_point(point1, point2, time1, time2, query_t2)

    # Create and return the adjusted trajectory segment
    return LineString([new_start, new_end])