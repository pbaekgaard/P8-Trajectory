
import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error

from Evaluation.Queries._helper_functions_and_classes import (
    similarity_score_distance, similarity_score_time)


def query_accuracy_evaluation(y_true, y_pred, original_df):
    accuracy_results = []
    # WHERE
    accuracy_results.append(("Where", where_query_accuracy_evaluation(y_true[0], y_pred[0], original_df)))
    # print("where accuracy done")

    # DISTANCE
    accuracy_results.append(("Distance",distance_query_accuracy_evaluation(y_true[1], y_pred[1])))
    # print("distance accuracy done")

    # WHEN
    accuracy_results.append(("When",when_query_accuracy_evaluation(y_true[2], y_pred[2], original_df)))
    # print("when accuracy done")

    # HOW LONG
    accuracy_results.append(("How Long",how_long_query_accuracy_evaluation(y_true[3], y_pred[3], original_df)))
    # print("how_long accuracy done")

    # COUNT
    accuracy_results.append(("Count",count_query_accuracy_evaluation(y_true[4], y_pred[4])))
    # print("count accuracy done")

    # KNN
    accuracy_results.append(("KNN",knn_query_accuracy_evaluation(y_true[5], y_pred[5], len(original_df.groupby(["trajectory_id"])))))
    # print("knn accuracy done")

    # WINDOW
    accuracy_results.append(("Window",window_query_accuracy_evaluation(y_true[6], y_pred[6], len(original_df.groupby(["trajectory_id"])))))
    # print("window accuracy done")

    # TODO: Return all results so we can visualize the individual query type
    return sum(accuracy_result[1] for accuracy_result in accuracy_results) / len(accuracy_results), accuracy_results


def where_query_accuracy_evaluation(y_true, y_pred, original_df):
    group_by = original_df.groupby("trajectory_id")

    results = []
    for i in range(len(y_true)):
        sum_score = 0
        true_set = set(y_true[i]["trajectory_id"])
        pred_set = set(y_pred[i]["trajectory_id"])
        recurring_ids = true_set.intersection(pred_set)
        unique_ids = true_set.symmetric_difference(pred_set)
        counter = 0

        for trajectory_id, trajectory in group_by:
            if trajectory_id in recurring_ids:
                true_point = y_true[i][y_true[i]["trajectory_id"] == trajectory_id][["longitude", "latitude"]]
                pred_point = y_pred[i][y_pred[i]["trajectory_id"] == trajectory_id][["longitude", "latitude"]]
                sum_sim = similarity_score_distance(true_point, pred_point, trajectory)
                if sum_sim is not None:
                    sum_score += sum_sim
                else:
                    counter += 1
            elif trajectory_id in unique_ids:
                sum_score += 0
            else:
                sum_score += 1
        sum_score /= (len(group_by) - counter)
        results.append(sum_score)
    return sum(results) / len(results)

def distance_query_accuracy_evaluation(y_true, y_pred):
    results = []
    for i in range(len(y_true)):
        true_set = set(y_true[i]["trajectory_id"])
        pred_set = set(y_pred[i]["trajectory_id"])
        recurring_ids = list(true_set.intersection(pred_set))
        unique_ids = true_set.symmetric_difference(pred_set)
        if len(recurring_ids) == 0:
            results.append(0)
            continue
        y_true_vals = [distance for distance in y_true[i][y_true[i]["trajectory_id"].isin(recurring_ids)]["distance"]]
        y_pred_vals = [distance for distance in y_pred[i][y_pred[i]["trajectory_id"].isin(recurring_ids)]["distance"]] 
        rmse = root_mean_squared_error(y_true_vals, y_pred_vals)
        range_val = max(y_true_vals) - min(y_true_vals)

        if range_val == 0:
            result = max(1 - rmse / max(y_true_vals), 0)
        else:
            result = max(1 - rmse / range_val, 0)
        results.append(result)
    return sum(results) / len(results)


def when_query_accuracy_evaluation(y_true, y_pred, original_df):
    group_by = original_df.groupby("trajectory_id")

    results = []
    for i in range(len(y_true)):
        true_set = set(y_true[i]["trajectory_id"])
        pred_set = set(y_pred[i]["trajectory_id"])
        recurring_ids = list(true_set.intersection(pred_set))
        unique_ids = true_set.symmetric_difference(pred_set)
        sum_score = 0
        for trajectory_id, trajectory in group_by:
            if trajectory_id in recurring_ids:
                sum_score += similarity_score_time(y_true[i][y_true[i]["trajectory_id"] == trajectory_id]["timestamp"].iloc[0], y_pred[i][y_pred[i]["trajectory_id"] == trajectory_id]["timestamp"].iloc[0], trajectory)
            elif trajectory_id in unique_ids:
                sum_score += 0
            else:
                sum_score += 1
        results.append(sum_score / len(group_by))
    return sum(results) / len(results)


def how_long_query_accuracy_evaluation(y_true, y_pred, original_df):
    group_by = original_df.groupby("trajectory_id")
    epsilon = 1.0 * 10**(-6) # 1 Microsecond

    results = []
    for i in range(len(y_true)):
        true_set = set(y_true[i]["trajectory_id"])
        pred_set = set(y_pred[i]["trajectory_id"])
        recurring_ids = list(true_set.intersection(pred_set))
        unique_ids = true_set.symmetric_difference(pred_set)
        sum_score = 0
        for trajectory_id in recurring_ids:
            sum_score += max(1 - (abs((y_true[i][y_true[i]["trajectory_id"] == trajectory_id]["time_difference"].iloc[0] - y_pred[i][y_pred[i]["trajectory_id"] == trajectory_id]["time_difference"].iloc[0])) / max(y_true[i][y_true[i]["trajectory_id"] == trajectory_id]["time_difference"].iloc[0], epsilon)), 0)
        sum_score += 0 * len(unique_ids)
        sum_score += 1 * (len(group_by) - (len(unique_ids) + len(recurring_ids)))
        results.append(sum_score / len(group_by))
    return sum(results) / len(results)


def count_query_accuracy_evaluation(y_true, y_pred):
    results = []
    for i in range(0, len(y_true)):
        if max(y_true[i], y_pred[i]) == 0:
            results.append(1)
        else:
            results.append(1 - (abs(y_true[i] - y_pred[i]) / max(y_true[i], y_pred[i])))
    return sum(results) / len(results)


def knn_query_accuracy_evaluation(y_true, y_pred, trajectories_count):
    #TODO: EVT tjek at rækkefølgen på ids er rigtig også
    results = []
    for i in range(0, len(y_true)):
        true_set = set(y_true[i])
        pred_set = set(y_pred[i])
        recurring_ids = len(true_set.intersection(pred_set))
        unique_ids = len(true_set.symmetric_difference(pred_set))
        TN = (trajectories_count - recurring_ids - unique_ids)
        results.append((recurring_ids + TN) / (recurring_ids + TN + unique_ids))

    return sum(results) / len(results)


def window_query_accuracy_evaluation(y_true, y_pred, trajectories_count):
    results = []
    for i in range(0, len(y_true)):
        true_set = set(y_true[i])
        pred_set = set(y_pred[i])
        recurring_ids = len(true_set.intersection(pred_set))
        unique_ids = len(true_set.symmetric_difference(pred_set))
        TN = (trajectories_count - recurring_ids - unique_ids)
        results.append((recurring_ids + TN) / (recurring_ids + TN + unique_ids))

    return sum(results) / len(results)

