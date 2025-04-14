import pandas as pd
from sklearn.metrics import r2_score
from datetime import timedelta

from ML.Evaluation.Queries._helper_functions_and_classes import similarity_score_distance, similarity_score_time


def query_accuracy_evaluation(y_true, y_pred, trajectories_count, original_df):
    accuracy_results = []
    # WHERE
    #TODO: HUSK
    accuracy_results.append(where_query_accuracy_evaluation(y_true[0], y_pred[0], original_df))
    print("where accuracy done")
    # DISTANCE
    accuracy_results.append(distance_query_accuracy_evaluation(y_true[1], y_pred[1]))
    print("distance accuracy done")
    # WHEN
    #TODO: HUSK
    accuracy_results.append(when_query_accuracy_evaluation(y_true[2], y_pred[2], original_df))
    print("when accuracy done")

    # HOW LONG
    accuracy_results.append(how_long_query_accuracy_evaluation(y_true[3], y_pred[3], original_df))
    print("how_long accuracy done")

    # COUNT
    accuracy_results.append(count_query_accuracy_evaluation(y_true[4], y_pred[4]))
    print("count accuracy done")

    # KNN
    accuracy_results.append(knn_query_accuracy_evaluation(y_true[5], y_pred[5], trajectories_count))
    print("knn accuracy done")

    # WINDOW
    accuracy_results.append(window_query_accuracy_evaluation(y_true[6], y_pred[6], trajectories_count))
    print("window accuracy done")

    # TODO: Return all results so we can visualize the individual query type
    return sum(accuracy_results) / len(accuracy_results)


def where_query_accuracy_evaluation(y_true, y_pred, trajectories_count, original_df):
    print("")
    results = []
    for i in range(len(y_true)):
        sum_score = 0
        true_set = set(y_true[i]["trajectory_id"])
        pred_set = set(y_pred[i]["trajectory_id"])
        recurring_ids = true_set.intersection(pred_set)
        unique_ids = true_set.symmetric_difference(pred_set)
        TN = (trajectories_count - len(recurring_ids) - len(unique_ids))

        for trajectory_id, trajectory in group_by:
            if trajectory_id in recurring_ids:
                true_point = y_true[i][y_true[i]["trajectory_id"] == trajectory_id][["longitude", "latitude"]]
                pred_point = y_pred[i][y_pred[i]["trajectory_id"] == trajectory_id][["longitude", "latitude"]]
                sum_score += similarity_score_distance(true_point, pred_point, trajectory)
            elif trajectory_id in unique_ids:
                sum_score += 0
            else:
                sum_score += 1
        sum_score /= len(group_by)
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
        r2 = r2_score([distance for distance in y_true[i][y_true[i]["trajectory_id"].isin(recurring_ids)]["distance"]],
                 [distance for distance in y_pred[i][y_pred[i]["trajectory_id"].isin(recurring_ids)]["distance"]])
        result = r2 - (len(unique_ids) / (len(recurring_ids) + len(unique_ids)))
        if result < 0:
            result = 0
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


def how_long_query_accuracy_evaluation(y_true, y_pred):
    return 0.5


def count_query_accuracy_evaluation(y_true, y_pred):
    results = []
    for i in range(0, len(y_true)):
        if max(y_true[i], y_pred[i]) == 0:
            results.append(1)
        else:
            results.append(abs(y_true[i] - y_pred[i]) / max(y_true[i], y_pred[i]))
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

