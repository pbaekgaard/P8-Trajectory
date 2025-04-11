import pandas as pd

from tools.scripts._preprocess import main as _load_data
from tools.scripts._load_data import load_compressed_data as _load_compressed_data

from ML.Evaluation._file_access_helper_functions import save_to_file, load_data_from_file


def query_accuracy_evaluation(y_true, y_pred):
    accuracy_results = []
    # WHERE
    accuracy_results.append(where_query_accuracy_evaluation(y_true[0], y_pred[0]))
    # DISTANCE
    accuracy_results.append(distance_query_accuracy_evaluation(y_true[1], y_pred[1]))
    # WHEN
    accuracy_results.append(when_query_accuracy_evaluation(y_true[2], y_pred[2]))
    # HOW LONG
    accuracy_results.append(how_long_query_accuracy_evaluation(y_true[3], y_pred[3]))
    # COUNT
    accuracy_results.append(count_query_accuracy_evaluation(y_true[4], y_pred[4]))
    # KNN
    accuracy_results.append(knn_query_accuracy_evaluation(y_true[5], y_pred[5]))
    # WINDOW
    accuracy_results.append(window_query_accuracy_evaluation(y_true[6], y_pred[6]))

    return sum(accuracy_results) / len(accuracy_results)

def where_query_accuracy_evaluation(y_true, y_pred):
    return 0.5

def distance_query_accuracy_evaluation(y_true, y_pred):
    return 0.5


def when_query_accuracy_evaluation(y_true, y_pred):
    return 0.5


def how_long_query_accuracy_evaluation(y_true, y_pred):
    return 0.5

def count_query_accuracy_evaluation(y_true, y_pred):
    results = []
    for i in range(0, len(y_true)):
        results.append(abs(y_true[i] - y_pred[i]) / max(abs(y_true[i]), abs(y_pred[i])))
    return sum(results) / len(results)


def knn_query_accuracy_evaluation(y_true, y_pred):
    return 0.5


def window_query_accuracy_evaluation(y_true, y_pred):
    return 0.5