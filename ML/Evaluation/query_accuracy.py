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


def query_accuracy_evaluation():
    # Load compressed results and original results and calculate accuracy
    pass
