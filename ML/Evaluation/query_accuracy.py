import pandas as pd

from tools.scripts._preprocess import main as _load_data
from tools.scripts._load_data import load_compressed_data as _load_compressed_data

from ML.Evaluation._file_access_helper_functions import save_to_file, load_data_from_file


def query_accuracy_evaluation():
    # Load compressed results and original results and calculate accuracy
    pass

