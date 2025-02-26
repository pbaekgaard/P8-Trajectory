#!/usr/bin/env python3
import math
import warnings
from pathlib import Path
from time import time

import pandas as pd

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def main(dataType='feather') -> pd.DataFrame :
    time_before = time()
    PREPROCESSED_DATA_EXTENSION = '.feather'
    if dataType == 'parquet':
        PREPROCESSED_DATA_EXTENSION = '.parquet'

    PREPROCESSED_DATA_DIR = str(Path(__file__).parent.parent.parent.resolve()) + "/data/processed/"
    PREPROCESSED_DATA_FILENAME = 'processed_data'
    PREPROCESSED_DATA_ABS = PREPROCESSED_DATA_DIR + PREPROCESSED_DATA_FILENAME + PREPROCESSED_DATA_EXTENSION
    df = None
    if dataType == 'parquet':
        df = pd.read_parquet(PREPROCESSED_DATA_ABS)
    else:
        df = pd.read_feather(PREPROCESSED_DATA_ABS)
    assert df is not None, "data was not loaded properly. Something went wrong!"
    time_after = time()
    time_delta = round(time_after - time_before, 2)
    print(f"Loaded data in {time_delta} seconds!")
    return df

if __name__ == '__main__':
    main()
