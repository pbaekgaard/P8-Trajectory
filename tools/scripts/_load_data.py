#!/usr/bin/env python3
import glob
import os
import warnings

import pandas as pd

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/taxi_log_2008_by_id"))
COMPRESSED_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/compressed_data"))

def main() -> pd.DataFrame :
    files = glob.glob(os.path.join(DATA_DIR, "*.txt"))

    headers = ["trajectory_id", "timestamp", "longitude", "latitude"]
    dfs = [pd.read_csv(file, names=headers, delimiter=",").dropna(inplace=False) for file in files]

    df = pd.concat(dfs, ignore_index=True)

    df = df.sort_values(by=["trajectory_id"], ascending=True)

    return df

def load_compressed_data():
    files = glob.glob(os.path.join(COMPRESSED_DATA_DIR, "*.txt"))

    headers = ["trajectory_id", "timestamp", "longitude", "latitude"]
    dfs = [pd.read_csv(file, names=headers, delimiter=",").dropna(inplace=False) for file in files]

    df = pd.concat(dfs, ignore_index=True)
    return df

if __name__ == '__main__':
    main()
