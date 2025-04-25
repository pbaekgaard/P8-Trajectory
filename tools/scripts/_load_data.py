#!/usr/bin/env python3
import glob
import os
import warnings

import pandas as pd

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw/taxi_log_2008_by_id"))
PKL_DATA_DIR = os.path.abspath(os.path.join("../../data", "orignial_data.pkl"))
COMPRESSED_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/compressed_data"))


def main() -> pd.DataFrame :
    if not os.path.exists(PKL_DATA_DIR):
        print("loading data from .txt files")
        files = glob.glob(os.path.join(DATA_DIR, "*.txt"))

        headers = ["trajectory_id", "timestamp", "longitude", "latitude"]
        dfs = [pd.read_csv(file, names=headers, delimiter=",").dropna(inplace=False) for file in files]

        df = pd.concat(dfs, ignore_index=True)

        df = df.sort_values(by=["trajectory_id"], ascending=True)

        df.to_pickle(PKL_DATA_DIR)
    else:
        print("loading data from PKL file")
        df = pd.read_pickle(PKL_DATA_DIR)
    return df


def load_compressed_data():
    # TODO: remove once compressed data is available
    return None
    files = glob.glob(os.path.join(COMPRESSED_DATA_DIR, "*.txt"))

    headers = ["trajectory_id", "timestamp", "longitude", "latitude"]
    dfs = [pd.read_csv(file, names=headers, delimiter=",").dropna(inplace=False) for file in files]

    df = pd.concat(dfs, ignore_index=True)
    return df

def count_trajectories():
    return len([f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f)) and f.endswith('.txt')])

if __name__ == '__main__':
    main()
