import inspect
import os
import sys
from typing import Tuple

import numpy as np
import ostc
import pandas as pd
from pandas.io import parquet

# Get the current frame
frame = inspect.currentframe()
CACHE_DIR = os.path.join(os.path.dirname(__file__), "../__CACHE/")

# Check if frame is not None
if frame is not None:
    # Get the file associated with the current frame
    current_file = inspect.getfile(frame)
    
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(current_file))

    print(current_dir)
    parentdir = os.path.dirname(current_dir)
    sys.path.insert(0, parentdir) 
else:
    print("Unable to get current frame.")

from tools.scripts._get_data import main as get_data
from tools.scripts._load_data import main as load_data


def get_first_x_trajectories(num_trajectories: int, trajectories: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    :param num_trajectories: how many trajectories you want
    :param trajectories: from where to get the trajectories
    :returns: tuple consisting of:
        - dataframe of the selected trajectories
        - unique_trajectories: lookup table for the selected trajectories
    selects the first x (num_trajectories) trajectories with a unique trajectory_id.
    """
    unique_trajectories = trajectories['trajectory_id'].unique()[:num_trajectories]
    df = trajectories[trajectories['trajectory_id'].isin(unique_trajectories)]

    return df, pd.DataFrame(unique_trajectories)

def main():
    df : pd.DataFrame
    unique_trajectories : pd.DataFrame

    if os.path.exists(CACHE_DIR):
        df = parquet.read_parquet(os.path.join(CACHE_DIR, "sample_dataframe.parquet"))
        unique_trajectories = parquet.read_parquet(os.path.join(CACHE_DIR, "sample_unique_trajectories.parquet"))
    else:
        os.makedirs(CACHE_DIR)
        get_data()
        traj_df = load_data()
        df, unique_trajectories = get_first_x_trajectories(10, traj_df)
        df.to_parquet(os.path.join(CACHE_DIR, "sample_dataframe.parquet"))
        unique_trajectories.to_parquet(os.path.join(CACHE_DIR, "sample_unique_trajectories.parquet"))


    nparray : np.ndarray = df.to_numpy()
    # print(nparray)
    # print(nparray.shape)
    ostc.print_ndarray(nparray)


if __name__ == "__main__":
    main()
