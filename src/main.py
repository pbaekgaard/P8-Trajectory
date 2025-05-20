import inspect
import os
import sys
from typing import Tuple

import numpy as np
import ostc
import pandas as pd
from pandas.io import parquet

sys.path.append(os.path.dirname(os.path.abspath(__file__+"/..")))
from ML.Evaluation._file_access_helper_functions import get_best_params
from ML.reference_set_construction import (generate_reference_set,
                                           get_first_x_trajectories)


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


def main():
    df : pd.DataFrame
    unique_trajectories : pd.DataFrame

    if os.path.exists(CACHE_DIR):
        df = parquet.read_parquet(os.path.join(CACHE_DIR, "sample_dataframe.parquet"))
        unique_trajectories = parquet.read_parquet(os.path.join(CACHE_DIR, "sample_unique_trajectories.parquet"))
    else:
        os.makedirs(CACHE_DIR)
        get_data()
        traj_df = ()
        df = get_first_x_trajectories(traj_df, 10)
        unique_trajectories = df[df["trajectory_id"].isin(df["trajectory_id"].unique())]
        df.to_parquet(os.path.join(CACHE_DIR, "sample_dataframe.parquet"))
        unique_trajectories.to_parquet(os.path.join(CACHE_DIR, "sample_unique_trajectories.parquet"))


    clustering_method, clustering_param, batch_size, d_model, num_heads, clustering_metric, num_layers = get_best_params()
    raw_trajectories, reference_set,_,_,_ = generate_reference_set(
        df=df, clustering_method=clustering_method, clustering_param=clustering_param,
        batch_size=batch_size, d_model=d_model, num_heads=num_heads, clustering_metric=clustering_metric,
        num_layers=num_layers
    )
    raw_traj : np.ndarray = raw_trajectories.to_numpy()
    ref_set : np.ndarray = reference_set.to_numpy()
    ostc.compress(raw_traj, ref_set)
    




if __name__ == "__main__":
    main()
