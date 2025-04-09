import yaml
import csv
from sklearn.model_selection import ParameterGrid
from enum import Enum
import faulthandler
import os

from ML.reference_set_construction import generate_reference_set, get_first_x_trajectories
import tools.scripts._get_data as _get_data
import tools.scripts._load_data as _load_data


faulthandler.enable()

# Your Enum
class ClusteringMethod(Enum):
    KMEDOIDS = 1
    AGGLOMERATIVE = 2


def calculate_score():
    #DUMMY
    return 42


# Load and prepare data
_get_data.main()
traj_df = _load_data.main()
df, unique_trajectories = get_first_x_trajectories(trajectories=traj_df, num_trajectories=10)

# Load params
with open("params.yml", "r") as f:
    param_config = yaml.safe_load(f)

# Build static grid (everything except clustering_param)
static_keys = ["batch_size", "d_model", "num_heads", "num_layers", "clustering_method", "clustering_metric"]
static_grid = list(ParameterGrid({k: param_config[k] for k in static_keys}))

# Prepare log file
log_file = "grid_search_results.csv"
log_fields = static_keys + ["clustering_param", "score"]

# Initialize log file
if not os.path.exists(log_file):
    with open(log_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()

# Run grid search and log
for static_params in static_grid:
    method_name = static_params["clustering_method"]
    clustering_method_enum = ClusteringMethod[method_name]
    clustering_params = param_config["clustering_param"][method_name]

    for clustering_param in clustering_params:
        print(f"🔍 Running config: {static_params}, clustering_param: {clustering_param}")

        try:
            df_out, representative_trajectories, reference_set, representative_indices, trajectory_representations = generate_reference_set(
                batch_size=static_params["batch_size"],
                d_model=static_params["d_model"],
                num_heads=static_params["num_heads"],
                num_layers=static_params["num_layers"],
                df=df,
                clustering_method=clustering_method_enum,
                clustering_param=clustering_param,
                clustering_metric=static_params["clustering_metric"],
                unique_trajectories=unique_trajectories
            )

            score = calculate_score
            # Log this config
            log_row = static_params.copy()
            log_row["clustering_param"] = clustering_param
            log_row["score"] = score

            with open(log_file, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=log_fields)
                writer.writerow(log_row)

        except Exception as e:
            print(f"❌ Failed on config {static_params} with clustering_param {clustering_param}: {e}")
