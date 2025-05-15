import csv
import faulthandler
import os
import sys
from enum import Enum

import ostc
import pandas as pd
import yaml
from rich.progress import (BarColumn, Progress, TaskProgressColumn, TextColumn,
                           TimeRemainingColumn)
from sklearn.model_selection import ParameterGrid

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from reference_set_construction import generate_reference_set

import tools.scripts._preprocess as _load_data
from Evaluation._file_access_helper_functions import (find_newest_version,
                                                      load_data_from_file,
                                                      save_to_file)
from Evaluation.query_accuracy import query_accuracy_evaluation
from Evaluation.querying import (query_compressed_dataset,
                                 query_original_dataset)

faulthandler.enable()

# Your Enum
class ClusteringMethod(Enum):
    KMEDOIDS = 1
    AGGLOMERATIVE = 2


def custom_scoring(y_pred):
    accuracy, _ = query_accuracy_evaluation(y_true=y_true, y_pred=y_pred, original_df=dataset)
    return accuracy


# Load and prepare data
data = [
            # Beijing Trajectories
            [0, "2008-02-02 15:36:08", 116.51172, 39.92123],  # Trajectory 1
            [0, "2008-02-02 15:40:10", 116.51222, 39.92173],
            [0, "2008-02-02 16:00:00", 116.51372, 39.92323],

            [1, "2008-02-02 14:00:00", 116.50000, 39.90000],  # Trajectory 2
            [1, "2008-02-02 14:15:00", 116.51000, 39.91000],

            [2, "2008-02-02 16:10:00", 116.55000, 39.95000],  # Trajectory 3
            [2, "2008-02-02 16:12:00", 116.55200, 39.95200],

            [3, "2008-02-02 13:30:00", 116.50050, 39.91050],  # Trajectory 4
            [3, "2008-02-02 13:45:00", 116.52050, 39.93050],
            [3, "2008-02-02 14:00:00", 116.54050, 39.95050],

            [4, "2008-02-02 17:10:00", 116.57000, 39.97000],  # Trajectory 5
            [4, "2008-02-02 17:15:00", 116.58000, 39.98000],

            [5, "2008-02-02 18:00:00", 116.59000, 39.99000],  # Trajectory 6
            [5, "2008-02-02 18:05:00", 116.60000, 39.99200],
            [5, "2008-02-02 18:10:00", 116.61000, 39.99300]
        ]

# dataset = pd.DataFrame(data, columns=["trajectory_id", "timestamp", "longitude", "latitude"])

dataset = _load_data.main()
# df, unique_trajectories = get_first_x_trajectories(trajectories=dataset, num_trajectories=10)
unique_trajectories = dataset["trajectory_id"].unique()

queries = load_data_from_file({
    "filename": "queries_for_evaluation",
})

if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../Evaluation/files", f"{find_newest_version()}-original_query_results.pkl")):
    y_true = query_original_dataset(dataset, queries)
else:
    y_true = load_data_from_file({
        "filename": "original_query_results",
    })["data"]



static_keys = ["batch_size", "d_model", "num_heads", "num_layers", "clustering_method", "clustering_metric"]

# Prepare log file
path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../Evaluation", "files"))
param_config_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "grid_search_params.yml"))
log_file = "grid_search_results.csv"
path_log_file = os.path.join(path, log_file)
log_fields = static_keys + ["clustering_param", "score"]

# Load params
with open(param_config_path, "r") as f:
    param_config = yaml.safe_load(f)

# Build static grid (everything except clustering_param)
static_grid = list(ParameterGrid({k: param_config[k] for k in static_keys}))

# Initialize log file
if not os.path.exists(path_log_file):
    with open(path_log_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()

# Run grid search and log
count = 1
total_iterations = sum(len(param_config["clustering_param"][sp["clustering_method"]]) for sp in static_grid)
with Progress(
    TextColumn("[bold blue]{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    TimeRemainingColumn(),
    transient=False,
) as progress:
    # total_task = progress.add_task("[green]Total Progress", total=total_iterations)
    # static_task = progress.add_task("[yellow]Static Grid Params", total=len(static_grid))
    
    for static_params in static_grid:
        method_name = static_params["clustering_method"]
        clustering_method_enum = ClusteringMethod[method_name]
        clustering_params = param_config["clustering_param"][method_name]

        # cluster_task = progress.add_task(f"[cyan]{method_name} configs", total=len(clustering_params))

        for clustering_param in clustering_params:
            # progress.console.print(f"🔍 [bold]Running config:[/] {static_params}, clustering_param: {clustering_param}")
            # progress.console.print(f"🔍 Running iteration {count} of {total_iterations}")

            # try:
            df_out, reference_set, _, _, _, ref_ids = generate_reference_set(
                batch_size=static_params["batch_size"],
                d_model=static_params["d_model"],
                num_heads=static_params["num_heads"],
                num_layers=static_params["num_layers"],
                df=dataset,
                clustering_method=clustering_method_enum,
                clustering_param=clustering_param,
                clustering_metric=static_params["clustering_metric"],
            )

            compressed_data, merged_df, _, _ = ostc.compress(
                df_out.to_records(index=False), 
                reference_set.to_records(index=False),
                ref_ids
            )

            y_pred = query_compressed_dataset(
                compressed_dataset=compressed_data, 
                merged_df=merged_df, 
                queries=queries
            )

            score = custom_scoring(y_pred=y_pred)

            log_row = static_params.copy()
            log_row["clustering_param"] = clustering_param
            log_row["score"] = score

            with open(path_log_file, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=log_fields)
                writer.writerow(log_row)

            count += 1
            progress.update(cluster_task, advance=1)
            progress.update(total_task, advance=1)
            # except Exception as e:
            #     progress.console.print(f"[red]❌ Failed on config {static_params} with clustering_param {clustering_param}: {e}")
        
        progress.update(static_task, advance=1)
        progress.remove_task(cluster_task)
