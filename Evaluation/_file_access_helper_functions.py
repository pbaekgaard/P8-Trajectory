import csv
import os
import pickle
from enum import Enum


class ClusteringMethod(Enum):
    KMEDOIDS = 1
    AGGLOMERATIVE = 2

def save_to_file(metadata, data):
    filename = metadata["filename"]
    version = metadata["version"]
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "files"), exist_ok=True)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "files", f"{version}-{filename}.pkl"), "wb") as f:
        pickle.dump(data, f)


def load_data_from_file(metadata):
    filename = metadata["filename"]
    if "version" in metadata:
        version = metadata["version"]
    else:
        version = find_newest_version()
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "files", f"{version}-{filename}.pkl"), "rb") as f:
        data = pickle.load(f)
    return data

def find_newest_version():
    Dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "files"))
    files = [f for f in os.listdir(Dir) if os.path.isfile(os.path.join(Dir, f)) and f.endswith('.pkl')]
    version_number = 0
    for file in files:
        split_file = file.split('-')
        if len(split_file) > 1:
            version_number = max(int(split_file[0]), version_number)

    return version_number


def get_best_params():
    Dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "files"))
    log_file = os.path.join(Dir, "grid_search_results.csv")

    best_row = None
    best_score = float('-inf')

    with open(log_file, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                score = float(row["score"])
                if score > best_score:
                    best_score = score
                    best_row = row
            except ValueError:
                continue  # Skip rows with invalid scores

    if best_row is None:
        raise ValueError("No valid rows with scores found.")

    # Extract and return the desired parameters
    return (
        ClusteringMethod[best_row["clustering_method"]],
        int(best_row["clustering_param"]),
        int(best_row["batch_size"]),
        int(best_row["d_model"]),
        int(best_row["num_heads"]),
        best_row["clustering_metric"],
        int(best_row["num_layers"]),
    )
