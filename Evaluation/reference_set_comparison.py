import sys
import os
import csv

from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/components")))

from ML.reference_set_construction import generate_reference_set, visualize_in_PCA
from _file_access_helper_functions import get_best_params

from tools.scripts._preprocess import main as _load_data
from tools.scripts._delimit_trajectory_length import main as delimit_trajectory_length

def compute_deviation_of_likeliness_of_traj_in_ref_set(likeliness_ref_set_one: dict, likeliness_ref_set_two: dict):
    # Compute the deviation of the likeliness that a trajectory appears a reference set
    total_deviation = 0

    for key, value in likeliness_ref_set_one.items():
        if key not in likeliness_ref_set_two:
            total_deviation += value
        else:
            total_deviation += abs(value - likeliness_ref_set_two[key])

    for key, value in likeliness_ref_set_two.items():
        if key not in likeliness_ref_set_one:
            total_deviation += value

    normalized_deviation = total_deviation / len(set(likeliness_ref_set_one.keys()).union(set(likeliness_ref_set_two.keys())))
    return normalized_deviation

def similarity_comparison(full_representative_indices, limited_representative_indices):
    matches = len(set(full_representative_indices).intersection(limited_representative_indices))

    return matches / max(len(full_representative_indices), len(limited_representative_indices))

def likeliness_comparison(full_representative_indices, limited_representative_indices):
    return full_representative_indices, limited_representative_indices

def reference_set_comparison(length, ref_set_size):
    print('Reference Set Comparison')
    full_dataset = _load_data(
        [   'deduplication',
            'timestamporder',
            'limit_samplerate',
            'remove_illegal',
            'convert_timestamp_to_unix',
        ]
    )
    # full_dataset = full_dataset[full_dataset["trajectory_id"].isin(full_dataset["trajectory_id"].drop_duplicates().iloc[:500])]
    limited_dataset = delimit_trajectory_length(full_dataset, length)

    iterations = 10
    similarities = []
    likeliness_ref_set_one = {}
    likeliness_ref_set_two = {}

    for i in range(iterations):
        clustering_method, clustering_param, batch_size, d_model, num_heads, clustering_metric, num_layers = get_best_params()
        if clustering_method.name == "KMEDOIDS":
            clustering_param = ref_set_size

        full_df, _, full_reference_set, full_representative_indices, full_trajectory_representations = generate_reference_set(
            df=full_dataset, clustering_method=clustering_method, clustering_param=clustering_param,
            batch_size=batch_size, d_model=d_model, num_heads=num_heads, clustering_metric=clustering_metric,
            num_layers=num_layers
        )

        limited_df, _, limited_reference_set, limited_representative_indices, limited_trajectory_representations = generate_reference_set(
            df=limited_dataset, clustering_method=clustering_method, clustering_param=clustering_param,
            batch_size=batch_size, d_model=d_model, num_heads=num_heads, clustering_metric=clustering_metric,
            num_layers=num_layers
        )

        # visualize_in_PCA(full_df, full_trajectory_representations, full_representative_indices, full_reference_set,
        #                  clustering_method, "Full")
        # visualize_in_PCA(limited_df, limited_trajectory_representations, limited_representative_indices,
        #                  limited_reference_set, clustering_method, "Limited")

        similarities.append(similarity_comparison(full_representative_indices, limited_representative_indices))
        for value in full_representative_indices:
            if value in likeliness_ref_set_one:
                likeliness_ref_set_one[value].append(1)  # 1 represents that it was in the reference set here
            else:
                likeliness_ref_set_one[value] = [1]  # 1 represents that it was in the reference set here

        for value in limited_representative_indices:
            if value in likeliness_ref_set_two:
                likeliness_ref_set_two[value].append(1) # 1 represents that it was in the reference set here
            else:
                likeliness_ref_set_two[value] = [1] # 1 represents that it was in the reference set here

    similarity = sum(similarities) / iterations

    likeliness_ref_set_one = {key: sum(value) / iterations for key, value in likeliness_ref_set_one.items()}
    likeliness_ref_set_two = {key: sum(value) / iterations for key, value in likeliness_ref_set_two.items()}
    deviation = compute_deviation_of_likeliness_of_traj_in_ref_set(likeliness_ref_set_one, likeliness_ref_set_two)

    return similarity, deviation


if __name__ == '__main__':
    lengths = [10, 50, 100]
    ref_set_sizes = [5, 25, 50]
    results = {}
    file_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "files", "ref_set_comparison.csv"))
    fields = ["length", "ref_set_size", "similarity", "deviation"]

    # Initialize log file
    if not os.path.exists(file_path):
        with open(file_path, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

    lengths_and_ref_set_size = [(length, ref_set_size) for length in lengths for ref_set_size in ref_set_sizes]
    for length, ref_set_size in lengths_and_ref_set_size:
        similarity, deviation = reference_set_comparison(length, ref_set_size)
        results["length"] = length
        results["ref_set_size"] = ref_set_size
        results["similarity"] = similarity
        results["deviation"] = deviation
        with open(file_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writerow(results)




