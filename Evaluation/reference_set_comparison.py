import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/components")))

from ML.reference_set_construction import generate_reference_set, visualize_in_PCA
from _file_access_helper_functions import get_best_params

from tools.scripts._preprocess import main as _load_data
from tools.scripts._delimit_trajectory_length import main as delimit_trajectory_length

def compute_similarity_of_reference_sets(reference_set_one, reference_set_two):
    # This only counts if the numbers are equal, but does not consider if the same clusters were formed but with different centroids.
    # I don't know if it actually makes sense if it does that tho.
    zipped_list = list(zip(reference_set_one, reference_set_two))



    return sum(1 for a, b in zipped_list if a == b) / len(zipped_list)



def reference_set_comparison():
    print('Reference Set Comparison')
    full_dataset = _load_data(
        [   'deduplication',
            'timestamporder',
            'limit_samplerate',
            'remove_illegal',
            'convert_timestamp_to_unix',
            'take_10_trajectories', # Remove if you want every trajectory
        ]
    )

    length = 10
    limited_dataset = delimit_trajectory_length(full_dataset, length)

    print(len(full_dataset))
    print(len(limited_dataset))

    clustering_method, clustering_param, batch_size, d_model, num_heads, clustering_metric, num_layers = get_best_params()
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

    visualize_in_PCA(full_df, full_trajectory_representations, full_representative_indices, full_reference_set, clustering_method, "Full")
    visualize_in_PCA(limited_df, limited_trajectory_representations, limited_representative_indices, limited_reference_set, clustering_method, "Limited")

    full_reference_set = [x + 1 for x in full_reference_set]
    limited_reference_set = [x + 1 for x in limited_reference_set]

    print("Full reference set ", full_reference_set)
    print("Limited reference set ", limited_reference_set)

    similarity_score = compute_similarity_of_reference_sets(full_reference_set, limited_reference_set)

    print(similarity_score)

if __name__ == '__main__':
    reference_set_comparison()