import os
import sys
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids

sys.path.append(os.path.dirname(os.path.abspath(__file__ + "/../../../")))
import faulthandler
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum

import tools.scripts._preprocess as _load_data
from src.components.ML.TrajectoryTransformer import TrajectoryTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ClusteringMethod(Enum):
    KMEDOIDS = 1
    AGGLOMERATIVE = 2

#TODO: CLEANUP herinde. vi behøver f.eks ikke unique_trajectories. get first_x er kun midlertidig.



def split_into_batches(df: pd.DataFrame, batch_size: int = 3) -> List[pd.DataFrame]:
    df = df.sort_values(by=["trajectory_id", "timestamp"])
    trajectory_ids = df["trajectory_id"].unique()
    grouped = df.groupby("trajectory_id")

    batches = []

    for i in range(0, len(trajectory_ids), batch_size):
        batch_trajectory_ids = trajectory_ids[i:i + batch_size]

        batch_df = pd.concat([grouped.get_group(trajectory_id) for trajectory_id in batch_trajectory_ids])

        batches.append(batch_df)

    return batches


def pad_trajectory(group: pd.DataFrame, max_length: int) -> pd.DataFrame:
    """Pads a single trajectory group to match max_length."""
    missing_entries = max_length - len(group)
    if missing_entries > 0:
        missing_rows = pd.DataFrame({
            "trajectory_id": [group["trajectory_id"].iloc[0]] * missing_entries,
            "timestamp": [pd.NaT] * missing_entries,
            "longitude": [np.nan] * missing_entries,
            "latitude": [np.nan] * missing_entries,
            "t_relative": [np.nan] * missing_entries
        })
        group = pd.concat([group, missing_rows], ignore_index=True)
    return group


def pad_batches(df: pd.DataFrame) -> pd.DataFrame:
    """Pads all trajectories in the DataFrame to match the longest trajectory."""
    trajectory_lengths = df.groupby("trajectory_id").size()
    longest_trajectory = trajectory_lengths.max()

    return df.groupby("trajectory_id", group_keys=False).apply(lambda g: pad_trajectory(g, longest_trajectory))


def df_to_tensor(df: pd.DataFrame):
    """
    :param df: DataFrame to be converted to tensor.
    :returns: A tuple containing two elements:
        - batch_tensor: Tensor for the whole batch.
        - mask_tensor: Tensor of the same dimensions as batch_tensor.
          Boolean values indicating whether a value is a mask or not.

    Convert DataFrame into tensor and generate a mask for padded values.
    """

    df = df.sort_values(["trajectory_id", "t_relative"])
    grouped = df.groupby("trajectory_id")
    batch_tensors = []
    masks = []

    for _, group in grouped:
        features = group[["t_relative", "longitude", "latitude"]].fillna(0.0).values
        tensor = torch.tensor(features, dtype=torch.float32)
        batch_tensors.append(tensor)

        mask = (tensor == 0.0).all(dim=-1)
        masks.append(mask)

    batch_tensor = torch.stack(batch_tensors)
    mask_tensor = torch.stack(masks)  # Shape: (batch, seq_len)
    return batch_tensor, mask_tensor


def visualize_in_PCA(df, trajectory_representations: np.ndarray, representative_indices: np.ndarray, reference_set: List, clusteringMethod: ClusteringMethod):
    """
    Visualizes the trajectory embeddings in 2D using PCA and labels them with their indices.

    Args:
        trajectory_representations (np.ndarray): The encoded trajectory representations (shape: [num_trajectories, embedding_dim]).
        representative_indices (np.ndarray): Indices of representative trajectories (medoids).
    """
    # Apply PCA to reduce embeddings from high-dimensional to 2D
    pca = PCA(n_components=2)
    trajectory_pca = pca.fit_transform(trajectory_representations)

    # Extract representative trajectories' positions
    representative_pca = trajectory_pca[representative_indices]

    plt.figure(figsize=(10, 7))

    # Plot all trajectories in light blue
    plt.scatter(trajectory_pca[:, 0], trajectory_pca[:, 1], c=reference_set, label="All Trajectories", alpha=0.6)

    # Plot representative trajectories in red with larger markers
    plt.scatter(
        representative_pca[:, 0], representative_pca[:, 1],
        c=representative_indices, label="Representative Trajectories",
        edgecolors="black", s=150, marker="X"
    )

    # Add trajectory indices as labels
    for i, (x, y) in enumerate(trajectory_pca):
        plt.text(x, y, str(i) + ":" + str(df['trajectory_id'].unique()[i]), fontsize=10, ha='right', va='bottom', color='black')

    # Labels and legend
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Trajectory Embeddings Visualized with PCA using " + clusteringMethod.name)
    plt.legend()
    plt.grid(True)
    plt.show()


def normalize_df(df):
    """
    :param df: DataFrame to be normalized.
    :return: normalized Dataframe.
    normalizes according to highest and lowest value for that column in entire df. 1 is max, 0 is min.
    """
    norm_df = df.copy()
    norm_df['t_relative'] = (df['t_relative'] - df['t_relative'].min()) / (df['t_relative'].max() - df['t_relative'].min())
    norm_df['longitude'] = (df['longitude'] - df['longitude'].min()) / (df['longitude'].max() - df['longitude'].min())
    norm_df['latitude'] = (df['latitude'] - df['latitude'].min()) / (df['latitude'].max() - df['latitude'].min())
    return norm_df


def get_first_x_trajectories(trajectories: pd.DataFrame, num_trajectories: int = None) -> pd.DataFrame:
    """
    :param num_trajectories: how many trajectories you want
    :param trajectories: from where to get the trajectories
    :returns: tuple consisting of:
        - dataframe of the selected trajectories
        - unique_trajectories: lookup table for the selected trajectories
    selects the first x (num_trajectories) trajectories with a unique trajectory_id.
    """
    if num_trajectories == None:
        unique_trajectories = trajectories['trajectory_id'].unique()
    else:
        unique_trajectories = trajectories['trajectory_id'].unique()[:num_trajectories]
    df = trajectories[trajectories['trajectory_id'].isin(unique_trajectories)]

    return df

def generate_reference_set(df: pd.DataFrame, clustering_method: ClusteringMethod, clustering_param: int | float, batch_size: int, d_model: int, num_heads: int, clustering_metric: str, num_layers: int) -> (pd.DataFrame, pd.DataFrame, List, List, List):
    df = pd.DataFrame(df, columns=["trajectory_id", "timestamp", "longitude", "latitude"])
    df['t_relative'] = df.groupby('trajectory_id')['timestamp'].transform(
        lambda x: x - x.min()
    )  # convert to delta_seconds from start.
    normalized_df = normalize_df(df)

    model = TrajectoryTransformer(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
    ).to(device)
    # model.train() # IF TRAIN
    model.eval()

    df_batches = split_into_batches(normalized_df, batch_size=batch_size)
    trajectory_tensors = []

    def process_batch(batch):
        padded_df = pad_batches(batch)
        batch_tensor, mask_tensor = df_to_tensor(padded_df)
        batch_tensor = batch_tensor.to(device)
        mask_tensor = mask_tensor.to(device)

        with torch.no_grad():
            encoded_output = model.forward(batch_tensor, mask_tensor)

        return encoded_output.detach().cpu()

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_batch, batch) for batch in df_batches]
        for future in as_completed(futures):
            trajectory_tensors.append(future.result())


    trajectory_tensors = torch.cat(trajectory_tensors, dim=0).numpy()

    clustering = None
    cluster_labels = None
    representative_indices = []

    match clustering_method.value:
        case ClusteringMethod.KMEDOIDS.value:
            clustering = KMedoids(n_clusters=clustering_param, metric=clustering_metric)
            cluster_labels = clustering.fit_predict(trajectory_tensors)
            representative_indices = clustering.medoid_indices_
            
        case ClusteringMethod.AGGLOMERATIVE.value:
            clustering = AgglomerativeClustering(
                n_clusters=None, metric=clustering_metric, linkage="complete", distance_threshold=clustering_param
            )
            cluster_labels = clustering.fit_predict(trajectory_tensors)

            for cluster_id in np.unique(cluster_labels):
                cluster_points = trajectory_tensors[cluster_labels == cluster_id]

                centroid = np.mean(cluster_points, axis=0)
                distances = cdist([centroid], cluster_points, metric="euclidean")

                closest_index = np.argmin(distances)
                original_index = np.where(cluster_labels == cluster_id)[0][closest_index]

                representative_indices.append(original_index)

    ref_ids_list = []
    reference_set = []
    for cluster_label in cluster_labels:
        reference_set.append(representative_indices[cluster_label]) # ref set links to medoid ID.
        ref_ids_list.append(df['trajectory_id'].unique()[representative_indices[cluster_label]]) # ref set links to trajID

    rep_ids = df['trajectory_id'].unique()[representative_indices]
    ref_ids_dict = dict(zip(df['trajectory_id'].unique(), ref_ids_list))
    mask = np.isin(df['trajectory_id'].values, rep_ids)
    representative_trajectories = df.loc[mask]
    df = df.loc[~mask]


    return df, representative_trajectories, reference_set, representative_indices, trajectory_tensors, ref_ids_dict


if __name__ == "__main__":
    faulthandler.enable()  # så kan vi se, hvis vi løber tør for memory
    batch_size = 5
    clusteringMethod = ClusteringMethod.KMEDOIDS
    n_clusters = 3
    distance_threshold = 0.25
    clustering_metric = "euclidean"


    all_df = _load_data.main()

    df, representative_trajectories, reference_set, representative_indices, trajectory_representations = generate_reference_set(
        batch_size=batch_size,
        d_model=128,
        num_heads=4,
        num_layers=3,
        df=all_df,
        clustering_method=clusteringMethod,
        clustering_param=n_clusters if clusteringMethod == ClusteringMethod.KMEDOIDS else distance_threshold,
        clustering_metric=clustering_metric
    )

    visualize_in_PCA(all_df, trajectory_representations, representative_indices, reference_set, clusteringMethod)

    """
    HELP FOR NOOBS:
    reference_set[0]  # hvilket cluster tilhører jeg
    unique_trajectories[0]  # hvad er mit trajectory_id
    df[df['trajectory_id'] == unique_trajectories[0]]  # alle mine punkter
    """







