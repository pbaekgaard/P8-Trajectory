import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import List
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class TimeEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(1, d_model)  # Project time into d_model space
        self.activation = nn.ReLU()  # Optional non-linearity

    def forward(self, t):
        t = t.unsqueeze(-1)  # Shape: (batch, seq_len, 1)
        return self.activation(self.linear(t))  # Output: (batch, seq_len, d_model)


class SpaceEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(2, d_model)  # Project (longitude, latitude) into d_model space
        self.activation = nn.ReLU()

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(coords))  # Output: (batch, seq_len, d_model)


class TrajectoryTransformer(nn.Module):
    def __init__(self, d_model=128, num_heads=4, num_layers=3, output_dim=64):
        super().__init__()

        self.time_emb = TimeEmbedding(d_model)  # Time embedding
        self.spatial_emb = SpaceEmbedding(d_model)  # Spatial embedding

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model * 2, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(d_model * 2, output_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        t_relative = x[:, :, 0]  # Extract time
        coords = x[:, :, 1:]  # Extract (longitude, latitude)

        # Compute embeddings
        t_emb = self.time_emb(t_relative)  # (batch, seq_len, d_model)
        s_emb = self.spatial_emb(coords)  # (batch, seq_len, d_model)

        # Concatenate both embeddings
        input_emb = torch.cat([t_emb, s_emb], dim=-1)  # (batch, seq_len, d_model * 2)

        # Transformer encoding
        encoded = self.transformer_encoder(input_emb, src_key_padding_mask=mask)

        pooled = encoded.mean(dim=1)

        return self.output_proj(pooled)  # (batch_size, 64)


def split_into_batches(df: pd.DataFrame, batch_size: int = 3) -> List[pd.DataFrame]:
    # Ensure the dataframe is sorted by trajectory_id and timestamp
    df = df.sort_values(by=["trajectory_id", "timestamp"])

    # Get the list of unique trajectory_ids
    trajectory_ids = df["trajectory_id"].unique()

    # List to store batches
    batches = []

    # Loop to split into batches of trajectory_ids
    for i in range(0, len(trajectory_ids), batch_size):
        batch_trajectory_ids = trajectory_ids[i:i + batch_size]

        # Get the subset of the original dataframe for this batch
        batch_df = df[df["trajectory_id"].isin(batch_trajectory_ids)]

        # Append the batch DataFrame to the list
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
    """Convert DataFrame into tensor and generate mask for padded values"""
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


def visualize_in_PCA(trajectory_representations: np.ndarray, representative_indices: np.ndarray):
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
    plt.scatter(trajectory_pca[:, 0], trajectory_pca[:, 1], c="blue", label="All Trajectories", alpha=0.6)

    # Plot representative trajectories in red with larger markers
    plt.scatter(
        representative_pca[:, 0], representative_pca[:, 1],
        c="red", label="Representative Trajectories",
        edgecolors="black", s=150, marker="X"
    )

    # Add trajectory indices as labels
    for i, (x, y) in enumerate(trajectory_pca):
        plt.text(x, y, str(i), fontsize=12, ha='right', va='bottom', color='black')

    # Labels and legend
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Trajectory Embeddings Visualized with PCA")
    plt.legend()
    plt.grid(True)
    plt.show()


def normalize_df(df):
    df['t_relative'] = (df['t_relative'] - df['t_relative'].min()) / (df['t_relative'].max() - df['t_relative'].min())
    df['longitude'] = (df['longitude'] - df['longitude'].min()) / (df['longitude'].max() - df['longitude'].min())
    df['latitude'] = (df['latitude'] - df['latitude'].min()) / (df['latitude'].max() - df['latitude'].min())
    return df


if __name__ == "__main__":
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
        [5, "2008-02-02 18:10:00", 116.61000, 39.99300],

        # New York Trajectories
        [6, "2008-02-02 10:00:00", -74.01000, 40.71000],  # Trajectory 7
        [6, "2008-02-02 10:05:00", -74.01200, 40.71200],
        [6, "2008-02-02 10:10:00", -74.01500, 40.71500],

        [7, "2008-02-02 09:00:00", -74.00000, 40.70000],  # Trajectory 8
        [7, "2008-02-02 09:15:00", -74.00500, 40.70500],

        [8, "2008-02-02 11:10:00", -74.03000, 40.73000],  # Trajectory 9
        [8, "2008-02-02 11:12:00", -74.03200, 40.73200],

        [9, "2008-02-02 08:30:00", -74.02000, 40.72000],  # Trajectory 10
        [9, "2008-02-02 08:45:00", -74.02500, 40.72500],
        [9, "2008-02-02 09:00:00", -74.03000, 40.73000],

        [10, "2008-02-02 12:10:00", -74.04000, 40.74000],  # Trajectory 11
        [10, "2008-02-02 12:15:00", -74.04500, 40.74500],

        [11, "2008-02-02 13:00:00", -74.05000, 40.75000],  # Trajectory 12
        [11, "2008-02-02 13:05:00", -74.05500, 40.75200],
        [11, "2008-02-02 13:10:00", -74.06000, 40.75500],

        # random ass bulshiet
        # [12, "2008-02-02 15:36:08", -116.51172, 19.92123],  # Trajectory 1
        # [12, "2008-02-02 15:40:15", -116.51222, 19.92173],
        # [12, "2008-02-02 16:00:05", -116.51372, 19.92323],
        #
        # [13, "2008-02-02 14:00:05", 16.50000, 59.90000],  # Trajectory 2
        # [13, "2008-02-02 14:15:05", 16.51000, 59.91000],
        #
        # [14, "2008-02-02 16:10:05", 40.55000, 9.95000],  # Trajectory 3
        # [14, "2008-02-02 16:12:05", 40.55200, 9.95200],
    ]
    df = pd.DataFrame(data, columns=["trajectory_id", "timestamp", "longitude", "latitude"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['t_relative'] = df.groupby('trajectory_id')['timestamp'].transform(
        lambda x: (x - x.min()).dt.total_seconds())  # convert to delta_seconds from start.
    df = normalize_df(df)


    model = TrajectoryTransformer()
    df_batches = split_into_batches(df, batch_size=3)

    trajectory_representations = []
    for batch in df_batches:
        padded_df = pad_batches(batch)
        batch_tensor, mask_tensor = df_to_tensor(padded_df)
        print("Batch Tensor Shape:", batch_tensor.shape)  # Should be (batch_size, seq_len, 3)
        print("mask tensor:", mask_tensor)
        encoded_output = model(batch_tensor, mask_tensor)
        print("Encoded Representation Shape:", encoded_output.shape)  # Should be (batch_size, 64)
        trajectory_representations.append(encoded_output)

    trajectory_representations = torch.cat(trajectory_representations, dim=0).detach().cpu().numpy()

    print("trajectory_rep: ", trajectory_representations)

    kmedoids = KMedoids(n_clusters=4, metric="euclidean")
    cluster_labels = kmedoids.fit_predict(trajectory_representations)

    representative_indices = kmedoids.medoid_indices_
    print("Representative Trajectories Indices:", representative_indices)

    representative_trajectories = df.iloc[representative_indices]
    print("Representative Trajectories:\n", representative_trajectories)

    visualize_in_PCA(trajectory_representations, representative_indices)
