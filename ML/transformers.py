import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import tools.scripts._get_data as _get_data
import tools.scripts._load_data as _load_data
from sklearn_extra.cluster import KMedoids  # For K-Medoids clustering

# Check if GPU is available
# Step 1: Detect available device (CUDA for Windows/Linux with NVIDIA GPU, MPS for macOS M1/M2, or CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use CUDA (NVIDIA GPU)
    print("Using CUDA (NVIDIA GPU)")
elif torch.backends.mps.is_available():
    device = torch.device("mps")   # Use MPS (Apple M1/M2 GPU)
    print("Using MPS (Apple GPU)")
else:
    device = torch.device("cpu")   # Fallback to CPU
    print("Using CPU")

### 1. Sample Trajectories Generation
def generate_sample_trajectories():
    """
    Generate sample trajectories with some being close and others farther apart.
    Each trajectory consists of (latitude, longitude, time).
    :return: A list of trajectories (lat, lon, time)
    """
    # Trajectories that are close together (clustered around San Francisco)
    traj0 = [(37.7749, -122.4194, 0), (37.7750, -122.4195, 1), (37.7751, -122.4196, 2)]
    traj1 = [(37.7752, -122.4197, 0), (37.7753, -122.4198, 1), (37.7754, -122.4199, 2)]
    traj4 = [(34.0522, -118.2437, 0), (34.0523, -118.2438, 1), (34.0524, -118.2439, 2)]  # Los Angeles

    # Trajectories that are close together (clustered around New York)
    traj2 = [(40.7128, -74.0060, 0), (40.7129, -74.0061, 1), (40.7130, -74.0062, 2)]
    traj3 = [(40.7131, -74.0063, 0), (40.7132, -74.0064, 1), (40.7133, -74.0065, 2)]

    # Farther apart trajectories (random locations)
    traj5 = [(51.5074, -0.1278, 0), (51.5075, -0.1279, 1), (51.5076, -0.1280, 2)]  # London

    traj6 = [(35.6895, 139.6917, 0), (35.6896, 139.6918, 1), (35.6897, 139.6919, 2)]  # Tokyo
    traj7 = [(35.6895, 139.6917, 3000), (35.6896, 139.6918, 6000), (35.6897, 139.6919, 9000)]  # Tokyo

    return [traj0, traj1, traj2, traj3, traj4, traj5, traj6, traj7]

### 2. Preprocess the Trajectories
def preprocess_trajectory(trajectories, max_len):
    feature_vectors = []
    for traj in trajectories:
        features = []
        for i, (lat, lon, t) in enumerate(traj):
            """
            if i > 0:
                #time_delta = t - traj[i - 1][2]
            else:
                time_delta = 0
            features.append([lat, lon, time_delta])
            """
            features.append([lat, lon])

        while len(features) < max_len:
            features.append([0.0, 0.0, 0.0])  # Padding with zeros
        feature_vectors.append(features)

    return torch.tensor(feature_vectors, dtype=torch.float32)

def normalize_df(df):
    df['longitude'] = (df['longitude'] - df['longitude'].min()) / (df['longitude'].max() - df['longitude'].min())
    df['latitude'] = (df['latitude'] - df['latitude'].min()) / (df['latitude'].max() - df['latitude'].min())
    return df

def select_and_process_trajectories(df: pd.DataFrame, x: int, max_len: int) -> tuple:
    """
    Select x trajectories from a DataFrame and convert them into a tensor of shape (x, max_len, 3).
    Each trajectory is processed into feature vectors [latitude, longitude, time_delta],
    padded or truncated to max_len.
    """
    #df = normalize_df(df)
    unique_ids = df['trajectory_id'].unique()
    selected_ids = unique_ids[:x]

    groups = df.groupby('trajectory_id')
    feature_vectors = []
    padding_masks = []
    original_selected_ids = []

    for traj_id in selected_ids:
        traj_df = groups.get_group(traj_id).sort_values('timestamp')
        traj_df['timestamp'] = pd.to_datetime(traj_df['timestamp'])
        #traj_df['time_delta'] = (traj_df['timestamp'] - traj_df['timestamp'].shift(1)).dt.total_seconds().fillna(0)
        #features = traj_df[['latitude', 'longitude', 'time_delta']].values.tolist()
        features = traj_df[['latitude', 'longitude']].values.tolist()

        # Create a mask: False for valid data, True for padding
        traj_len = len(features)
        mask = [False] * traj_len + [True] * (max_len - traj_len) if traj_len < max_len else [False] * max_len

        if len(features) > max_len:
            features = features[:max_len]
        else:
            while len(features) < max_len:
                features.append([0.0, 0.0])

        print("traj id", traj_id)
        print(features)
        feature_vectors.append(features)
        padding_masks.append(mask)
        original_selected_ids.append(traj_id)

    # Duplicate the trajectories
    final_feature_vectors = feature_vectors + feature_vectors  # Duplicate the features
    final_padding_masks = padding_masks + padding_masks  # Duplicate the masks
    final_selected_ids = original_selected_ids + original_selected_ids  # Duplicate the IDs

    tensor = torch.tensor(final_feature_vectors, dtype=torch.float32)
    padding_mask = torch.tensor(final_padding_masks, dtype=torch.bool)
    return tensor, padding_mask, final_selected_ids

### 3. Positional Encoding and Transformer Model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :]

class TrajectoryTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, max_len):
        super(TrajectoryTransformer, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model).to(device)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len).to(device)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True).to(device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers).to(device)
        self.output_layer = nn.Linear(d_model, d_model).to(device)

    def forward(self, src, src_key_padding_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        trajectory_embeddings = self.output_layer(output)

        # Mask the padded positions before pooling
        if src_key_padding_mask is not None:
            # Expand the mask to match the embedding dimensions
            mask = src_key_padding_mask.unsqueeze(-1).expand(-1, -1, self.d_model)
            # Zero out the embeddings of padded positions
            trajectory_embeddings = trajectory_embeddings.masked_fill(mask, 0.0)
            # Compute the mean only over valid positions
            valid_counts = (~src_key_padding_mask).float().unsqueeze(-1)  # Number of valid positions
            sum_embeddings = trajectory_embeddings.sum(dim=1)  # Sum over sequence dimension
            pooled_embedding = sum_embeddings / (valid_counts.sum(dim=1) + 1e-8)  # Avoid division by zero
        else:
            # If no mask, compute the mean as before
            pooled_embedding = torch.mean(trajectory_embeddings, dim=1)

        return pooled_embedding

def plot_embeddings(methodName, reduced_embeddings, cluster_labels, selected_ids):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=cluster_labels, cmap='viridis', s=100)
    for i, txt in enumerate(selected_ids):
        plt.annotate(str(txt), (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=12)
    plt.title(f"2D PCA of Transformer Embeddings with {methodName} Clusters")
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid(True)
    plt.show()

# Instantiate the transformer model
input_dim = 2
d_model = 64
nhead = 4
num_layers = 2
traj_to_select = 10
max_len = 1

start_event = torch.mps.Event(enable_timing=True)
end_event = torch.mps.Event(enable_timing=True)

start_event.record()

_get_data.main()
traj_df = _load_data.main()

end_event.record()
torch.mps.synchronize()
print(f"Load data time: {start_event.elapsed_time(end_event):.3f} ms")

start_event.record()

# Generate and preprocess trajectories
trajectories, padding_mask, selected_ids = select_and_process_trajectories(traj_df, x=traj_to_select, max_len=max_len)
#sample_trajectories = generate_sample_trajectories()
#selected_ids = range(0, len(sample_trajectories))
#trajectories = preprocess_trajectory(sample_trajectories, max_len).to(device)

input_features = trajectories.to(device)  # Use the raw trajectories for feature extraction
padding_mask = padding_mask.to(device)

end_event.record()
torch.mps.synchronize()
print(f"Select and process traj time: {start_event.elapsed_time(end_event):.3f} ms")

start_event.record()

model = TrajectoryTransformer(input_dim, d_model, nhead, num_layers, max_len).to(device)

end_event.record()
torch.mps.synchronize()
print(f"Instantiate model time: {start_event.elapsed_time(end_event):.3f} ms")


start_event.record()

### 4. Run Through Transformer to Get Embeddings
output_embeddings = model(input_features).detach().cpu().numpy()
print(f"Output embeddings shape: {output_embeddings.shape}")  # (batch_size, d_model)

end_event.record()
torch.mps.synchronize()
print(f"Inference time: {start_event.elapsed_time(end_event):.3f} ms")

start_event.record()

### 5. Apply PCA to Reduce to 2D
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(output_embeddings)
#reduced_embeddings = MDS(n_components=2, dissimilarity="euclidean", random_state=42).fit_transform(output_embeddings)
print(f"Reduced embeddings shape: {reduced_embeddings.shape}")  # (batch_size, 2)

end_event.record()
torch.mps.synchronize()
print(f"PCA time: {start_event.elapsed_time(end_event):.3f} ms")

start_event.record()
### 7. Apply K-Medoids Clustering
n_clusters = 3
#kmeans_cluster_labels = KMeans(n_clusters=n_clusters).fit_predict(reduced_embeddings)
kmedoids_cluster_labels = KMedoids(n_clusters=n_clusters, metric="euclidean", random_state=42).fit_predict(output_embeddings)
#agglomerative_cluster_labels = AgglomerativeClustering(n_clusters=None, metric="euclidean", linkage="complete", distance_threshold=1.0).fit_predict(reduced_embeddings)

#print("K-Means cluster labels:", kmeans_cluster_labels)
print("K-Medoids cluster labels:", kmedoids_cluster_labels)
#print("Agglomerative cluster labels:", agglomerative_cluster_labels)

end_event.record()
torch.mps.synchronize()
print(f"Clustering time: {start_event.elapsed_time(end_event):.3f} ms")


#plot_embeddings("K-Means", reduced_embeddings, kmeans_cluster_labels, selected_ids)
plot_embeddings("K-Medoids", reduced_embeddings, kmedoids_cluster_labels, selected_ids)
#plot_embeddings("Agglomerative", reduced_embeddings, agglomerative_cluster_labels, selected_ids)

