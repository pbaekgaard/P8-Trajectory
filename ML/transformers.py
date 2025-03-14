import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import tools.scripts._get_data as _get_data
import tools.scripts._load_data as _load_data
from sklearn_extra.cluster import KMedoids  # For K-Medoids clustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score

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

    # Trajectories that are close together (clustered around New York)
    traj2 = [(40.7128, -74.0060, 0), (40.7129, -74.0061, 1), (40.7130, -74.0062, 2)]
    traj3 = [(40.7131, -74.0063, 0), (40.7132, -74.0064, 1), (40.7133, -74.0065, 2)]

    # Farther apart trajectories (random locations)
    traj4 = [(34.0522, -118.2437, 0), (34.0523, -118.2438, 1), (34.0524, -118.2439, 2)]  # Los Angeles
    traj5 = [(51.5074, -0.1278, 0), (51.5075, -0.1279, 1), (51.5076, -0.1280, 2)]  # London
    traj6 = [(35.6895, 139.6917, 0), (35.6896, 139.6918, 1), (35.6897, 139.6919, 2)]  # Tokyo

    return [traj0, traj1, traj2, traj3, traj4, traj5, traj6]

### 2. Preprocess the Trajectories
def preprocess_trajectory(trajectories, max_len):
    feature_vectors = []
    for traj in trajectories:
        features = []
        for i, (lat, lon, t) in enumerate(traj):
            if i > 0:
                time_delta = t - traj[i - 1][2]
            else:
                time_delta = 0
            features.append([lat, lon, time_delta])

        while len(features) < max_len:
            features.append([0.0, 0.0, 0.0])  # Padding with zeros
        feature_vectors.append(features)

    return torch.tensor(feature_vectors, dtype=torch.float32)

def select_and_process_trajectories(df: pd.DataFrame, x: int, max_len: int) -> tuple:
    """
    Select x trajectories from a DataFrame and convert them into a tensor of shape (x, max_len, 3).
    Each trajectory is processed into feature vectors [latitude, longitude, time_delta],
    padded or truncated to max_len.
    """
    unique_ids = df['trajectory_id'].unique()
    selected_ids = unique_ids[:x]

    groups = df.groupby('trajectory_id')
    feature_vectors = []

    for traj_id in selected_ids:
        traj_df = groups.get_group(traj_id).sort_values('timestamp')
        traj_df['timestamp'] = pd.to_datetime(traj_df['timestamp'])
        traj_df['time_delta'] = (traj_df['timestamp'] - traj_df['timestamp'].shift(1)).dt.total_seconds().fillna(0)
        features = traj_df[['latitude', 'longitude', 'time_delta']].values.tolist()

        if len(features) > max_len:
            features = features[:max_len]
        else:
            while len(features) < max_len:
                features.append([0.0, 0.0, 0.0])
        print("id ", traj_id)
        print(features)
        feature_vectors.append(features)

    tensor = torch.tensor(feature_vectors, dtype=torch.float32)
    return tensor, selected_ids

### 3. Positional Encoding and Transformer Model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
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
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, d_model)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        trajectory_embeddings = self.output_layer(output)
        pooled_embedding = torch.mean(trajectory_embeddings, dim=1)
        return pooled_embedding

# Instantiate the transformer model
input_dim = 3
d_model = 64
nhead = 4
num_layers = 2
traj_to_select = 5
max_len = 3

_get_data.main()
traj_df = _load_data.main()

# Generate and preprocess trajectories
#trajectories, selected_ids = select_and_process_trajectories(traj_df, x=traj_to_select, max_len=max_len)
sample_trajectories = generate_sample_trajectories()
selected_ids = range(0, len(sample_trajectories))
trajectories = preprocess_trajectory(sample_trajectories, max_len)

input_features = trajectories  # Use the raw trajectories for feature extraction

model = TrajectoryTransformer(input_dim, d_model, nhead, num_layers, max_len)

### 4. Run Through Transformer to Get Embeddings
output_embeddings = model(input_features).detach().numpy()
print(f"Output embeddings shape: {output_embeddings.shape}")  # (batch_size, d_model)

### 5. Apply PCA to Reduce to 2D
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(output_embeddings)
print(f"Reduced embeddings shape: {reduced_embeddings.shape}")  # (batch_size, 2)

### 6. Plot 1: PCA Before K-Medoids Clustering
plt.figure(figsize=(8, 6))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c='blue', s=100)
for i, txt in enumerate(selected_ids):
    plt.annotate(str(txt), (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=12)
plt.title('2D PCA of Transformer Embeddings (Before Clustering)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid(True)
plt.show()

### 7. Apply K-Medoids Clustering
n_clusters = 3
kmedoids = KMedoids(n_clusters=n_clusters, random_state=42)
cluster_labels = kmedoids.fit_predict(output_embeddings)
print("K-Medoids cluster labels:", cluster_labels)

### 8. Plot 2: PCA After K-Medoids Clustering
plt.figure(figsize=(8, 6))
scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=cluster_labels, cmap='viridis', s=100)
for i, txt in enumerate(selected_ids):
    plt.annotate(str(txt), (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=12)
plt.title('2D PCA of Transformer Embeddings with K-Medoids Clusters')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid(True)
plt.colorbar(scatter, label='Cluster Label')
plt.show()

### 9. Evaluate Clustering Quality
silhouette_avg = silhouette_score(output_embeddings, cluster_labels)
calinski_harabasz = calinski_harabasz_score(output_embeddings, cluster_labels)
print(f"Silhouette Score: {silhouette_avg}")
print(f"Calinski-Harabasz Index: {calinski_harabasz}")