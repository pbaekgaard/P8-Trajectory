import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.manifold import MDS
from sklearn.cluster import AgglomerativeClustering



# Example taxi trajectories with timestamps
trajectories = [
    np.array([[0, 0, 0], [1, 2, 10], [2, 4, 20], [3, 6, 30], [4, 8, 40]]),  # Every 10 sec
    np.array([[0, 0, 0], [1, 1.5, 8], [2, 3.5, 18], [3, 5, 28], [4, 7.5, 38]]),  # Slight time shift
    np.array([[0, 0, 0], [1, 1, 5], [2, 1, 10], [3, 2, 15], [4, 3, 20]])  # Different time intervals
]

# Convert timestamps to seconds since first observation
for traj in trajectories:
    traj[:, 2] -= traj[0, 2]  # Normalize time so first point is t=0

# Define a function for computing DTW with time weighting
def time_weighted_euclidean(p1, p2, time_weight=0.5):
    """
    Compute Euclidean distance considering time as an additional dimension.
    `time_weight` controls how much time contributes to the distance.
    """
    spatial_dist = euclidean(p1[:2], p2[:2])  # Compute spatial distance (lat/lon)
    time_dist = abs(p1[2] - p2[2])  # Compute absolute time difference
    return spatial_dist + time_weight * time_dist  # Adjust contribution of time

# Compute DTW distance matrix with time weighting
num_trajs = len(trajectories)
dtw_matrix = np.zeros((num_trajs, num_trajs))

for i in range(num_trajs):
    for j in range(i+1, num_trajs):
        distance, _ = fastdtw(trajectories[i], trajectories[j], dist=lambda x, y: time_weighted_euclidean(x, y, time_weight=0.3))
        dtw_matrix[i, j] = distance
        dtw_matrix[j, i] = distance  # Symmetric matrix

print("DTW Distance Matrix with Time Penalty:")
print(dtw_matrix)

# Convert DTW distances to 2D embeddings for visualization
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
dtw_embeddings = mds.fit_transform(dtw_matrix)

# Apply Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=None, metric='euclidean', linkage='complete', distance_threshold=5.0)
cluster_labels = hierarchical.fit_predict(dtw_embeddings)

# Plot clusters
plt.scatter(dtw_embeddings[:, 0], dtw_embeddings[:, 1], c=cluster_labels, cmap='viridis', marker='o')
for i, txt in enumerate(range(len(trajectories))):
    plt.annotate(txt, (dtw_embeddings[i, 0], dtw_embeddings[i, 1]))
plt.title("Hierarchical Clustering with Time-Aware DTW")
plt.xlabel("MDS Component 1")
plt.ylabel("MDS Component 2")
plt.show()

print("Cluster Assignments:", cluster_labels)


def find_cluster_medoids(dtw_matrix, cluster_labels):
    medoids = {}
    for cluster_id in np.unique(cluster_labels):
        if cluster_id == -1:  # Skip noise points
            continue

        # Get indices of trajectories belonging to this cluster
        cluster_indices = np.where(cluster_labels == cluster_id)[0]

        # Compute sum of DTW distances for each trajectory in the cluster
        intra_cluster_dtw = dtw_matrix[np.ix_(cluster_indices, cluster_indices)]
        medoid_idx = cluster_indices[np.argmin(intra_cluster_dtw.sum(axis=1))]

        medoids[cluster_id] = medoid_idx  # Store the trajectory index
    return medoids


# Get medoids
medoids = find_cluster_medoids(dtw_matrix, cluster_labels)
print("Cluster Medoids (Trajectory Indexes):", medoids)

# Print medoid trajectories
for cluster_id, traj_idx in medoids.items():
    print(f"Cluster {cluster_id}: Medoid Trajectory Index = {traj_idx}")
    print(trajectories[traj_idx])  # Print trajectory points



