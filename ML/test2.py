import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.manifold import MDS
from sklearn_extra.cluster import KMedoids



# Generate sample taxi trajectories (latitude, longitude)
trajectories = [
    np.array([[0, 0], [1, 2], [2, 4], [3, 6], [4, 8]]),  # Straight path
    np.array([[0, 0], [1, 1.5], [2, 3.5], [3, 5], [4, 7.5]]),  # Slightly curved
    np.array([[0, 0], [1, 1], [2, 1], [3, 2], [4, 3]]),  # Different pattern
    np.array([[0, 0], [1, 2.2], [2, 4.3], [3, 6.1], [4, 8.2]])  # Almost same as first
]

# Plot trajectories
for traj in trajectories:
    plt.plot(traj[:, 0], traj[:, 1], marker='o')
plt.title("Sample Taxi Trajectories")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()


# Compute DTW distance matrix
num_trajs = len(trajectories)
dtw_matrix = np.zeros((num_trajs, num_trajs))

for i in range(num_trajs):
    for j in range(i+1, num_trajs):  # Distance is symmetric, so compute half
        distance, _ = fastdtw(trajectories[i], trajectories[j], dist=euclidean)
        dtw_matrix[i, j] = distance
        dtw_matrix[j, i] = distance  # Symmetric

print("DTW Distance Matrix:")
print(dtw_matrix)


# Apply MDS to reduce DTW distance matrix into a 2D embedding
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
dtw_embeddings = mds.fit_transform(dtw_matrix)

# Plot DTW Embeddings
plt.scatter(dtw_embeddings[:, 0], dtw_embeddings[:, 1], c='blue', marker='o')
for i, txt in enumerate(range(len(trajectories))):
    plt.annotate(txt, (dtw_embeddings[i, 0], dtw_embeddings[i, 1]))
plt.title("DTW-Based Trajectory Embeddings")
plt.xlabel("MDS Component 1")
plt.ylabel("MDS Component 2")
plt.show()



# Perform clustering
num_clusters = 2
kmedoids = KMedoids(n_clusters=num_clusters, metric='precomputed', random_state=42)
clusters = kmedoids.fit_predict(dtw_matrix)

# Plot clusters
plt.scatter(dtw_embeddings[:, 0], dtw_embeddings[:, 1], c=clusters, cmap='viridis', marker='o')
for i, txt in enumerate(range(len(trajectories))):
    plt.annotate(txt, (dtw_embeddings[i, 0], dtw_embeddings[i, 1]))
plt.title("DTW-Based Clustering of Taxi Trajectories")
plt.xlabel("MDS Component 1")
plt.ylabel("MDS Component 2")
plt.show()


