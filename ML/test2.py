import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector
from sklearn.preprocessing import MinMaxScaler
from sklearn_extra.cluster import KMedoids

# Example taxi trajectories (each trajectory is a sequence of GPS points)
trajectories = [
    [(40.7128, -74.0060), (40.7135, -74.0055), (40.7150, -74.0040)],  # Example Trajectory 1
    [(40.7306, -73.9352), (40.7310, -73.9345), (40.7320, -73.9330)],  # Example Trajectory 2
    [(40.7500, -73.9800), (40.7510, -73.9795), (40.7525, -73.9780)],   # Example Trajectory 3
    [(40.7500, -73.9800), (40.7510, -73.9795), (40.7525, -73.9780)]
]

# Convert to NumPy array and normalize
scaler = MinMaxScaler()
max_length = max(len(t) for t in trajectories)  # Find longest trajectory

def pad_and_normalize(trajectories):
    normalized = []
    for traj in trajectories:
        traj = np.array(traj)  # Convert list to array
        traj = scaler.fit_transform(traj)  # Normalize GPS coordinates
        # Pad to max_length with zeros
        traj_padded = np.pad(traj, ((0, max_length - len(traj)), (0, 0)), mode='constant')
        normalized.append(traj_padded)
    return np.array(normalized)

X_train = pad_and_normalize(trajectories)

# Model parameters
timesteps = X_train.shape[1]  # Max trajectory length
input_dim = 2  # (latitude, longitude)
latent_dim = 64  # Size of trajectory embedding

# Encoder
inputs = Input(shape=(timesteps, input_dim))
encoded = LSTM(128, activation='relu', return_sequences=True)(inputs)
encoded = LSTM(64, activation='relu', return_sequences=False)(encoded)

# Bottleneck (embedding layer)
embedding = Dense(latent_dim, activation='relu', name="trajectory_embedding")(encoded)

# Decoder
decoded = RepeatVector(timesteps)(embedding)
decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)
decoded = LSTM(128, activation='relu', return_sequences=True)(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)  # Output (lat, lon)

# Compile model
autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Define embedding model (extracts the latent representation)
embedding_model = Model(inputs, embedding)

# Train the autoencoder
autoencoder.fit(X_train, X_train, epochs=50, batch_size=16, validation_split=0.2)

# Extract trajectory embeddings
trajectory_embeddings = embedding_model.predict(X_train)

# Print the embedding for the first trajectory
print(trajectory_embeddings[0])


# Number of clusters (you can tune this)
num_clusters = 2

# Apply K-Medoids clustering
kmedoids = KMedoids(n_clusters=num_clusters, random_state=42)
labels = kmedoids.fit_predict(trajectory_embeddings)

# Print cluster assignments
print("Cluster assignments:", labels)

# Get the indices of medoids (representative trajectories)
medoid_indices = kmedoids.medoid_indices_
print("Medoid indices:", medoid_indices)

# Get representative trajectories (reference set)
reference_set = [X_train[idx] for idx in medoid_indices]
