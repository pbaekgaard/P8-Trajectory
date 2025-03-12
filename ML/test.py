import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import datetime
import numpy as np
from sklearn_extra.cluster import KMedoids



# Sample Dataset Class
def timestamp_to_float(timestamp: str):
    return datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").timestamp()


class TrajectoryDataset(Dataset):
    def __init__(self, trajectories):
        self.data = [torch.tensor(t, dtype=torch.float32) for t in trajectories]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Transformer Model for Trajectory Embedding
class TrajectoryTransformer(nn.Module):
    def __init__(self, input_dim=3, model_dim=128, num_heads=4, num_layers=2, output_dim=64):
        super(TrajectoryTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)  # Project (lon, lat, time) to model_dim
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads), num_layers=num_layers
        )
        self.fc = nn.Linear(model_dim, output_dim)  # Output 64D embedding

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=0)  # Global average pooling
        return self.fc(x)


# Example Training Function
def train_model(model, dataloader, epochs=10, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # Use contrastive loss or triplet loss for better representation

    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            embeddings = model(batch)
            loss = criterion(embeddings, torch.zeros_like(embeddings))  # Dummy loss, replace with a real one
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


# Example usage
if __name__ == "__main__":
    sample_trajectories = [
        [[116.51172, 39.92123, timestamp_to_float("2008-02-02 15:36:08")],
         [116.51135, 39.93883, timestamp_to_float("2008-02-02 15:46:08")]],
        [[116.51627, 39.91034, timestamp_to_float("2008-02-02 15:56:08")],
         [116.47186, 39.91248, timestamp_to_float("2008-02-02 16:06:08")]],
    ]

    dataset = TrajectoryDataset(sample_trajectories)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = TrajectoryTransformer()
    train_model(model, dataloader)

    sample_input = torch.tensor([[
                                    [116.51172, 39.92123, 1204308968],  # Sample trajectory
                                    [116.51135, 39.93883, 1204309568],
                                    [116.51140, 39.93880, 1204309580],
                                    [116.51172, 39.92123, 1204308968],
                                    [1.1, 400.2, 0]
                                  ]], dtype=torch.float32)

    output = model(sample_input).detach().numpy()
    print(output)
    print(output.shape)

    X = np.asarray([[1, 2], [1, 4], [1, 0],
                    [4, 2], [4, 4], [4, 0]])
    kmedoids_labels = KMedoids(n_clusters=2, random_state=0).fit_predict(output)

    print(kmedoids_labels)
