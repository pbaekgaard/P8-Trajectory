import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import yaml
import itertools
import os
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
import tools.scripts._preprocess as _load_data



from src.components.ML.TrajectoryTransformer import TrajectoryTransformer
from src.components.ML.reference_set_construction import pad_batches, df_to_tensor, normalize_df

# =====================
# SimCLR Components
# =====================
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        # B batchsize
        # D Dimension
        batch_size = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)  # [2B, D]
        z = F.normalize(z, dim=1)

        similarity = torch.matmul(z, z.T) / self.temperature  # [2B, 2B]

        mask = torch.eye(2 * batch_size, device=z.device).bool()
        similarity = similarity.masked_fill(mask, -float('inf'))  # mask out self-similarities

        # Positive pairs are offset by batch_size
        positives = torch.cat([
            torch.diag(similarity, batch_size),
            torch.diag(similarity, -batch_size)
        ]).view(2 * batch_size, 1)  # [2B, 1]

        # Mask out the positives in similarity matrix before selecting negatives
        # So negatives = all similarities except self and positive

        negatives = similarity.clone()
        for i in range(batch_size):
            negatives[i, i + batch_size] = -float('inf')
            negatives[i + batch_size, i] = -float('inf')

        # Concatenate positive logits with negatives logits
        logits = torch.cat([positives, negatives], dim=1)  # [2B, 2B]

        # Targets: positives are always index 0
        targets = torch.zeros(2 * batch_size, dtype=torch.long, device=z.device)

        loss = F.cross_entropy(logits, targets)
        # loss = 1 - F.cosine_similarity(z_i, z_j).mean()
        return loss




# =====================
# Dataset + Augmentation
# =====================
class TrajectoryDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.trajectories = list(df.groupby("trajectory_id"))

    def __len__(self):
        return len(self.trajectories)

    def augment(self, traj_df):
        # Example augmentation: jitter + crop
        jitter = traj_df.copy()
        jitter[["longitude", "latitude"]] += np.random.normal(0, 0.0001, size=(len(jitter), 2)) # ~16,6m latitude, ~12,8m longitude, assuming Beijing City
        jitter[["t_relative"]] += np.random.normal(0, 0.0000167, size=(len(jitter), 1)) # 7,2s, assuming dataset ranges over 5 days. Actual is ~days so maybe 9 seconds.
        if len(jitter) > 10:
            jitter = jitter.sample(frac=np.random.uniform(0.7, 1.0)).sort_values("timestamp")
        return jitter

    def __getitem__(self, idx):
        tid, traj = self.trajectories[idx]
        traj1 = self.augment(traj)
        traj2 = self.augment(traj)
        return traj1, traj2

# =====================
# Training Loop
# =====================
def train(model, dataloader, optimizer, loss_fn, device, epochs):
    model.train()
    loss = 100
    for epoch in range(epochs):
        total_loss = 0
        for traj1, traj2 in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            traj1 = pd.concat(traj1, ignore_index=True)
            traj2 = pd.concat(traj2, ignore_index=True)
            padded1 = pad_batches(traj1)
            padded2 = pad_batches(traj2)

            x1, mask1 = df_to_tensor(padded1)
            x2, mask2 = df_to_tensor(padded2)

            x1, mask1 = x1.to(device), mask1.to(device)
            x2, mask2 = x2.to(device), mask2.to(device)

            z1 = model(x1, mask1)
            z2 = model(x2, mask2)

            loss = loss_fn(z1, z2)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    return loss

def save_models():
    param_config_path = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "grid_search_params.yml"))
    model_files_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")



    with open(param_config_path, "r") as f:
        grid = yaml.safe_load(f)

    # Create all combinations of parameter values
    param_combinations = list(itertools.product(
        grid['batch_size'],
        grid['d_model'],
        grid['num_heads'],
        grid['num_layers']
    ))

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess data once
    df = _load_data.main()
    df['t_relative'] = df.groupby('trajectory_id')['timestamp'].transform(lambda x: x - x.min())
    df = normalize_df(df)

    # Loop through all combinations
    for batch_size, d_model, num_heads, num_layers in param_combinations:
        print(
            f"Training with batch_size={batch_size}, d_model={d_model}, num_heads={num_heads}, num_layers={num_layers}")

        dataset = TrajectoryDataset(df)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: list(zip(*x)))

        model = TrajectoryTransformer(d_model=d_model, num_heads=num_heads, num_layers=num_layers).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)
        loss_fn = NTXentLoss(temperature=0.09)

        loss = train(model, dataloader, optimizer, loss_fn, device, epochs=50)

        filename = f"{batch_size}-{d_model}-{num_heads}-{num_layers}-{loss}_transformer.pt"
        torch.save(model.state_dict(), os.path.join(model_files_path, filename))
        print(f"Saved model to {filename}")



if __name__ == "__main__":
    save_models()
