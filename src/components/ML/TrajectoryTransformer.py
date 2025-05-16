import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(1, d_model)  # Project time into d_model space

    def forward(self, t) -> torch.Tensor:
        t = t.unsqueeze(-1)  # Shape: (batch, seq_len, 1)
        return self.linear(t)  # Output: (batch, seq_len, d_model)


class SpaceEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(2, d_model)  # Project (longitude, latitude) into d_model space

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.linear(coords)  # Output: (batch, seq_len, d_model)


class TrajectoryTransformer(nn.Module):
    def __init__(self, d_model=128, num_heads=4, num_layers=3, output_dim=64):
        super().__init__()

        self.time_emb = TimeEmbedding(d_model)  # Time embedding
        self.spatial_emb = SpaceEmbedding(d_model)  # Spatial embedding

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model * 2, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(d_model * 2, output_dim, bias=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        :param x: (num_trajectories, max length of rows, [0: tid, 1,2: coords])
        :param mask: samme som x:  (num_trajectories, max length of rows, True|False)
        :return: Embedded Tensor of size (num_trajectories, output_dim)
        """
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
