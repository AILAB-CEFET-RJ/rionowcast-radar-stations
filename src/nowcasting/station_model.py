"""Baselines neurais temporais que usam apenas medições de estações."""

from __future__ import annotations

import torch
from torch import nn


class StationMLP(nn.Module):
    """MLP que recebe valores e disponibilidade das estações em uma janela temporal."""

    def __init__(self, station_count: int, t_in: int = 5, t_out: int = 5, hidden_dim: int = 256):
        super().__init__()
        self.station_count, self.t_out = station_count, t_out
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(t_in * station_count * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, t_out * station_count),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).reshape(-1, self.t_out, self.station_count)
