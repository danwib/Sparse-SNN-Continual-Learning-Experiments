# src/usage_sparse_snn_cl/training/controllers.py
from __future__ import annotations

import torch
from torch import nn


class PerNeuronController(nn.Module):
    """
    Simple MLP that maps per-neuron feature vectors to gates in [0, 1].
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 32, num_layers: int = 2, temperature: float = 1.0):
        super().__init__()
        layers = []
        in_dim = feature_dim
        for _ in range(max(0, num_layers - 1)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self.temperature = temperature

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: (num_neurons, feature_dim)
        returns: (num_neurons,) gates in [0, 1]
        """
        logits = self.net(features).squeeze(-1)
        if self.temperature != 1.0:
            logits = logits / self.temperature
        return torch.sigmoid(logits)
