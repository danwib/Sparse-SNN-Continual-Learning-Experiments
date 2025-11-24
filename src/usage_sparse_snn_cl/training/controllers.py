# src/usage_sparse_snn_cl/training/controllers.py
from __future__ import annotations

import torch
from torch import nn


class PerNeuronController(nn.Module):
    """
    MLP that maps per-neuron feature vectors to:
      - plasticity gate g_i in [0,1]
      - stability attenuation s_i in [0,1]
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 32, num_layers: int = 2, temperature: float = 1.0):
        super().__init__()
        layers = []
        in_dim = feature_dim
        for _ in range(max(0, num_layers - 1)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.body = nn.Sequential(*layers)
        head_in = in_dim
        self.plasticity_head = nn.Linear(head_in, 1)
        self.stability_head = nn.Linear(head_in, 1)
        self.temperature = temperature

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        features: (num_neurons, feature_dim)
        returns:
          gates: (num_neurons,) plasticity gates ∈ [0,1]
          stability: (num_neurons,) attenuation factors ∈ [0,1]
        """
        x = self.body(features)
        gate_logits = self.plasticity_head(x).squeeze(-1)
        if self.temperature != 1.0:
            gate_logits = gate_logits / self.temperature
        gates = torch.sigmoid(gate_logits)

        stab_logits = self.stability_head(x).squeeze(-1)
        stability = torch.sigmoid(stab_logits)
        return gates, stability
