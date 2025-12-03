# src/usage_sparse_snn_cl/training/stability_surrogate.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class StabilitySurrogateConfig:
    """
    Lightweight container for surrogate training hyperparameters.
    """

    enabled: bool = False
    warmup_batches: int = 0
    hidden_dim: int = 64
    num_layers: int = 2
    lr: float = 1e-3
    epochs: int = 200
    batch_size: Optional[int] = None
    gate_beta: float = 25.0


class StabilitySurrogate(nn.Module):
    """
    Simple MLP regressor that maps descriptors to a scalar stability loss.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim
        for _ in range(max(0, num_layers - 1)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


def _summarise_matrix(features: torch.Tensor) -> torch.Tensor:
    """
    Concatenates mean, std, min, max along dim=0.
    """
    mean = features.mean(dim=0)
    std = features.std(dim=0, unbiased=False)
    min_val, _ = features.min(dim=0)
    max_val, _ = features.max(dim=0)
    return torch.cat([mean, std, min_val, max_val], dim=0)


def _summarise_vector(vec: torch.Tensor) -> torch.Tensor:
    """
    Returns mean/std/min/max for a 1-D tensor.
    """
    return torch.stack(
        [
            vec.mean(),
            vec.std(unbiased=False),
            vec.min(),
            vec.max(),
        ]
    )


def build_surrogate_descriptor(
    feature_matrix: Optional[torch.Tensor],
    loss_task: torch.Tensor,
    batch_acc: float,
    spike_batch: Optional[torch.Tensor],
    gates: Optional[torch.Tensor],
    stability_control: Optional[torch.Tensor],
    logits: torch.Tensor,
    inputs: torch.Tensor,
    lambda_stab: float,
    grad_stats: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Assembles a compact descriptor vector from per-neuron summaries and
    batch-level statistics.
    """
    device = loss_task.device
    stats: List[torch.Tensor] = []
    if feature_matrix is not None:
        stats.append(_summarise_matrix(feature_matrix))
    if spike_batch is not None:
        # spike_batch: (batch, hidden); summarise over the flattened batch.
        flat_spikes = spike_batch.reshape(-1, spike_batch.shape[-1])
        stats.append(_summarise_matrix(flat_spikes))
    if gates is not None:
        stats.append(_summarise_vector(gates))
    if stability_control is not None:
        stats.append(_summarise_vector(stability_control))

    logits_stats = torch.stack(
        [
            logits.mean(),
            logits.std(unbiased=False),
            logits.abs().max(),
        ]
    )
    input_stats = torch.stack(
        [
            inputs.mean(),
            inputs.std(unbiased=False),
        ]
    )
    scalars = torch.tensor(
        [loss_task.item(), batch_acc, lambda_stab],
        device=device,
    )
    stats.extend([logits_stats, input_stats, scalars])
    if grad_stats is not None:
        stats.append(grad_stats.to(device))
    return torch.cat(stats).detach()


def train_stability_surrogate(
    descriptors: torch.Tensor,
    targets: torch.Tensor,
    config: StabilitySurrogateConfig,
    device: torch.device,
) -> tuple[StabilitySurrogate, List[float]]:
    """
    Fits the surrogate MLP on the collected descriptor/target pairs.
    Returns the trained model and per-epoch training losses.
    """
    dataset = TensorDataset(descriptors, targets)
    batch_size = config.batch_size or len(dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = StabilitySurrogate(
        input_dim=descriptors.size(1),
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    losses: List[float] = []

    for _ in range(max(1, config.epochs)):
        total_loss = 0.0
        count = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x)
            loss = nn.functional.mse_loss(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * batch_x.size(0)
            count += batch_x.size(0)
        losses.append(total_loss / max(1, count))

    return model, losses


def make_surrogate_config(config_dict: Optional[Dict[str, Any]]) -> StabilitySurrogateConfig:
    if not config_dict:
        return StabilitySurrogateConfig()
    return StabilitySurrogateConfig(
        enabled=bool(config_dict.get("enabled", False)),
        warmup_batches=int(config_dict.get("warmup_batches", 0)),
        hidden_dim=int(config_dict.get("hidden_dim", 64)),
        num_layers=int(config_dict.get("num_layers", 2)),
        lr=float(config_dict.get("lr", 1e-3)),
        epochs=int(config_dict.get("epochs", 200)),
        batch_size=config_dict.get("batch_size"),
        gate_beta=float(config_dict.get("gate_beta", 25.0)),
    )


def loss_to_gate(stab_loss: torch.Tensor, beta: float) -> torch.Tensor:
    """
    Maps a teacher distillation loss to a [0,1] gate: high loss → strong attenuation.
    """
    gate = torch.exp(-beta * stab_loss)
    return gate.clamp(0.0, 1.0)
