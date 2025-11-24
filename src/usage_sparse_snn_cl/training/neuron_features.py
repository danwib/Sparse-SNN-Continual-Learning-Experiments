# src/usage_sparse_snn_cl/training/neuron_features.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class NeuronFeatureConfig:
    """
    Hyperparameters governing the smoothing behaviour of each signal.
    """

    spike_decay_fast: float = 0.2   # fast EMA for on-line activity
    spike_decay_slow: float = 0.05  # slow EMA for long-term activity
    conflict_decay: float = 0.2     # EMA on gradient conflict
    stability_decay: float = 0.2    # EMA on stability / teacher error
    usage_decay: float = 0.1        # EMA on gradient-derived usage


class NeuronFeatureTracker:
    """
    Maintains per-neuron signals that will later feed the Level-2 controller.
    Currently tracks the hidden layer in SimpleSNNMLP.
    """

    def __init__(
        self,
        hidden_size: int,
        input_size: int,
        output_size: int,
        layer_id_normalised: float,
        device: torch.device,
        config: Optional[NeuronFeatureConfig] = None,
    ) -> None:
        self.hidden_size = hidden_size
        self.device = device
        self.config = config or NeuronFeatureConfig()

        # Dynamic signals
        self.spike_rate_fast = torch.zeros(hidden_size, device=device)
        self.spike_rate_slow = torch.zeros(hidden_size, device=device)
        self.spike_rate_past = torch.zeros(hidden_size, device=device)
        self.usage = torch.zeros(hidden_size, device=device)
        self.grad_conflict = torch.zeros(hidden_size, device=device)
        self.stability_error = torch.zeros(hidden_size, device=device)

        # Static context features
        self.layer_id = torch.full((hidden_size,), layer_id_normalised, device=device)
        log_fan_in = float(torch.log(torch.tensor(float(input_size))))
        log_fan_out = float(torch.log(torch.tensor(float(output_size))))
        self.log_fan_in = torch.full((hidden_size,), log_fan_in, device=device)
        self.log_fan_out = torch.full((hidden_size,), log_fan_out, device=device)

    def to(self, device: torch.device) -> None:
        """
        Move all tensors to `device`.
        """
        self.device = device
        for name, tensor in self.__dict__.items():
            if isinstance(tensor, torch.Tensor):
                setattr(self, name, tensor.to(device))

    @staticmethod
    def _ema_update(storage: torch.Tensor, value: torch.Tensor, decay: float) -> torch.Tensor:
        """
        Helper for EMA update with decay in [0,1], returns the updated tensor.
        """
        storage.mul_(1.0 - decay).add_(value * decay)
        return storage

    def update_spike_rates(self, batch_spike_rate: torch.Tensor) -> None:
        """
        Update fast/slow EMAs using the batch-mean spike rate: (batch, hidden) -> (hidden,).
        """
        spike_mean = batch_spike_rate.mean(dim=0)
        self._ema_update(self.spike_rate_fast, spike_mean, self.config.spike_decay_fast)
        self._ema_update(self.spike_rate_slow, spike_mean, self.config.spike_decay_slow)

    def snapshot_past_spike_rates(self) -> None:
        """
        Called after each task: treats the slow EMA as the representative "past task" spike rate.
        """
        self.spike_rate_past.copy_(self.spike_rate_slow)

    def update_usage_from_tracker(self, usage_tracker) -> None:
        """
        Aggregates UsageTracker tensors into a per-neuron usage signal.
        Averages usage across all incoming weights (+ bias if available).
        """
        weight_usage = usage_tracker.usage.get("hidden.linear.weight")
        if weight_usage is None:
            return
        # weight_usage: (hidden, input)
        per_neuron = weight_usage.mean(dim=1)

        bias_usage = usage_tracker.usage.get("hidden.linear.bias")
        if bias_usage is not None:
            per_neuron = 0.5 * (per_neuron + bias_usage)

        self.usage.copy_(per_neuron.to(self.device))

    def update_usage_from_gradients(self, grad_weight: torch.Tensor, grad_bias: torch.Tensor | None = None) -> None:
        """
        Updates usage EMA using raw gradient magnitudes (without a UsageTracker).
        """
        grad_mag = grad_weight.detach().abs().mean(dim=1)
        if grad_bias is not None:
            grad_mag = 0.5 * (grad_mag + grad_bias.detach().abs())
        self._ema_update(self.usage, grad_mag, self.config.usage_decay)

    def update_grad_conflict(self, conflict_signal: torch.Tensor) -> None:
        """
        Update EMA of gradient conflict; expects shape (hidden,) – e.g., cosine similarity per neuron.
        """
        self._ema_update(self.grad_conflict, conflict_signal.to(self.device), self.config.conflict_decay)

    def update_stability_error(self, stability_signal: torch.Tensor) -> None:
        """
        Update EMA of per-neuron stability error (e.g., |grad from distillation|).
        """
        self._ema_update(self.stability_error, stability_signal.to(self.device), self.config.stability_decay)

    def get_feature_matrix(self) -> torch.Tensor:
        """
        Returns a (hidden_size, num_features) tensor ready for the controller.
        """
        features = [
            self.spike_rate_fast,
            self.spike_rate_slow,
            self.spike_rate_past,
            self.usage,
            self.grad_conflict,
            self.stability_error,
            self.layer_id,
            self.log_fan_in,
            self.log_fan_out,
        ]
        return torch.stack(features, dim=1)
