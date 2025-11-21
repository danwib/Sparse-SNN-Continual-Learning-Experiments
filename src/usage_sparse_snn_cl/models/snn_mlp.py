# src/usage_sparse_snn_cl/models/snn_mlp.py
from __future__ import annotations
from typing import Dict, Tuple
import torch
from torch import nn


class SpikeFn(torch.autograd.Function):
    """
    Heaviside in forward; fast-sigmoid surrogate gradient in backward.
    """

    @staticmethod
    def forward(ctx, v: torch.Tensor, v_th: float):
        ctx.save_for_backward(v)
        ctx.v_th = v_th
        return (v > v_th).float()

    @staticmethod
    def backward(ctx, grad_output):
        (v,) = ctx.saved_tensors
        v_th = ctx.v_th
        # fast-sigmoid derivative
        scale = 10.0
        x = scale * (v - v_th)
        grad_v = grad_output * (scale * torch.sigmoid(x) * (1 - torch.sigmoid(x)))
        return grad_v, None


spike_fn = SpikeFn.apply


class LIFLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, v_th: float = 1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.v_th = v_th

    def forward(self, x: torch.Tensor, state: torch.Tensor | None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, in_features)
        state: (batch, out_features) membrane potential or None
        returns: (spikes, new_state)
        """
        if state is None:
            state = torch.zeros(x.size(0), self.linear.out_features, device=x.device)

        v = state + self.linear(x)
        s = spike_fn(v, self.v_th)
        v = v - s * self.v_th  # reset by subtraction
        return s, v


class SimpleSNNMLP(nn.Module):
    """
    One hidden LIF layer, readout from time-averaged hidden spikes.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, time_steps: int, v_th: float = 1.0):
        super().__init__()
        self.time_steps = time_steps
        self.hidden = LIFLayer(input_size, hidden_size, v_th=v_th)
        self.readout = nn.Linear(hidden_size, output_size)

    def forward(self, x_flat: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        x_flat: (batch, input_size) – we will rate-encode as Poisson spikes.
        Returns:
          logits: (batch, output_size)
          aux: {"hidden_spike_rate": (batch, hidden_size)}
        """
        batch_size = x_flat.size(0)
        hidden_state = None
        hidden_spikes_acc = 0.0

        # rate-based Poisson encoding
        for t in range(self.time_steps):
            # x_flat is in [0,1]; draw Bernoulli spikes
            x_t = torch.bernoulli(x_flat)
            s_h, hidden_state = self.hidden(x_t, hidden_state)
            hidden_spikes_acc = hidden_spikes_acc + s_h

        hidden_rate = hidden_spikes_acc / self.time_steps  # (batch, hidden_size)
        logits = self.readout(hidden_rate)
        aux = {"hidden_spike_rate": hidden_rate}
        return logits, aux
