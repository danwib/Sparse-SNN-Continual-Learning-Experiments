# src/usage_sparse_snn_cl/training/usage_tracker.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict
import torch
from torch import nn


@dataclass
class UsageConfig:
    decay: float = 0.99        # EMA decay
    alpha: float = 5.0         # gradient scaling: 1 / (1 + alpha * usage)


@dataclass
class UsageTracker:
    """
    Tracks per-parameter 'usage' based on gradient magnitudes.
    usage[param_name] has same shape as param and stores an EMA of |grad|.
    """
    model: nn.Module
    config: UsageConfig
    usage: Dict[str, torch.Tensor] = field(default_factory=dict)

    def maybe_init(self) -> None:
        if self.usage:
            return
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.usage[name] = torch.zeros_like(p.data)

    @torch.no_grad()
    def update_from_grads(self) -> None:
        """
        After backward() but before optimizer.step():
        update usage with |grad|.
        """
        self.maybe_init()
        for name, p in self.model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            u = self.usage[name]
            grad_abs = p.grad.detach().abs()
            u.mul_(self.config.decay).add_(grad_abs * (1.0 - self.config.decay))

    @torch.no_grad()
    def scale_grads_for_new_task(self) -> None:
        """
        Acquisition phase:
        scale gradients so that low-usage parameters learn much faster
        and high-usage parameters are almost frozen.
        """
        self.maybe_init()
        alpha = self.config.alpha

        for name, p in self.model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue

            u = self.usage[name]

            # Normalise usage within this parameter tensor
            # so that mean usage ≈ 1
            u_norm = u / (u.mean() + 1e-8)

            # High usage → big u_norm → exp(-alpha * u_norm) ~ 0
            # Low usage → small u_norm → scale close to 1
            scale = torch.exp(-alpha * u_norm)

            p.grad.mul_(scale)


    def usage_weighted_l1(self, lambda_weight: float, lambda_usage_weight: float) -> torch.Tensor:
        """
        For consolidation: compute usage-weighted L1 penalty.
        Important (high-usage) weights are penalised less.
        """
        self.maybe_init()
        reg = 0.0
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            u = self.usage[name]
            # smaller penalty for high usage
            penalty_scale = 1.0 / (1.0 + lambda_usage_weight * u)
            reg = reg + (penalty_scale * p.abs()).mean()
        return lambda_weight * reg

    @torch.no_grad()
    def prune_small(self, threshold: float) -> None:
        """
        Hard-prune weights: set to zero if |w| < threshold and usage is small.
        """
        self.maybe_init()
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            u = self.usage[name]
            mask_keep = (p.abs() >= threshold) | (u > u.mean())  # keep relatively important
            p.data.mul_(mask_keep)

    def summary(self) -> Dict[str, float]:
        """
        Global summary stats of usage values across all parameters.
        """
        self.maybe_init()
        all_usages = [u.detach().flatten() for u in self.usage.values()]
        if not all_usages:
            return {"usage_mean": 0.0, "usage_max": 0.0}
        concat = torch.cat(all_usages)
        return {
            "usage_mean": float(concat.mean().item()),
            "usage_max": float(concat.max().item()),
        }
