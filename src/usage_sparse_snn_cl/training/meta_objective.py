# src/usage_sparse_snn_cl/training/meta_objective.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import torch


@dataclass
class MetaObjectiveConfig:
    """
    Weighs stability vs. plasticity when scoring a whole continual-learning episode.
    """

    stability_weight: float = 0.5
    plasticity_weight: float = 0.5
    mid_plasticity_weight: float = 0.4


def stability_plasticity_objective(
    episode_metrics: Dict[str, List[Dict[str, torch.Tensor]]],
    config: MetaObjectiveConfig,
) -> torch.Tensor:
    """
    Differentiable proxy objective for the controller.
    Uses per-task evaluation losses recorded at the end of the episode.
    Encourages low CE loss on Task 1 (stability) and on the latest task (plasticity).
    """
    final_eval = episode_metrics.get("final_eval", [])
    if not final_eval:
        return torch.tensor(0.0, dtype=torch.float32)

    stability_loss = final_eval[0]["loss"]
    plasticity_loss = final_eval[-1]["loss"]

    mid_entries = [
        entry
        for entry in episode_metrics.get("mid_eval", [])
        if entry["task"] == len(final_eval)
    ]

    score = -(
        config.stability_weight * stability_loss
        + config.plasticity_weight * plasticity_loss
    )
    if mid_entries and config.mid_plasticity_weight > 0.0:
        mid_mean = sum(entry["loss"] for entry in mid_entries) / len(mid_entries)
        score = score - config.mid_plasticity_weight * mid_mean
    return score
