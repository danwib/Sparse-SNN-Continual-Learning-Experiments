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


def stability_plasticity_objective(
    episode_metrics: Dict[str, List[List[Dict[str, float]]]],
    config: MetaObjectiveConfig,
) -> torch.Tensor:
    """
    Produces a scalar objective that can supervise the Level-2 controller.
    Uses the stored episode metrics emitted by `train_task_sequence`.
    Currently combines:
      - stability: accuracy on the very first task after the final post-consolidation eval
      - plasticity: accuracy on the most recent task after the final post-consolidation eval
    """
    if not episode_metrics["post_consolidation"]:
        return torch.tensor(0.0)

    final_post = episode_metrics["post_consolidation"][-1]
    stability_acc = final_post[0]["acc"]
    plasticity_acc = final_post[-1]["acc"]

    score = (
        config.stability_weight * stability_acc
        + config.plasticity_weight * plasticity_acc
    )
    return torch.tensor(score, dtype=torch.float32)
