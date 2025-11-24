# src/usage_sparse_snn_cl/training/meta_training.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import copy
import torch
from torch import optim

from usage_sparse_snn_cl.models.snn_mlp import SimpleSNNMLP
from usage_sparse_snn_cl.data.mnist_split import make_mnist_split_loaders
from usage_sparse_snn_cl.training.neuron_features import NeuronFeatureTracker, NeuronFeatureConfig
from usage_sparse_snn_cl.training.controllers import PerNeuronController
from usage_sparse_snn_cl.training.meta_objective import (
    MetaObjectiveConfig,
    stability_plasticity_objective,
)
from usage_sparse_snn_cl.training.functional_loop import functional_train_task_sequence


@dataclass
class MetaTrainingConfig:
    """
    Controls the outer-loop optimisation of the controller.
    """

    episodes: int = 3
    controller_lr: float = 1e-3
    objective: MetaObjectiveConfig = field(default_factory=MetaObjectiveConfig)


@dataclass
class ResourceLimitConfig:
    """
    Optional overrides that keep meta episodes lightweight.
    """

    max_batches_per_epoch: Optional[int] = None
    max_consolidation_batches: Optional[int] = None
    epochs_per_task: Optional[int] = None
    consolidation_epochs: Optional[int] = None
    num_workers: int = 0  # default to single-process loading for safety
    truncate_window: Optional[int] = None


def _apply_resource_limits(cfg: Dict[str, Any], limits: ResourceLimitConfig) -> Dict[str, Any]:
    scoped = copy.deepcopy(cfg)
    train_cfg = scoped.setdefault("train", {})
    if limits.max_batches_per_epoch is not None:
        train_cfg["max_batches_per_epoch"] = limits.max_batches_per_epoch
    if limits.max_consolidation_batches is not None:
        train_cfg["max_consolidation_batches"] = limits.max_consolidation_batches
    if limits.epochs_per_task is not None:
        train_cfg["epochs_per_task"] = limits.epochs_per_task
    if limits.consolidation_epochs is not None:
        train_cfg["consolidation_epochs"] = limits.consolidation_epochs
    train_cfg.setdefault("do_consolidation", True)
    if limits.truncate_window is not None:
        train_cfg["truncate_window"] = limits.truncate_window
    scoped.setdefault("data", {})["num_workers"] = limits.num_workers
    return scoped


def _make_feature_tracker(cfg: Dict[str, Any], device: torch.device) -> NeuronFeatureTracker:
    tracker = NeuronFeatureTracker(
        hidden_size=cfg["model"]["hidden_size"],
        input_size=cfg["model"]["input_size"],
        output_size=cfg["model"]["output_size"],
        layer_id_normalised=0.5,
        device=device,
        config=NeuronFeatureConfig(),
    )
    tracker.to(device)
    return tracker


def meta_train_controller(
    cfg: Dict[str, Any],
    device: torch.device,
    meta_cfg: MetaTrainingConfig,
    resource_limits: Optional[ResourceLimitConfig] = None,
) -> PerNeuronController:
    """
    Runs multiple continual-learning episodes and updates the controller parameters
    using the chosen meta-objective.
    """
    base_cfg = copy.deepcopy(cfg)
    if resource_limits is not None:
        base_cfg = _apply_resource_limits(base_cfg, resource_limits)

    feature_tracker_template = _make_feature_tracker(base_cfg, device)
    feature_dim = feature_tracker_template.get_feature_matrix().shape[1]

    controller = PerNeuronController(feature_dim=feature_dim).to(device)
    meta_optimizer = optim.Adam(controller.parameters(), lr=meta_cfg.controller_lr)

    for episode in range(meta_cfg.episodes):
        print(f"[meta] episode {episode+1}/{meta_cfg.episodes}")
        # Fresh model + data loaders per episode
        episode_cfg = copy.deepcopy(base_cfg)

        model = SimpleSNNMLP(
            input_size=episode_cfg["model"]["input_size"],
            hidden_size=episode_cfg["model"]["hidden_size"],
            output_size=episode_cfg["model"]["output_size"],
            time_steps=episode_cfg["model"]["time_steps"],
            v_th=episode_cfg["model"]["v_th"],
        )
        model.to(device)

        loaders = make_mnist_split_loaders(
            root=episode_cfg["data"]["root"],
            splits=episode_cfg["data"]["splits"],
            batch_size=episode_cfg["data"]["batch_size"],
            num_workers=episode_cfg["data"]["num_workers"],
        )

        feature_tracker = _make_feature_tracker(episode_cfg, device)

        episode_metrics = functional_train_task_sequence(
            model=model,
            task_loaders=loaders,
            cfg=episode_cfg,
            device=device,
            controller=controller,
            feature_tracker=feature_tracker,
        )

        objective = stability_plasticity_objective(episode_metrics, meta_cfg.objective)
        meta_loss = -objective

        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()

        print(f"[meta] objective={objective.item():.4f}")
        if "final_eval" in episode_metrics:
            first = episode_metrics["final_eval"][0]
            last = episode_metrics["final_eval"][-1]
            print(
                f"[meta] task1_loss={first['loss'].item():.4f}, "
                f"task{last['task']}_loss={last['loss'].item():.4f}"
            )
        if "gate_stats" in episode_metrics and episode_metrics["gate_stats"]:
            gate_means = [entry["gate_mean"] for entry in episode_metrics["gate_stats"]]
            gate_stds = [entry["gate_std"] for entry in episode_metrics["gate_stats"]]
            stab_means = [entry["stab_mean"] for entry in episode_metrics["gate_stats"]]
            stab_stds = [entry["stab_std"] for entry in episode_metrics["gate_stats"]]
            print(
                f"[meta] gate_mean={sum(gate_means)/len(gate_means):.3f}, "
                f"gate_std={sum(gate_stds)/len(gate_stds):.3f}, "
                f"stab_mean={sum(stab_means)/len(stab_means):.3f}, "
                f"stab_std={sum(stab_stds)/len(stab_stds):.3f}"
            )

    return controller
