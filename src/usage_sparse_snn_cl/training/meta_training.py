# src/usage_sparse_snn_cl/training/meta_training.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any
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
) -> PerNeuronController:
    """
    Runs multiple continual-learning episodes and updates the controller parameters
    using the chosen meta-objective.
    """
    feature_tracker_template = _make_feature_tracker(cfg, device)
    feature_dim = feature_tracker_template.get_feature_matrix().shape[1]

    controller = PerNeuronController(feature_dim=feature_dim).to(device)
    meta_optimizer = optim.Adam(controller.parameters(), lr=meta_cfg.controller_lr)

    for episode in range(meta_cfg.episodes):
        print(f"[meta] episode {episode+1}/{meta_cfg.episodes}")
        # Fresh model + data loaders per episode
        model = SimpleSNNMLP(
            input_size=cfg["model"]["input_size"],
            hidden_size=cfg["model"]["hidden_size"],
            output_size=cfg["model"]["output_size"],
            time_steps=cfg["model"]["time_steps"],
            v_th=cfg["model"]["v_th"],
        )
        model.to(device)

        loaders = make_mnist_split_loaders(
            root=cfg["data"]["root"],
            splits=cfg["data"]["splits"],
            batch_size=cfg["data"]["batch_size"],
            num_workers=cfg["data"]["num_workers"],
        )

        feature_tracker = _make_feature_tracker(cfg, device)

        episode_metrics = functional_train_task_sequence(
            model=model,
            task_loaders=loaders,
            cfg=cfg,
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

    return controller
