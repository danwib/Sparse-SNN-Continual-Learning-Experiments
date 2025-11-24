# scripts/run_meta_and_full.py
import argparse
import sys
from pathlib import Path
import yaml
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from usage_sparse_snn_cl.training.meta_training import (
    meta_train_controller,
    MetaTrainingConfig,
    ResourceLimitConfig,
)
from usage_sparse_snn_cl.training.neuron_features import NeuronFeatureTracker, NeuronFeatureConfig
from usage_sparse_snn_cl.training.controllers import PerNeuronController
from usage_sparse_snn_cl.training.train_sequence import train_task_sequence
from usage_sparse_snn_cl.models.snn_mlp import SimpleSNNMLP
from usage_sparse_snn_cl.data.mnist_split import make_mnist_split_loaders


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_full_training(cfg: dict, device: torch.device, controller_path: Path) -> None:
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

    feature_tracker = NeuronFeatureTracker(
        hidden_size=cfg["model"]["hidden_size"],
        input_size=cfg["model"]["input_size"],
        output_size=cfg["model"]["output_size"],
        layer_id_normalised=0.5,
        device=device,
        config=NeuronFeatureConfig(),
    )
    feature_tracker.to(device)
    feature_dim = feature_tracker.get_feature_matrix().shape[1]

    controller = PerNeuronController(feature_dim=feature_dim)
    state = torch.load(controller_path, map_location=device)
    controller.load_state_dict(state)
    controller.to(device)
    controller.eval()
    for p in controller.parameters():
        p.requires_grad = False

    print("[full] starting standard continual training with loaded controller...")
    train_task_sequence(
        model=model,
        task_loaders=loaders,
        cfg=cfg,
        device=device,
        feature_tracker=feature_tracker,
        controller=controller,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-config", type=str, required=True, help="Config for meta-training phase")
    parser.add_argument("--full-config", type=str, default=None, help="Config for full training (defaults to meta config)")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--epochs-per-task", type=int, default=3)
    parser.add_argument("--consol-epochs", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--max-consol-batches", type=int, default=10)
    parser.add_argument("--truncate-window", type=int, default=5)
    parser.add_argument("--controller-out", type=str, default="artifacts/controller_meta.pth")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--controller-lr", type=float, default=1e-3)
    parser.add_argument("--full-num-workers", type=int, default=0)
    args = parser.parse_args()

    meta_cfg_path = Path(args.meta_config)
    full_cfg_path = Path(args.full_config) if args.full_config else meta_cfg_path

    meta_cfg_dict = load_config(meta_cfg_path)
    full_cfg_dict = load_config(full_cfg_path)

    device = torch.device(args.device)

    meta_cfg = MetaTrainingConfig(episodes=args.episodes, controller_lr=args.controller_lr)
    resource_cfg = ResourceLimitConfig(
        max_batches_per_epoch=args.max_batches,
        max_consolidation_batches=args.max_consol_batches,
        epochs_per_task=args.epochs_per_task,
        consolidation_epochs=args.consol_epochs,
        num_workers=0,
        truncate_window=args.truncate_window,
    )

    print("[meta] starting meta-training...")
    controller = meta_train_controller(meta_cfg_dict, device, meta_cfg, resource_limits=resource_cfg)

    controller_out = Path(args.controller_out)
    controller_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(controller.state_dict(), controller_out)
    print(f"[meta] controller saved to {controller_out}")

    full_cfg_dict["data"]["num_workers"] = args.full_num_workers
    run_full_training(full_cfg_dict, device, controller_out)


if __name__ == "__main__":
    main()
