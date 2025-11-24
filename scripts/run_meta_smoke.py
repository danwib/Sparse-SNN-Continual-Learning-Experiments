# scripts/run_meta_smoke.py
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=2)
    parser.add_argument("--max-consol-batches", type=int, default=2)
    parser.add_argument("--epochs-per-task", type=int, default=1)
    parser.add_argument("--consol-epochs", type=int, default=1)
    parser.add_argument("--truncate-window", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)

    meta_cfg = MetaTrainingConfig(episodes=args.episodes, controller_lr=1e-3)
    resource_cfg = ResourceLimitConfig(
        max_batches_per_epoch=args.max_batches,
        max_consolidation_batches=args.max_consol_batches,
        epochs_per_task=args.epochs_per_task,
        consolidation_epochs=args.consol_epochs,
        num_workers=0,
        truncate_window=args.truncate_window,
    )

    controller = meta_train_controller(cfg, device, meta_cfg, resource_limits=resource_cfg)
    total_params = sum(p.numel() for p in controller.parameters())
    print(f"[meta-smoke] controller params: {total_params}")


if __name__ == "__main__":
    main()
