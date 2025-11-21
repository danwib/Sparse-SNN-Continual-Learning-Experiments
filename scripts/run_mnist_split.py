# scripts/run_mnist_split.py
import argparse
import sys
from pathlib import Path

import yaml
import torch

# Add src/ to sys.path so we can import usage_sparse_snn_cl
ROOT = Path(__file__).resolve().parents[1]  # repo root (sparse_cl_tests/)
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from usage_sparse_snn_cl.models.snn_mlp import SimpleSNNMLP
from usage_sparse_snn_cl.data.mnist_split import make_mnist_split_loaders
from usage_sparse_snn_cl.training.train_sequence import train_task_sequence



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg.get("device", "cpu"))

    loaders = make_mnist_split_loaders(
        root=cfg["data"]["root"],
        splits=cfg["data"]["splits"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    model = SimpleSNNMLP(
        input_size=cfg["model"]["input_size"],
        hidden_size=cfg["model"]["hidden_size"],
        output_size=cfg["model"]["output_size"],
        time_steps=cfg["model"]["time_steps"],
        v_th=cfg["model"]["v_th"],
    )

    train_task_sequence(model, loaders, cfg, device)


if __name__ == "__main__":
    main()
