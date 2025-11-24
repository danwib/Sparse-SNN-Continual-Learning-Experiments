# src/usage_sparse_snn_cl/training/train_sequence.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from usage_sparse_snn_cl.training.usage_tracker import UsageTracker, UsageConfig
from usage_sparse_snn_cl.training.replay_buffer import ReplayBuffer
from usage_sparse_snn_cl.training.neuron_features import NeuronFeatureTracker, NeuronFeatureConfig
from usage_sparse_snn_cl.training.controllers import PerNeuronController
from usage_sparse_snn_cl.data.mnist_split import flatten_and_normalise


import copy
import torch.nn.functional as F



def weight_sparsity(model: nn.Module) -> float:
    """
    Returns fraction of weights that are exactly zero.
    """
    total = 0
    nonzero = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        data = p.detach()
        total += data.numel()
        nonzero += (data != 0).sum().item()
    if total == 0:
        return 0.0
    return 1.0 - nonzero / total


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def apply_neuron_gates(model: nn.Module, gates: torch.Tensor) -> None:
    """
    Applies per-neuron gates to the hidden layer gradients.
    gates: (hidden_size,) tensor in [0,1]
    """
    gates = gates.clamp(0.0, 1.0)
    hidden_linear: nn.Linear = model.hidden.linear
    readout: nn.Linear = model.readout

    if hidden_linear.weight.grad is not None:
        hidden_linear.weight.grad.mul_(gates.unsqueeze(1))
    if hidden_linear.bias.grad is not None:
        hidden_linear.bias.grad.mul_(gates)

    if readout.weight.grad is not None:
        readout.weight.grad.mul_(gates.unsqueeze(0))
    # readout bias is not neuron-specific; leave untouched.


def _clone_hidden_weight_grad(model: nn.Module) -> torch.Tensor | None:
    grad = model.hidden.linear.weight.grad
    if grad is None:
        return None
    return grad.detach().clone()


def _per_neuron_cosine_similarity(grad_a: torch.Tensor, grad_b: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    dot = (grad_a * grad_b).sum(dim=1)
    norm_a = torch.norm(grad_a, dim=1)
    norm_b = torch.norm(grad_b, dim=1)
    return dot / (norm_a * norm_b + eps)


def train_single_task(
    model: nn.Module,
    optimizer: optim.Optimizer,
    usage_tracker: UsageTracker,
    feature_tracker: NeuronFeatureTracker | None,
    controller: nn.Module | None,
    train_loader: DataLoader,
    replay_buffer: ReplayBuffer | None,
    device: torch.device,
    epochs: int,
    is_acquisition: bool,
    lambda_spike: float = 0.0,
    lambda_weight: float = 0.0,
    lambda_usage_weight: float = 0.0,
    # NEW: stability / distillation on previous tasks
    stability_buffer: ReplayBuffer | None = None,
    teacher_model: nn.Module | None = None,
    lambda_stab: float = 0.0,
) -> None:

    model.train()
    ce = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for x, y in train_loader:
            x = flatten_and_normalise(x).to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits, aux = model(x)

            loss_task = ce(logits, y)
            # spike sparsity: encourage low average hidden spike rate
            if lambda_spike > 0.0 and "hidden_spike_rate" in aux:
                spike_rate = aux["hidden_spike_rate"]
                spike_mean = spike_rate.mean()
                loss_task = loss_task + lambda_spike * spike_mean
                if feature_tracker is not None:
                    feature_tracker.update_spike_rates(spike_rate.detach())
            elif feature_tracker is not None and "hidden_spike_rate" in aux:
                feature_tracker.update_spike_rates(aux["hidden_spike_rate"].detach())

            stability_term = None
            stab_loss = None
            stability_control = None
            # Stability / distillation loss on previous tasks (Design 1)
            if (
                is_acquisition
                and stability_buffer is not None
                and teacher_model is not None
                and lambda_stab > 0.0
            ):
                # Sample a batch of Task-1 inputs
                x_old, _ = stability_buffer.sample(train_loader.batch_size)
                x_old = x_old.to(device)

                with torch.no_grad():
                    teacher_logits, _ = teacher_model(x_old)

                student_logits, _ = model(x_old)
                if controller is not None and feature_tracker is not None:
                    _, stability_control = controller(feature_tracker.get_feature_matrix())
                    stab_scale = stability_control.mean()
                else:
                    stab_scale = 1.0
                stab_loss = F.mse_loss(student_logits, teacher_logits)
                stability_term = lambda_stab * stab_scale * stab_loss

            total_loss = loss_task
            if stability_term is not None:
                total_loss = total_loss + stability_term

            # Gradient-based feature updates (conflict / stability)
            if (
                feature_tracker is not None
                and stability_term is not None
                and is_acquisition
            ):
                optimizer.zero_grad()
                loss_task.backward(retain_graph=True)
                grad_task = _clone_hidden_weight_grad(model)
                optimizer.zero_grad()
                stability_term.backward(retain_graph=True)
                grad_stab = _clone_hidden_weight_grad(model)
                optimizer.zero_grad()

                if grad_task is not None and grad_stab is not None:
                    conflict = _per_neuron_cosine_similarity(grad_task, grad_stab)
                    stability_signal = torch.norm(grad_stab, dim=1)
                    feature_tracker.update_grad_conflict(conflict)
                    feature_tracker.update_stability_error(stability_signal)
            elif feature_tracker is not None:
                zeros = torch.zeros_like(feature_tracker.spike_rate_fast)
                feature_tracker.update_grad_conflict(zeros)
                feature_tracker.update_stability_error(zeros)

            total_loss.backward()

            # update usage from grads
            usage_tracker.update_from_grads()
            if feature_tracker is not None:
                feature_tracker.update_usage_from_tracker(usage_tracker)

            if is_acquisition:
                if controller is not None and feature_tracker is not None:
                    gates, _ = controller(feature_tracker.get_feature_matrix())
                    apply_neuron_gates(model, gates)
                # scale grads so low-usage params learn faster
                usage_tracker.scale_grads_for_new_task()
            else:
                # consolidation: add usage-weighted L1 penalty via manual grad
                if lambda_weight > 0.0:
                    reg = usage_tracker.usage_weighted_l1(lambda_weight, lambda_usage_weight)
                    reg.backward()

            optimizer.step()

            if replay_buffer is not None and is_acquisition:
                replay_buffer.add_batch(x, y)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """
    Returns (accuracy, mean_hidden_spike_rate).
    """
    model.eval()
    accs: List[float] = []
    spike_means: List[float] = []

    for x, y in loader:
        x = flatten_and_normalise(x).to(device)
        y = y.to(device)
        logits, aux = model(x)
        accs.append(accuracy(logits, y))

        if "hidden_spike_rate" in aux:
            spike_means.append(aux["hidden_spike_rate"].mean().item())

    mean_acc = sum(accs) / len(accs)
    mean_spike = sum(spike_means) / len(spike_means) if spike_means else 0.0
    return mean_acc, mean_spike



def train_task_sequence(
    model: nn.Module,
    task_loaders: List[Tuple[DataLoader, DataLoader]],
    cfg: Dict[str, Any],
    device: torch.device,
    feature_tracker: NeuronFeatureTracker | None = None,
    controller: nn.Module | None = None,
) -> Dict[str, Any]:
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    usage_cfg = UsageConfig(
        decay=cfg["train"]["usage_decay"],
        alpha=cfg["train"]["usage_alpha"],
    )
    usage_tracker = UsageTracker(model=model, config=usage_cfg)

    if feature_tracker is None:
        feature_tracker = NeuronFeatureTracker(
            hidden_size=cfg["model"]["hidden_size"],
            input_size=cfg["model"]["input_size"],
            output_size=cfg["model"]["output_size"],
            layer_id_normalised=0.5,
            device=device,
            config=NeuronFeatureConfig(),
        )
    feature_tracker.to(device)

    if controller is None:
        feature_dim = feature_tracker.get_feature_matrix().shape[1]
        controller = PerNeuronController(feature_dim=feature_dim)
    controller.to(device)

    replay_buffer = ReplayBuffer(cfg["train"]["replay_buffer_size"]) \
        if cfg["train"]["do_consolidation"] else None

    teacher_model = None
    stability_buffer = None

    episode_metrics = {
        "pre_consolidation": [],
        "post_consolidation": [],
    }

    num_tasks = len(task_loaders)
    for task_id, (train_loader, test_loader) in enumerate(task_loaders):
        print(f"\n=== Task {task_id+1}/{num_tasks} ===")

        # 1) Acquisition on new task
        # For Task 1: stability_buffer/teacher_model are None → no stability term.
        # For Task 2+: they are set → Design 1 distillation is active.
        lambda_stab = float(cfg["train"].get("lambda_stab", 0.0))

        train_single_task(
            model=model,
            optimizer=optimizer,
            usage_tracker=usage_tracker,
            feature_tracker=feature_tracker,
            controller=controller,
            train_loader=train_loader,
            replay_buffer=replay_buffer,
            device=device,
            epochs=cfg["train"]["epochs_per_task"],
            is_acquisition=True,
            stability_buffer=stability_buffer,
            teacher_model=teacher_model,
            lambda_stab=lambda_stab,
        )

        # eval on all tasks so far (simple CL metric)
        pre_eval_stats: List[Dict[str, float]] = []
        for eval_id in range(task_id + 1):
            _, eval_loader = task_loaders[eval_id]
            acc, mean_spike = evaluate(model, eval_loader, device)
            print(
                f"  pre-consol eval on task {eval_id+1}: "
                f"acc={acc:.3f}, mean_hidden_spike_rate={mean_spike:.4f}"
            )
            pre_eval_stats.append(
                {"task": eval_id + 1, "acc": acc, "mean_hidden_spike_rate": mean_spike}
            )
        episode_metrics["pre_consolidation"].append(pre_eval_stats)



         # 2) Consolidation phase (optional)
        if cfg["train"]["do_consolidation"] and replay_buffer is not None and len(replay_buffer) > 0:
            print("  consolidation...")

            lambda_spike_base = float(cfg["train"]["lambda_spike"])
            lambda_weight_base = float(cfg["train"]["lambda_weight"])
            lambda_usage_weight = float(cfg["train"]["lambda_usage_weight"])

            ce = nn.CrossEntropyLoss()
            num_consol_epochs = cfg["train"]["consolidation_epochs"]

            for epoch in range(num_consol_epochs):
                # frac goes 0 → 1 across consolidation epochs
                frac = epoch / max(num_consol_epochs - 1, 1)

                # Two-stage behaviour:
                #   early epochs: sparsity_scale ≈ 1 (strong sparsity)
                #   late  epochs: sparsity_scale ≈ 0 (almost pure CE)
                sparsity_scale = 1.0 - frac
                print(
                    f"  consolidation epoch {epoch+1}/{num_consol_epochs} "
                    f"(sparsity_scale={sparsity_scale:.3f})"
                )

                # how many batches of replay per epoch
                num_batches = len(train_loader)
                max_consol_batches = cfg["train"].get("max_consolidation_batches")
                if max_consol_batches is not None:
                    num_batches = min(num_batches, max_consol_batches)

                for step in range(num_batches):
                    x_rep, y_rep = replay_buffer.sample(cfg["data"]["batch_size"])
                    x_rep = x_rep.to(device)
                    y_rep = y_rep.to(device)

                    optimizer.zero_grad()

                    logits, aux = model(x_rep)

                    # 1) task loss on replay (always active)
                    loss = ce(logits, y_rep)

                    # 2) spike sparsity – scaled by sparsity_scale
                    if (
                        "hidden_spike_rate" in aux
                        and lambda_spike_base > 0.0
                        and sparsity_scale > 0.0
                    ):
                        spike_rate = aux["hidden_spike_rate"]
                        loss = loss + (sparsity_scale * lambda_spike_base) * spike_rate.mean()

                    # 3) usage-weighted L1 – scaled by sparsity_scale
                    if lambda_weight_base > 0.0 and sparsity_scale > 0.0:
                        reg = usage_tracker.usage_weighted_l1(
                            sparsity_scale * lambda_weight_base,
                            lambda_usage_weight,
                        )
                        loss = loss + reg

                    # single backward pass for CE + sparsity
                    loss.backward()

                    # update usage stats from current grads
                    usage_tracker.update_from_grads()

                    optimizer.step()

                    # optional progress print every N steps
                    if (step + 1) % 50 == 0:
                        print(
                            f"    consol step {step+1}/{num_batches}, "
                            f"loss={loss.item():.3f}"
                        )


        # 2b) Evaluate again after consolidation to see its effect
        print("  post-consol evals:")
        post_eval_stats: List[Dict[str, float]] = []
        for eval_id in range(task_id + 1):
            _, eval_loader = task_loaders[eval_id]
            acc_post, mean_spike_post = evaluate(model, eval_loader, device)
            print(
                f"    task {eval_id+1}: "
                f"acc={acc_post:.3f}, mean_hidden_spike_rate={mean_spike_post:.4f}"
            )
            post_eval_stats.append(
                {"task": eval_id + 1, "acc": acc_post, "mean_hidden_spike_rate": mean_spike_post}
            )
        episode_metrics["post_consolidation"].append(post_eval_stats)


        # 3) Prune to restore sparsity
        usage_tracker.prune_small(cfg["train"]["prune_threshold"])
        if feature_tracker is not None:
            feature_tracker.snapshot_past_spike_rates()

        # 4) Report some stats
        w_sparsity = weight_sparsity(model)
        usage_stats = usage_tracker.summary()
        print(
            f"  stats after task {task_id+1}: "
            f"weight_sparsity={w_sparsity:.3f}, "
            f"usage_mean={usage_stats['usage_mean']:.5f}, "
            f"usage_max={usage_stats['usage_max']:.5f}"
        )

        # 5) After Task 1, snapshot a teacher + build stability buffer for Task 1
        if task_id == 0:
            print("  creating teacher snapshot and stability buffer for Task 1...")
            # Teacher: deep copy of current model, frozen
            teacher_model = copy.deepcopy(model).to(device)
            teacher_model.eval()
            for p in teacher_model.parameters():
                p.requires_grad = False

            
            stab_buf_size = cfg["train"].get("stability_buffer_size", 2000)
            stability_buffer = ReplayBuffer(stab_buf_size)

            task1_train_loader, _ = task_loaders[0]  # (train_loader, test_loader)
            for x_batch, y_batch in task1_train_loader:
                x_flat = flatten_and_normalise(x_batch)
                stability_buffer.add_batch(x_flat, y_batch)
            print(f"  stability buffer filled with {len(stability_buffer)} Task-1 batches")

    return episode_metrics
