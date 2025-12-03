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
from usage_sparse_snn_cl.training.stability_surrogate import (
    StabilitySurrogate,
    build_surrogate_descriptor,
    loss_to_gate,
    make_surrogate_config,
    train_stability_surrogate,
)


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


def clone_parameter_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Snapshot all trainable parameters; used for controller-only stability.
    """
    snapshot: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            snapshot[name] = param.detach().clone()
    return snapshot


def controller_stability_penalty(
    model: nn.Module,
    snapshot: Dict[str, torch.Tensor],
    stability_control: torch.Tensor | None,
) -> torch.Tensor:
    """
    Penalise deviation from a previous task's parameters using controller-provided
    per-neuron stability weights.
    """
    if not snapshot:
        device = next(model.parameters()).device
        return torch.zeros((), device=device)

    hidden_linear: nn.Linear = model.hidden.linear
    readout: nn.Linear = model.readout
    device = hidden_linear.weight.device
    if stability_control is None:
        stability_control = torch.ones(hidden_linear.weight.size(0), device=device)
    else:
        stability_control = stability_control.to(device)

    penalty = 0.0
    terms = 0

    prev_hidden_w = snapshot.get("hidden.linear.weight")
    if prev_hidden_w is not None:
        diff = hidden_linear.weight - prev_hidden_w.to(device)
        penalty = penalty + (stability_control.unsqueeze(1) * diff.pow(2)).mean()
        terms += 1

    prev_hidden_b = snapshot.get("hidden.linear.bias")
    if prev_hidden_b is not None:
        diff = hidden_linear.bias - prev_hidden_b.to(device)
        penalty = penalty + (stability_control * diff.pow(2)).mean()
        terms += 1

    prev_readout_w = snapshot.get("readout.weight")
    if prev_readout_w is not None:
        diff = readout.weight - prev_readout_w.to(device)
        penalty = penalty + (stability_control.unsqueeze(0) * diff.pow(2)).mean()
        terms += 1

    prev_readout_b = snapshot.get("readout.bias")
    if prev_readout_b is not None:
        diff = readout.bias - prev_readout_b.to(device)
        penalty = penalty + diff.pow(2).mean()
        terms += 1

    if terms == 0:
        return torch.zeros((), device=device)
    return penalty / terms


def _grad_stats_from_tensor_list(grads: List[torch.Tensor]) -> torch.Tensor | None:
    flats: List[torch.Tensor] = []
    for grad in grads:
        if grad is not None:
            flats.append(grad.reshape(-1))
    if not flats:
        return None
    flat = torch.cat(flats)
    mean_abs = flat.abs().mean()
    max_abs = flat.abs().max()
    l2_norm = torch.sqrt(torch.sum(flat.pow(2)))
    return torch.stack([mean_abs, max_abs, l2_norm])


def _teacher_weight_penalty(model: nn.Module, teacher_model: nn.Module) -> torch.Tensor:
    penalty = 0.0
    terms = 0
    for p, t in zip(model.parameters(), teacher_model.parameters()):
        if not p.requires_grad:
            continue
        diff = p - t.detach()
        penalty = penalty + diff.pow(2).mean()
        terms += 1
    if terms == 0:
        device = next(model.parameters()).device
        return torch.zeros((), device=device)
    return penalty / terms


def _run_surrogate_warmup(
    model: nn.Module,
    train_loader: DataLoader,
    stability_buffer: ReplayBuffer | None,
    teacher_model: nn.Module | None,
    feature_tracker: NeuronFeatureTracker | None,
    controller: nn.Module | None,
    device: torch.device,
    lambda_stab: float,
    surrogate_cfg: StabilitySurrogateConfig | None,
    trainable_params: List[torch.Tensor],
) -> tuple[StabilitySurrogate | None, int]:
    """
    Collects descriptor/target pairs using the teacher model and fits the surrogate.
    Returns the trained surrogate and the number of batches consumed.
    """
    if (
        surrogate_cfg is None
        or not surrogate_cfg.enabled
        or surrogate_cfg.warmup_batches <= 0
        or stability_buffer is None
        or teacher_model is None
    ):
        return None, 0

    ce = nn.CrossEntropyLoss()
    descriptors: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    loader_iter = iter(train_loader)
    batches = 0

    print(
        f"  [surrogate] collecting {surrogate_cfg.warmup_batches} Task-2 batches for warm-up..."
    )
    while batches < surrogate_cfg.warmup_batches:
        try:
            x, y = next(loader_iter)
        except StopIteration:
            break
        batches += 1
        x = flatten_and_normalise(x).to(device)
        y = y.to(device)

        logits, aux = model(x)
        loss_task = ce(logits, y)
        batch_acc = accuracy(logits, y)
        feature_matrix = (
            feature_tracker.get_feature_matrix().detach()
            if feature_tracker is not None
            else None
        )
        spike_batch = aux.get("hidden_spike_rate") if isinstance(aux, dict) else None
        gates = None
        stability_control = None
        if controller is not None and feature_matrix is not None:
            g, s = controller(feature_matrix)
            gates = g.detach()
            stability_control = s.detach()

        grads = torch.autograd.grad(
            loss_task,
            trainable_params,
            retain_graph=False,
            allow_unused=True,
            create_graph=False,
        )
        grad_stats = _grad_stats_from_tensor_list([g for g in grads if g is not None])

        descriptor = build_surrogate_descriptor(
            feature_matrix=feature_matrix,
            loss_task=loss_task.detach(),
            batch_acc=batch_acc,
            spike_batch=spike_batch.detach() if spike_batch is not None else None,
            gates=gates,
            stability_control=stability_control,
            logits=logits.detach(),
            inputs=x.detach(),
            lambda_stab=lambda_stab,
            grad_stats=grad_stats.detach() if grad_stats is not None else None,
        ).cpu()
        descriptors.append(descriptor)

        x_old, _ = stability_buffer.sample(train_loader.batch_size)
        x_old = x_old.to(device)
        with torch.no_grad():
            teacher_logits, _ = teacher_model(x_old)
        student_logits, _ = model(x_old)
        stab_loss = F.mse_loss(student_logits, teacher_logits).detach()
        gate_target = loss_to_gate(stab_loss, surrogate_cfg.gate_beta).cpu()
        targets.append(gate_target)

    if not descriptors:
        print("  [surrogate] insufficient data for warm-up; skipping surrogate training.")
        return None, 0

    desc_tensor = torch.stack(descriptors)
    target_tensor = torch.stack(targets)
    surrogate, losses = train_stability_surrogate(desc_tensor, target_tensor, surrogate_cfg, device)
    final_loss = losses[-1] if losses else 0.0
    print(
        f"  [surrogate] trained on {len(descriptors)} samples "
        f"(final surrogate loss={final_loss:.6f})"
    )
    return surrogate, len(descriptors)


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
    stability_snapshot: Dict[str, torch.Tensor] | None = None,
    stability_mode: str = "distillation",
    stability_surrogate: StabilitySurrogate | None = None,
    skip_initial_batches: int = 0,
    surrogate_cfg: StabilitySurrogateConfig | None = None,
    trainable_params: List[torch.Tensor] | None = None,
) -> None:

    model.train()
    ce = nn.CrossEntropyLoss()
    prev_grad_stats = torch.zeros(3, device=device)
    if trainable_params is None:
        trainable_params = [p for p in model.parameters() if p.requires_grad]

    skipped_batches = 0
    for epoch in range(epochs):
        for x, y in train_loader:
            if skip_initial_batches > 0 and skipped_batches < skip_initial_batches:
                skipped_batches += 1
                continue
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
            gates_ctrl = None
            feature_matrix = (
                feature_tracker.get_feature_matrix() if feature_tracker is not None else None
            )
            if controller is not None and feature_matrix is not None:
                gates_ctrl, stability_control = controller(feature_matrix)
            grad_stats = prev_grad_stats

            use_surrogate_gate = (
                stability_mode == "distillation"
                and is_acquisition
                and lambda_stab > 0.0
                and stability_surrogate is not None
                and teacher_model is not None
            )
            surrogate_gate = None
            if use_surrogate_gate:
                surrogate_batch_counter += 1
                spike_batch = aux.get("hidden_spike_rate") if isinstance(aux, dict) else None
                batch_descriptor = build_surrogate_descriptor(
                    feature_matrix=feature_matrix.detach() if feature_matrix is not None else None,
                    loss_task=loss_task.detach(),
                    batch_acc=accuracy(logits, y),
                    spike_batch=spike_batch.detach() if spike_batch is not None else None,
                    gates=gates_ctrl.detach() if gates_ctrl is not None else None,
                    stability_control=stability_control.detach() if stability_control is not None else None,
                    logits=logits.detach(),
                    inputs=x.detach(),
                    lambda_stab=lambda_stab,
                    grad_stats=grad_stats.detach() if grad_stats is not None else None,
                ).to(device)
                with torch.no_grad():
                    surrogate_gate = stability_surrogate(batch_descriptor.unsqueeze(0)).squeeze(0).item()
                surrogate_gate = float(max(0.0, min(1.0, surrogate_gate)))
                interval = 1
                if surrogate_cfg is not None and surrogate_cfg.drift_penalty_interval > 1:
                    interval = surrogate_cfg.drift_penalty_interval
                if surrogate_batch_counter % interval == 0:
                    drift_penalty = _teacher_weight_penalty(model, teacher_model)
                    stability_term = lambda_stab * (1.0 - surrogate_gate) * drift_penalty

            if not use_surrogate_gate:
                if (
                    stability_mode == "distillation"
                    and is_acquisition
                    and lambda_stab > 0.0
                    and stability_buffer is not None
                    and teacher_model is not None
                ):
                    stab_scale = (
                        stability_control.mean() if stability_control is not None else 1.0
                    )
                    # Sample a batch of Task-1 inputs
                    x_old, _ = stability_buffer.sample(train_loader.batch_size)
                    x_old = x_old.to(device)

                    with torch.no_grad():
                        teacher_logits, _ = teacher_model(x_old)

                    student_logits, _ = model(x_old)
                    stab_loss = F.mse_loss(student_logits, teacher_logits)
                    stability_term = lambda_stab * stab_scale * stab_loss
                elif (
                    stability_mode == "controller_reg"
                    and is_acquisition
                    and stability_snapshot is not None
                    and lambda_stab > 0.0
                ):
                    stability_term = lambda_stab * controller_stability_penalty(
                        model, stability_snapshot, stability_control
                    )

            total_loss = loss_task
            if stability_term is not None:
                total_loss = total_loss + stability_term

            gate_scale = None
            if surrogate_gate is not None:
                gate_scale = surrogate_gate

            # Gradient-based feature updates (conflict / stability)
            if (
                feature_tracker is not None
                and stability_term is not None
                and is_acquisition
                and stability_term.requires_grad
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

            grad_list = [p.grad for p in trainable_params]
            grad_stats_current = _grad_stats_from_tensor_list(
                [g.detach() for g in grad_list if g is not None]
            )
            prev_grad_stats = grad_stats_current

            if gate_scale is not None:
                for p in trainable_params:
                    if p.grad is not None:
                        p.grad.mul_(gate_scale)

            # update usage from grads
            usage_tracker.update_from_grads()
            if feature_tracker is not None:
                feature_tracker.update_usage_from_tracker(usage_tracker)

            if is_acquisition:
                if gates_ctrl is not None:
                    apply_neuron_gates(model, gates_ctrl)
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

    use_controller = bool(cfg["train"].get("use_controller", True))
    if not use_controller:
        controller = None
    elif controller is None:
        feature_dim = feature_tracker.get_feature_matrix().shape[1]
        controller = PerNeuronController(feature_dim=feature_dim)
    if controller is not None:
        controller.to(device)

    replay_buffer = ReplayBuffer(cfg["train"]["replay_buffer_size"]) \
        if cfg["train"]["do_consolidation"] else None

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    teacher_model = None
    stability_buffer = None
    stability_snapshot: Dict[str, torch.Tensor] | None = None
    stability_mode = cfg["train"].get("stability_mode", "distillation")
    surrogate_cfg_dict = cfg["train"].get("stability_surrogate")
    surrogate_cfg = None
    if surrogate_cfg_dict:
        surrogate_cfg = make_surrogate_config(surrogate_cfg_dict)

    episode_metrics = {
        "pre_consolidation": [],
        "post_consolidation": [],
    }

    num_tasks = len(task_loaders)
    epochs_cfg = cfg["train"]["epochs_per_task"]
    epochs_schedule = None
    if isinstance(epochs_cfg, list):
        epochs_schedule = [int(v) for v in epochs_cfg]
    else:
        default_epochs = int(epochs_cfg)
    for task_id, (train_loader, test_loader) in enumerate(task_loaders):
        print(f"\n=== Task {task_id+1}/{num_tasks} ===")

        # 1) Acquisition on new task
        # For Task 1: stability_buffer/teacher_model are None → no stability term.
        # For Task 2+: they are set → Design 1 distillation is active.
        lambda_stab = float(cfg["train"].get("lambda_stab", 0.0))

        stability_surrogate = None
        surrogate_skip = 0
        if (
            stability_mode == "distillation"
            and lambda_stab > 0.0
            and surrogate_cfg is not None
            and surrogate_cfg.enabled
            and task_id > 0
        ):
            stability_surrogate, surrogate_skip = _run_surrogate_warmup(
                model=model,
                train_loader=train_loader,
                stability_buffer=stability_buffer,
                teacher_model=teacher_model,
                feature_tracker=feature_tracker,
                controller=controller,
                device=device,
                lambda_stab=lambda_stab,
                surrogate_cfg=surrogate_cfg,
                trainable_params=trainable_params,
            )

        if epochs_schedule is not None:
            epochs = epochs_schedule[task_id] if task_id < len(epochs_schedule) else epochs_schedule[-1]
        else:
            epochs = default_epochs

        train_single_task(
            model=model,
            optimizer=optimizer,
            usage_tracker=usage_tracker,
            feature_tracker=feature_tracker,
            controller=controller,
            train_loader=train_loader,
            replay_buffer=replay_buffer,
            device=device,
            epochs=epochs,
            is_acquisition=True,
            stability_buffer=stability_buffer,
            teacher_model=teacher_model,
            lambda_stab=lambda_stab,
            stability_snapshot=stability_snapshot,
            stability_mode=stability_mode,
            stability_surrogate=stability_surrogate,
            skip_initial_batches=surrogate_skip,
            surrogate_cfg=surrogate_cfg,
            trainable_params=trainable_params,
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

        # 5) Build stability references for future tasks
        if stability_mode == "distillation" and task_id == 0:
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
        elif stability_mode == "controller_reg":
            stability_snapshot = clone_parameter_dict(model)

    return episode_metrics
