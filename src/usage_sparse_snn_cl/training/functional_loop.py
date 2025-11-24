# src/usage_sparse_snn_cl/training/functional_loop.py
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Any, List, Tuple
import torch
from torch import nn, optim
from torch.nn.utils.stateless import functional_call
import torch.nn.functional as F

from usage_sparse_snn_cl.training.neuron_features import NeuronFeatureTracker
from usage_sparse_snn_cl.training.controllers import PerNeuronController
from usage_sparse_snn_cl.data.mnist_split import flatten_and_normalise
from usage_sparse_snn_cl.training.train_sequence import _per_neuron_cosine_similarity
from usage_sparse_snn_cl.training.replay_buffer import ReplayBuffer


def _make_functional_state(model: torch.nn.Module) -> Tuple[OrderedDict, OrderedDict]:
    params = OrderedDict(
        (name, param.detach().clone().requires_grad_(True))
        for name, param in model.named_parameters()
    )
    buffers = OrderedDict((name, buf.detach().clone()) for name, buf in model.named_buffers())
    return params, buffers


class _BatchCycler:
    """
    Helper that cycles through a loader indefinitely, storing CPU tensors.
    """

    def __init__(self, loader):
        self.loader = loader
        self.iterator = iter(loader)

    def next(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            batch = next(self.iterator)
        return batch


def _apply_gates_to_grads(
    grads: Dict[str, torch.Tensor],
    gates: torch.Tensor,
) -> None:
    if "hidden.linear.weight" in grads:
        grads["hidden.linear.weight"] = grads["hidden.linear.weight"] * gates.unsqueeze(1)
    if "hidden.linear.bias" in grads:
        grads["hidden.linear.bias"] = grads["hidden.linear.bias"] * gates
    if "readout.weight" in grads:
        grads["readout.weight"] = grads["readout.weight"] * gates.unsqueeze(0)


def _functional_forward(
    model: torch.nn.Module,
    params: OrderedDict,
    x_flat: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    return functional_call(model, params, (x_flat,), kwargs=None)


def functional_train_task_sequence(
    model: torch.nn.Module,
    task_loaders: List[Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]],
    cfg: Dict[str, Any],
    device: torch.device,
    controller: PerNeuronController,
    feature_tracker: NeuronFeatureTracker,
) -> Dict[str, Any]:
    """
    Differentiable variant of the training loop that keeps the controller in the computation graph.
    Uses manual SGD-style updates on a stateless copy of the model parameters.
    Currently focuses on the acquisition phase (no replay consolidation).
    """
    params, buffers = _make_functional_state(model)
    with torch.no_grad():
        for name, buf in model.named_buffers():
            buf.copy_(buffers[name])
    controller.to(device)
    feature_tracker.to(device)

    lr = cfg["train"]["lr"]
    lambda_spike = float(cfg["train"]["lambda_spike"])
    lambda_stab = float(cfg["train"].get("lambda_stab", 0.0))
    do_consolidation = cfg["train"].get("do_consolidation", False)
    replay_buffer = ReplayBuffer(cfg["train"]["replay_buffer_size"]) if do_consolidation else None
    max_batches = cfg["train"].get("max_batches_per_epoch")
    truncate_window = cfg["train"].get("truncate_window")
    steps_since_trunc = 0
    mid_eval_batches = cfg["train"].get("mid_eval_batches")

    teacher_params = None
    stability_batcher = None

    final_eval: List[Dict[str, Any]] = []
    diag_gates: List[Dict[str, float]] = []
    mid_eval: List[Dict[str, Any]] = []

    for task_id, (train_loader, test_loader) in enumerate(task_loaders):
        print(f"[functional] task {task_id+1}/{len(task_loaders)}")

        mid_recorded = False
        for epoch in range(cfg["train"]["epochs_per_task"]):
            batch_iter = iter(train_loader)
            batch_count = 0
            while True:
                try:
                    x, y = next(batch_iter)
                except StopIteration:
                    break
                batch_count += 1
                if max_batches is not None and batch_count > max_batches:
                    break
                x = flatten_and_normalise(x).to(device)
                y = y.to(device)

                logits, aux = _functional_forward(model, params, x)
                loss_task = F.cross_entropy(logits, y)

                if lambda_spike > 0.0 and "hidden_spike_rate" in aux:
                    spike_rate = aux["hidden_spike_rate"]
                    loss_task = loss_task + lambda_spike * spike_rate.mean()
                    feature_tracker.update_spike_rates(spike_rate.detach())

                stability_term = None
                stability_control = None
                if (
                    lambda_stab > 0.0
                    and teacher_params is not None
                    and stability_batcher is not None
                ):
                    x_old, y_old = stability_batcher.next()
                    x_old = flatten_and_normalise(x_old).to(device)
                    with torch.no_grad():
                        teacher_logits, _ = _functional_forward(model, teacher_params, x_old)
                    student_logits, _ = _functional_forward(model, params, x_old)
                    features = feature_tracker.get_feature_matrix()
                    _, stability_control = controller(features)
                    stab_scale = stability_control.mean()
                    stab_loss = F.mse_loss(student_logits, teacher_logits)
                    stability_term = lambda_stab * stab_scale * stab_loss

                if stability_term is not None:
                    grad_task_hidden = torch.autograd.grad(
                        loss_task, params["hidden.linear.weight"], retain_graph=True, create_graph=True
                    )[0]
                    grad_stab_hidden = torch.autograd.grad(
                        stability_term, params["hidden.linear.weight"], retain_graph=True, create_graph=True
                    )[0]
                    conflict = _per_neuron_cosine_similarity(grad_task_hidden, grad_stab_hidden)
                    stability_signal = torch.norm(grad_stab_hidden, dim=1)
                    feature_tracker.update_grad_conflict(conflict)
                    feature_tracker.update_stability_error(stability_signal)
                else:
                    zeros = torch.zeros(feature_tracker.hidden_size, device=device)
                    feature_tracker.update_grad_conflict(zeros)
                    feature_tracker.update_stability_error(zeros)

                total_loss = loss_task + (stability_term if stability_term is not None else 0.0)

                grads = torch.autograd.grad(total_loss, tuple(params.values()), create_graph=True)
                grads_dict = {name: grad for (name, _), grad in zip(params.items(), grads)}

                feature_tracker.update_usage_from_gradients(
                    grads_dict["hidden.linear.weight"],
                    grads_dict.get("hidden.linear.bias"),
                )

                features = feature_tracker.get_feature_matrix()
                gates, stability_control = controller(features)
                _apply_gates_to_grads(grads_dict, gates)

                diag_gates.append(
                    {
                        "task": task_id + 1,
                        "gate_mean": float(gates.mean().detach().cpu()),
                        "gate_std": float(gates.std(unbiased=False).detach().cpu()),
                        "stab_mean": float(stability_control.mean().detach().cpu()),
                        "stab_std": float(stability_control.std(unbiased=False).detach().cpu()),
                    }
                )

                new_params = OrderedDict()
                for (name, p) in params.items():
                    grad = grads_dict.get(name)
                    if grad is None:
                        new_params[name] = p
                    else:
                        new_params[name] = p - lr * grad
                params = new_params
                steps_since_trunc += 1

                if truncate_window is not None and steps_since_trunc >= truncate_window:
                    params = OrderedDict(
                        (name, p.detach().clone().requires_grad_(True)) for name, p in params.items()
                    )
                    steps_since_trunc = 0

                if replay_buffer is not None:
                    replay_buffer.add_batch(x.detach(), y.detach())

                if (
                    not mid_recorded
                    and mid_eval_batches is not None
                    and batch_count >= mid_eval_batches
                ):
                    mid_recorded = True
                    x_mid, y_mid = next(iter(test_loader))
                    x_mid = flatten_and_normalise(x_mid).to(device)
                    y_mid = y_mid.to(device)
                    logits_mid, _ = _functional_forward(model, params, x_mid)
                    loss_mid = F.cross_entropy(logits_mid, y_mid)
                    acc_mid = (logits_mid.argmax(dim=1) == y_mid).float().mean()
                    mid_eval.append(
                        {"task": task_id + 1, "loss": loss_mid, "acc": acc_mid}
                    )

        # After acquisition, optional consolidation (non-differentiable)
        if replay_buffer is not None and len(replay_buffer) > 0:
            params = run_consolidation_phase(
                model=model,
                params=params,
                buffers=buffers,
                replay_buffer=replay_buffer,
                cfg=cfg,
                device=device,
            )

        # After finishing the task, evaluate on this task (single batch proxy)
        test_iter = iter(test_loader)
        x_eval, y_eval = next(test_iter)
        x_eval = flatten_and_normalise(x_eval).to(device)
        y_eval = y_eval.to(device)
        logits_eval, _ = _functional_forward(model, params, x_eval)
        eval_loss = F.cross_entropy(logits_eval, y_eval)
        eval_acc = (logits_eval.argmax(dim=1) == y_eval).float().mean()
        final_eval.append({"task": task_id + 1, "loss": eval_loss, "acc": eval_acc})

        feature_tracker.snapshot_past_spike_rates()

        # After Task 1, freeze teacher parameters and stability data
        if task_id == 0:
            teacher_params = OrderedDict((name, p.detach().clone()) for name, p in params.items())
            stability_batcher = _BatchCycler(train_loader)

    return {
        "final_eval": final_eval,
        "mid_eval": mid_eval,
        "gate_stats": diag_gates,
    }


def load_params_into_model(model: nn.Module, params: OrderedDict, buffers: OrderedDict) -> None:
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(params[name])
        for name, b in model.named_buffers():
            b.copy_(buffers[name])


def extract_params_from_model(model: nn.Module) -> OrderedDict:
    return OrderedDict(
        (name, p.detach().clone().requires_grad_(True))
        for name, p in model.named_parameters()
    )


def run_consolidation_phase(
    model: nn.Module,
    params: OrderedDict,
    buffers: OrderedDict,
    replay_buffer: ReplayBuffer,
    cfg: Dict[str, Any],
    device: torch.device,
) -> OrderedDict:
    load_params_into_model(model, params, buffers)

    ce = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    lambda_spike = float(cfg["train"]["lambda_spike"])
    num_epochs = cfg["train"]["consolidation_epochs"]
    batch_size = cfg["data"]["batch_size"]
    max_batches = cfg["train"].get("max_consolidation_batches")

    model.train()
    for epoch in range(num_epochs):
        steps = 0
        while steps < len(replay_buffer):
            if max_batches is not None and steps >= max_batches:
                break
            x_rep, y_rep = replay_buffer.sample(batch_size)
            x_rep = flatten_and_normalise(x_rep).to(device)
            y_rep = y_rep.to(device)

            optimizer.zero_grad()
            logits, aux = model(x_rep)
            loss = ce(logits, y_rep)
            if lambda_spike > 0.0 and "hidden_spike_rate" in aux:
                loss = loss + lambda_spike * aux["hidden_spike_rate"].mean()
            loss.backward()
            optimizer.step()
            steps += 1

    return extract_params_from_model(model)
