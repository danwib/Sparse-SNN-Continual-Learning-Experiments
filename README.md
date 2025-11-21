# Sparse SNN Continual Learning Experiments

This repository contains exploratory experiments on **sparse spiking neural networks (SNNs)** for **continual learning (CL)** on Split-MNIST–style tasks.

The core idea is to exploit the natural sparsity of SNNs to allocate *reserve capacity* for new tasks, while using replay and regularisation to consolidate knowledge and prevent catastrophic forgetting.

## What we are doing

- We train a simple SNN MLP on Split-MNIST (Task 1: digits 0–4, Task 2: digits 5–9).
- The network is encouraged to be **sparse** in two senses:
  - **Activity sparsity** via a spike-rate penalty.
  - **Weight sparsity** via a usage-weighted L1 regulariser.
- We track **per-parameter usage**, so that:
  - Frequently used weights / neurons are treated as “Task 1 core”.
  - Less-used weights are treated as “reserve capacity” for new tasks.
- After each task, we run a **consolidation phase**:
  - Replay from a buffer,
  - Minimise CE on replayed data,
  - Apply sparsity penalties to re-sparsify the network.

We then added a **stability / distillation loss** for Task 2:
- After Task 1, we snapshot a frozen teacher model and build a Task-1 stability buffer.
- During Task-2 training, in addition to Task-2 CE we penalise deviation from the teacher on Task-1 inputs.
- This behaves like a soft EWC / distillation-style constraint specialised to the SNN setting.

## What we have found so far

- **Naive CL** (no stability, basic sparsity + replay) shows strong catastrophic forgetting: Task-1 accuracy collapses after Task 2.
- Adding **usage-based gating** (favouring “dead” / low-usage neurons for new tasks) helps, but does not fully prevent forgetting.
- Adding a **Task-1 stability loss** significantly improves retention:
  - Task 1 can remain above ~96–98 % after Task 2 in some regimes.
  - Task 2 still reaches ~94–96 % after consolidation.
- The stability coefficient \(\lambda_{stab}\) clearly acts as a **stability–plasticity dial**:
  - Larger \(\lambda_{stab}\): stronger Task-1 protection but slower Task-2 learning.
  - Smaller \(\lambda_{stab}\): more Task-2 freedom, slightly more Task-1 drift and more variance across runs.

Overall, the combination of **sparsity, usage-aware gating, replay, and stability** yields a controllable trade-off between remembering old tasks and learning new ones.

## Where we intend to go next

- Run more systematic **ablations**:
  - Compare \(\lambda_{stab} = 0, 0.25, 0.5, 1.0\) across multiple seeds.
  - Compare with/without consolidation and with/without usage-based gating.
- Explore **more structured stability**:
  - Apply stability only to a “core” subset of high-usage neurons.
  - Vary how often the stability loss is applied.
- Move towards a **Nested Learning (NL) formulation**:
  - Treat usage-based gating and stability strength as outer-loop parameters.
  - Eventually meta-learn a plasticity policy that decides *where* and *how strongly* to update the SNN to minimise forgetting while maximising continual performance.

This is all exploratory research code; expect rough edges and ongoing changes as the ideas evolve.