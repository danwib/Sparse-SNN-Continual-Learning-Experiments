# src/usage_sparse_snn_cl/training/replay_buffer.py
from collections import deque
from typing import Tuple
import random

import torch


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        # store (x_batch, y_batch) pairs; each x,y are tensors
        self.storage: deque[Tuple[torch.Tensor, torch.Tensor]] = deque(maxlen=capacity)

    def add_batch(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Store a whole batch (x, y) in the buffer.
        We detach and move to CPU to keep GPU memory usage down.
        """
        self.storage.append((x.detach().cpu(), y.detach().cpu()))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return roughly `batch_size` samples by:
          - picking ONE stored batch at random
          - optionally sub-sampling it if it's larger than batch_size

        This keeps consolidation batch sizes similar to normal training.
        """
        assert len(self.storage) > 0
        x, y = random.choice(self.storage)  # one stored batch

        # Sub-sample if this stored batch is larger than requested
        if x.size(0) > batch_size:
            idx = torch.randperm(x.size(0))[:batch_size]
            x = x[idx]
            y = y[idx]

        return x, y

    def __len__(self) -> int:
        return len(self.storage)
