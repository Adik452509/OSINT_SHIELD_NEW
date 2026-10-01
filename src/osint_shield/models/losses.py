"""Losses for the multi-task model.

Class weights are computed from the labels passed in and nothing else. The
caller is responsible for passing **only the training fold's** labels -
computing them on the full corpus leaks the test distribution into training.
:func:`osint_shield.training.fold.run_fold` does this, and a test asserts it.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch import nn


def class_weights(y, n_classes: int, *, cap: float | None = None) -> torch.Tensor:
    """Inverse-frequency ("balanced") weights from one fold's labels.

    Args:
        y: integer labels of the **training fold only**.
        n_classes: size of the label space. Classes absent from ``y`` get
            weight 1.0 - they cannot contribute to the loss anyway, and a large
            weight would blow up the moment a later fold contains one.
        cap: upper bound on any single weight. Essential for propaganda: with
            ~4 positives in a ~340-row fold, "balanced" gives the positive class
            ~42x, and the head learns to predict Yes for everything.

    Returns:
        float32 tensor of shape ``(n_classes,)``.
    """
    y = np.asarray(y, dtype=int)
    if y.size == 0:
        raise ValueError("cannot compute class weights from an empty label array")
    if y.min() < 0 or y.max() >= n_classes:
        raise ValueError(f"labels must lie in [0, {n_classes}); got {y.min()}..{y.max()}")

    weights = np.ones(n_classes, dtype=np.float32)
    present = np.unique(y)
    if present.size > 1:
        weights[present] = compute_class_weight("balanced", classes=present, y=y)
    if cap is not None:
        weights = np.minimum(weights, cap)
    return torch.tensor(weights, dtype=torch.float32)


class MultiTaskLoss(nn.Module):
    """Weighted sum of per-task cross-entropy losses.

    Args:
        weights_per_task: class-weight tensor for each task.
        task_weights: multiplier for each task's loss in the total. Propaganda
            runs at 0.3 so its handful of positives cannot dominate the shared
            encoder's gradient.
    """

    def __init__(self, weights_per_task: dict[str, torch.Tensor], task_weights: dict[str, float]):
        super().__init__()
        missing = set(task_weights) - set(weights_per_task)
        if missing:
            raise ValueError(f"no class weights supplied for tasks: {sorted(missing)}")
        self.task_weights = dict(task_weights)
        # ModuleDict so .to(device) moves the class-weight buffers too
        self.criteria = nn.ModuleDict(
            {task: nn.CrossEntropyLoss(weight=weights_per_task[task]) for task in task_weights}
        )

    def forward(self, logits: dict, targets: dict):
        """Return ``(total_loss, {task: detached_loss})``.

        Logits are cast to float32 first: under bf16/fp16 autocast the
        cross-entropy reduction is where precision loss actually bites.
        """
        total = None
        parts = {}
        for task, weight in self.task_weights.items():
            loss = self.criteria[task](logits[task].float(), targets[task])
            parts[task] = loss.detach()
            total = loss * weight if total is None else total + loss * weight
        return total, parts
