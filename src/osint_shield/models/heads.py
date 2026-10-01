"""Classification heads."""

from __future__ import annotations

from torch import nn


class ClassificationHead(nn.Module):
    """Dropout followed by a single linear layer.

    Kept deliberately shallow: with ~335 training rows per fold, an MLP head
    adds capacity to overfit without adding anything the encoder cannot
    already represent.
    """

    def __init__(self, in_dim: int, n_classes: int, dropout: float = 0.1):
        super().__init__()
        if n_classes < 2:
            raise ValueError(f"a classification head needs at least 2 classes, got {n_classes}")
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.linear(self.dropout(x))
