"""Training callbacks."""

from __future__ import annotations


class EarlyStopping:
    """Stop when the monitored score has not improved for ``patience`` epochs.

    Args:
        patience: epochs without improvement before stopping.
        mode: ``"max"`` for scores, ``"min"`` for losses.
        min_delta: improvement smaller than this does not count.
        min_epochs: never stop before this many epochs have run. Guards the
            early phase where randomly initialised heads still predict the
            majority class and the score is flat for reasons unrelated to
            overfitting.
    """

    def __init__(self, patience: int, *, mode: str = "max", min_delta: float = 0.0,
                 min_epochs: int = 0):
        if mode not in {"max", "min"}:
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.best: float | None = None
        self.best_epoch: int | None = None
        self.bad_epochs = 0
        self._epochs_seen = 0

    def _is_better(self, score: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "max":
            return score > self.best + self.min_delta
        return score < self.best - self.min_delta

    def update(self, epoch: int, score: float) -> bool:
        """Record one epoch's score. Returns True if it is a new best."""
        self._epochs_seen = epoch + 1
        if self._is_better(score):
            self.best, self.best_epoch, self.bad_epochs = score, epoch, 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience and self._epochs_seen >= self.min_epochs
