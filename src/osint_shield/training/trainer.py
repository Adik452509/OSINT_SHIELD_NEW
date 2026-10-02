"""The training loop: AMP, gradient accumulation, warmup/decay, early stopping.

The trainer knows nothing about folds or the corpus. It fits a model on one
dataset, monitors another, and predicts a third - fold bookkeeping lives in
:mod:`osint_shield.training.fold` so the same loop serves the overfit check,
a single fold, and the full cross-validation.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, fields

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from ..evaluation.metrics import score_multitask
from .callbacks import EarlyStopping
from .dataset import ArticleDataset, make_collate

_AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


@dataclass
class TrainSettings:
    """Hyperparameters for one training run. Read from ``cfg["training"]``."""

    epochs: int = 10
    early_stopping: bool = True
    patience: int = 3
    min_epochs: int = 3
    monitor: str = "narrative_macro_f1"
    batch_size: int = 8
    grad_accum: int = 2
    eval_batch_size: int = 16
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    amp: bool = True
    amp_dtype: str = "bf16"
    num_workers: int = 0

    @classmethod
    def from_config(cls, cfg: dict, **overrides) -> "TrainSettings":
        """Build from ``cfg["training"]``; ``None``-valued overrides are ignored."""
        section = cfg.get("training", {})
        kwargs = {f.name: section[f.name] for f in fields(cls) if f.name in section}
        kwargs.update({k: v for k, v in overrides.items() if v is not None})
        settings = cls(**kwargs)
        settings.lr = float(settings.lr)
        settings.weight_decay = float(settings.weight_decay)
        if settings.amp_dtype not in _AMP_DTYPES:
            raise ValueError(f"amp_dtype must be one of {sorted(_AMP_DTYPES)}")
        return settings


def linear_warmup_decay(optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    """Linear warmup to the base LR, then linear decay to zero.

    Implemented here rather than imported from ``transformers`` - its
    scheduler helpers have moved between major versions, and this is six lines.
    """
    total_steps = max(1, total_steps)

    def factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        remaining = total_steps - step
        return max(0.0, remaining / max(1, total_steps - warmup_steps))

    return LambdaLR(optimizer, factor)


def param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    """Exclude biases and normalisation weights from weight decay."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "norm" in name.lower() or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def monitor_score(metrics: dict, monitor: str) -> float:
    """The scalar early stopping watches.

    ``combined`` averages narrative macro-F1 with severity **macro**-F1, not
    severity F1(High). F1(High) rewards the degenerate all-High predictor that
    randomly initialised heads often produce in epoch 1-2 (~0.5 on a 47-row
    inner-val set), and M5 restored three such untrained checkpoints because of
    it - see docs/DECISIONS.md D11.
    """
    if monitor == "narrative_macro_f1":
        return float(metrics["narrative_macro_f1"])
    if monitor == "combined":
        return float(np.mean([metrics["narrative_macro_f1"], metrics["severity_macro_f1"]]))
    raise ValueError(f"unknown monitor: {monitor!r}")


class Trainer:
    """Fits, evaluates and predicts with a :class:`MultiTaskModel`.

    Args:
        model: the model to train; moved to ``device``.
        loss_fn: a :class:`MultiTaskLoss`.
        settings: hyperparameters.
        device: ``torch.device``.
        pad_id: the tokenizer's padding id, for collation.
        log: callable receiving one progress line at a time.
    """

    def __init__(self, model, loss_fn, settings: TrainSettings, device, pad_id: int,
                 log=print):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.settings = settings
        self.device = device
        self.log = log
        self.collate = make_collate(pad_id)
        self.best_epoch: int | None = None

        n_total = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"      parameters: {n_trainable / 1e6:.1f} M trainable of {n_total / 1e6:.1f} M")

        self.amp_enabled = bool(settings.amp) and device.type == "cuda"
        self.amp_dtype = _AMP_DTYPES[settings.amp_dtype]
        if (self.amp_enabled and self.amp_dtype is torch.bfloat16
                and not torch.cuda.is_bf16_supported()):
            log("  bf16 unsupported on this GPU - falling back to fp16")
            self.amp_dtype = torch.float16
        # loss scaling is only needed for fp16; bf16 has fp32's exponent range
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.amp_enabled and self.amp_dtype is torch.float16
        )

    # ------------------------------------------------------------------ helpers
    def _loader(self, dataset: ArticleDataset, *, shuffle: bool, batch_size: int):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=self.collate, num_workers=self.settings.num_workers)

    def _forward(self, batch: dict) -> dict:
        ids = batch["input_ids"].to(self.device, non_blocking=True)
        mask = batch["attention_mask"].to(self.device, non_blocking=True)
        kw = batch.get("keyword_feats")
        if kw is not None:
            kw = kw.to(self.device, non_blocking=True)
        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype,
                            enabled=self.amp_enabled):
            return self.model(ids, mask, kw)

    # ------------------------------------------------------------------ training
    def _train_epoch(self, loader, optimizer, scheduler) -> tuple[float, dict]:
        s = self.settings
        self.model.train()
        optimizer.zero_grad(set_to_none=True)
        total, seen = 0.0, 0
        part_sums: dict[str, float] = defaultdict(float)

        for i, batch in enumerate(loader):
            logits = self._forward(batch)
            targets = {t: batch["targets"][t].to(self.device) for t in self.loss_fn.task_weights}
            loss, parts = self.loss_fn(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss ({loss.item()}) at batch {i}. If amp_dtype is fp16, "
                    "try bf16; if bf16, set amp: false to confirm it is a precision issue."
                )

            self.scaler.scale(loss / s.grad_accum).backward()

            # step on every accumulation boundary AND on the last batch, so a
            # partial final accumulation is applied rather than silently dropped
            if (i + 1) % s.grad_accum == 0 or (i + 1) == len(loader):
                self.scaler.unscale_(optimizer)
                clip_grad_norm_(self.model.parameters(), s.max_grad_norm)
                self.scaler.step(optimizer)
                self.scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            n = batch["input_ids"].shape[0]
            total += loss.item() * n
            seen += n
            for task, value in parts.items():
                part_sums[task] += value.item() * n

        return total / max(seen, 1), {t: v / max(seen, 1) for t, v in part_sums.items()}

    def fit(self, train_ds: ArticleDataset, val_ds: ArticleDataset | None = None,
            *, track_ds: ArticleDataset | None = None,
            early_stopping: bool = True) -> list[dict]:
        """Train, monitoring ``val_ds`` if given, and restore the best epoch.

        Args:
            track_ds: optional dataset scored every epoch **for the record only**
                - it never influences checkpoint selection. Used to measure how
                well early stopping chooses, by tracking the outer test fold.

        Returns:
            One dict per epoch: losses, learning rate, validation metrics,
            the monitored score, ``track_*`` metrics and wall-clock seconds.
        """
        s = self.settings
        loader = self._loader(train_ds, shuffle=True, batch_size=s.batch_size)
        total_steps = math.ceil(len(loader) / s.grad_accum) * s.epochs
        optimizer = AdamW(param_groups(self.model, s.weight_decay), lr=s.lr)
        scheduler = linear_warmup_decay(optimizer, int(total_steps * s.warmup_ratio), total_steps)

        stopper = None
        if early_stopping and val_ds is not None:
            stopper = EarlyStopping(s.patience, mode="max", min_epochs=s.min_epochs)

        best_state = None
        history: list[dict] = []
        for epoch in range(s.epochs):
            t0 = time.time()
            train_loss, parts = self._train_epoch(loader, optimizer, scheduler)
            row = {"epoch": epoch + 1, "train_loss": train_loss,
                   "lr": scheduler.get_last_lr()[0]}
            row.update({f"train_loss_{t}": v for t, v in parts.items()})

            marker = ""
            if val_ds is not None:
                metrics = self.evaluate(val_ds)
                row.update({f"val_{k}": v for k, v in metrics.items()
                            if isinstance(v, (int, float))})
                score = monitor_score(metrics, s.monitor)
                row["monitor"] = score
                if stopper is not None and stopper.update(epoch, score):
                    best_state = {k: v.detach().to("cpu", copy=True)
                                  for k, v in self.model.state_dict().items()}
                    marker = " *"

            # scored AFTER the selection decision above, and never read by it
            if track_ds is not None:
                tracked = self._evaluate_rng_neutral(track_ds)
                row.update({f"track_{k}": v for k, v in tracked.items()
                            if isinstance(v, (int, float))})

            row["seconds"] = time.time() - t0
            history.append(row)
            self.log(self._format(row, marker))

            if stopper is not None and stopper.should_stop:
                self.log(f"      early stop after epoch {epoch + 1} "
                         f"(best: epoch {stopper.best_epoch + 1}, {stopper.best:.3f})")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.best_epoch = (stopper.best_epoch + 1 if stopper and stopper.best_epoch is not None
                           else len(history))
        return history

    def _format(self, row: dict, marker: str) -> str:
        line = (f"      epoch {row['epoch']:>2}/{self.settings.epochs}  "
                f"loss {row['train_loss']:.3f}")
        if "val_narrative_macro_f1" in row:
            line += f"  val narr-F1 {row['val_narrative_macro_f1']:.3f}"
        if "val_severity_f1_high" in row:
            line += f"  sev-F1 {row['val_severity_f1_high']:.3f}"
        if "track_narrative_macro_f1" in row:
            line += f"  [test {row['track_narrative_macro_f1']:.3f}]"
        return line + f"  {row['seconds']:5.1f}s{marker}"

    # ---------------------------------------------------------------- inference
    @torch.no_grad()
    def predict_proba(self, dataset: ArticleDataset) -> dict[str, np.ndarray]:
        """Softmax probabilities per task, ``{task: (n, n_classes)}``."""
        self.model.eval()
        loader = self._loader(dataset, shuffle=False, batch_size=self.settings.eval_batch_size)
        chunks: dict[str, list] = defaultdict(list)
        for batch in loader:
            for task, logits in self._forward(batch).items():
                chunks[task].append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        return {task: np.concatenate(parts) for task, parts in chunks.items()}

    def _evaluate_rng_neutral(self, dataset: ArticleDataset) -> dict:
        """Evaluate without disturbing any random-number stream.

        Building a DataLoader iterator draws a seed from the global generator,
        so an extra evaluation would otherwise shift every later epoch's shuffle
        and dropout masks - tracking would silently change training.
        """
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            return self.evaluate(dataset)
        finally:
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)

    def evaluate(self, dataset: ArticleDataset, has_body=None) -> dict:
        """Score the model on a dataset with :func:`score_multitask`."""
        probs = self.predict_proba(dataset)
        preds = {task: p.argmax(axis=-1) for task, p in probs.items()}
        return score_multitask(dataset.targets, preds, has_body=has_body)
