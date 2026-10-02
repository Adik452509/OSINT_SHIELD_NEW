"""Inference with a saved model.

    predictor = Predictor.from_dir("runs/m7/model_seed42")
    out = predictor.predict(df)          # df: clean_headline, clean_text
    out[["narrative", "narrative_confidence", "severity", "severity_confidence"]]

Inputs are built exactly as in training - same text join, same keyword
features (standardised inside the model by the buffers saved with it), same
truncation - because ``prepare_data`` is shared, not reimplemented. A
headline-only article is fed as the plain headline (D16).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..training.dataset import make_collate
from ..training.final import load_model
from ..training.fold import TASK_COLUMNS, prepare_data

_AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


class Predictor:
    """A loaded model plus everything needed to turn articles into labels.

    Args:
        model: a :class:`MultiTaskModel` in eval mode.
        meta: the ``meta.json`` saved alongside it.
        tokenizer: the model's tokenizer.
        device: ``torch.device`` to run on.
        batch_size: articles per forward pass.
    """

    def __init__(self, model, meta: dict, tokenizer, device, *, batch_size: int = 16):
        self.model = model.eval()
        self.meta = meta
        self.cfg = meta["config"]
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        # json turns integer keys into strings - restore them
        self.labels = {task: {int(k): v for k, v in names.items()}
                       for task, names in meta["labels"].items()}
        train_cfg = self.cfg.get("training", {})
        self.amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
        self.amp_dtype = _AMP_DTYPES.get(train_cfg.get("amp_dtype", "bf16"), torch.bfloat16)

    @classmethod
    def from_dir(cls, model_dir: Path | str, *, device=None, tokenizer=None,
                 model_builder=None, batch_size: int = 16) -> "Predictor":
        """Load a model saved by :func:`osint_shield.training.final.save_model`."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, meta = load_model(Path(model_dir), device=device, model_builder=model_builder)
        if tokenizer is None:
            from ..models import load_tokenizer

            tokenizer = load_tokenizer(meta["model_name"])
        return cls(model, meta, tokenizer, device, batch_size=batch_size)

    def _frame(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.reset_index(drop=True).copy()
        if "clean_text" not in frame:
            frame["clean_text"] = None
        if "has_body" not in frame:
            frame["has_body"] = frame["clean_text"].notna() & (
                frame["clean_text"].fillna("").astype(str).str.strip() != "")
        for col in TASK_COLUMNS.values():
            if col not in frame:
                frame[col] = 0        # placeholder - labels are never read at inference
        return frame

    @torch.no_grad()
    def predict_proba(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Softmax probabilities per task, ``{task: (n, n_classes)}``."""
        if len(df) == 0:
            return {}
        frame = self._frame(df)
        data = prepare_data(frame, self.tokenizer, self.cfg)
        loader = DataLoader(data.dataset(np.arange(len(frame))), batch_size=self.batch_size,
                            shuffle=False, collate_fn=make_collate(data.pad_id))
        chunks: dict[str, list] = {}
        for batch in loader:
            ids = batch["input_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            kw = batch.get("keyword_feats")
            if kw is not None:
                kw = kw.to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype,
                                enabled=self.amp):
                logits = self.model(ids, mask, kw)
            for task, value in logits.items():
                chunks.setdefault(task, []).append(
                    torch.softmax(value.float(), dim=-1).cpu().numpy())
        return {task: np.concatenate(parts) for task, parts in chunks.items()}

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """One row per article: a label and a confidence per task."""
        probs = self.predict_proba(df)
        out = pd.DataFrame(index=range(len(df)))
        for task, p in probs.items():
            out[task] = [self.labels[task][i] for i in p.argmax(axis=1)]
            out[f"{task}_confidence"] = p.max(axis=1)
        return out
