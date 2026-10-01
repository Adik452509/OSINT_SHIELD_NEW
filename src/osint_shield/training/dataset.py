"""Tokenised article datasets and dynamic-padding collation.

Texts are tokenised once, up front, rather than per ``__getitem__`` call - the
corpus is small enough to hold, and it means every epoch and every fold reuses
the same token ids instead of re-running the tokenizer.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def tokenize_texts(tokenizer, texts, max_length: int) -> tuple[list[list[int]], np.ndarray]:
    """Tokenise with truncation and report which inputs were cut.

    Returns:
        ``(input_ids, truncated)``. An input is flagged truncated when it fills
        ``max_length`` exactly - at 512 tokens that is ~83% of bodied articles,
        which is the case for keyword features computed over the full text.
    """
    enc = tokenizer(list(texts), truncation=True, max_length=max_length,
                    add_special_tokens=True)
    ids = [list(x) for x in enc["input_ids"]]
    truncated = np.array([len(x) >= max_length for x in ids], dtype=bool)
    return ids, truncated


class ArticleDataset(Dataset):
    """Pre-tokenised articles with one integer target per task.

    Args:
        input_ids: token ids per article, unpadded.
        targets: ``{task: integer label array}``, all aligned to ``input_ids``.
        keyword_feats: optional ``(n, k)`` keyword feature matrix.
    """

    def __init__(self, input_ids, targets: dict[str, np.ndarray],
                 keyword_feats: np.ndarray | None = None):
        n = len(input_ids)
        for task, values in targets.items():
            if len(values) != n:
                raise ValueError(f"target {task!r} has {len(values)} rows, expected {n}")
        if keyword_feats is not None and len(keyword_feats) != n:
            raise ValueError(f"keyword_feats has {len(keyword_feats)} rows, expected {n}")

        self.input_ids = list(input_ids)
        self.targets = {t: np.asarray(v, dtype=int) for t, v in targets.items()}
        self.keyword_feats = (
            None if keyword_feats is None else np.asarray(keyword_feats, dtype=np.float32)
        )

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, i: int) -> dict:
        return {
            "input_ids": self.input_ids[i],
            "targets": {t: int(v[i]) for t, v in self.targets.items()},
            "kw": None if self.keyword_feats is None else self.keyword_feats[i],
        }


def make_collate(pad_id: int):
    """Build a collate function that pads each batch to its own longest item.

    Padding to the batch maximum rather than ``max_length`` gives identical
    results and skips computing attention over padding for short articles.
    """
    if pad_id is None:
        raise ValueError("tokenizer has no pad token id")

    def collate(batch: list[dict]) -> dict:
        longest = max(len(item["input_ids"]) for item in batch)
        ids = torch.full((len(batch), longest), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), longest), dtype=torch.long)
        for row, item in enumerate(batch):
            n = len(item["input_ids"])
            ids[row, :n] = torch.tensor(item["input_ids"], dtype=torch.long)
            mask[row, :n] = 1

        out = {
            "input_ids": ids,
            "attention_mask": mask,
            "targets": {
                task: torch.tensor([item["targets"][task] for item in batch], dtype=torch.long)
                for task in batch[0]["targets"]
            },
        }
        if batch[0]["kw"] is not None:
            out["keyword_feats"] = torch.tensor(
                np.stack([item["kw"] for item in batch]), dtype=torch.float32
            )
        return out

    return collate
