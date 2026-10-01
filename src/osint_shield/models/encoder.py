"""Shared encoder with one head per task, plus keyword fusion.

    headline </s> body  ->  encoder  ->  pooled vector (384 or 768)
                                              |
                       keyword counts  ->  standardise  ->  concat
                                              |
                          narrative head · severity head · propaganda head

The hidden size differs between candidates - 384 for mmBERT-small, 768 for
xlm-roberta-base - so it is always read from the encoder's config, never
hardcoded. (The plan PDF's diagram says 768; that is only true for XLM-R.)

``transformers`` is imported lazily so that
:func:`osint_shield.runtime.configure_environment` can set ``HF_HOME`` first.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .heads import ClassificationHead

TASKS = ("narrative", "severity", "propaganda")


def _from_pretrained(loader, name: str, **kwargs):
    """Load from the local cache first; only touch the network if that fails.

    The cache already holds both candidates (M1). Going straight to the
    network costs a metadata round-trip per file, which on a flaky connection
    means timeouts before a run has even started.
    """
    try:
        return loader.from_pretrained(name, local_files_only=True, **kwargs)
    except (OSError, ValueError):
        return loader.from_pretrained(name, **kwargs)


def load_tokenizer(name: str):
    """Load a tokenizer, preferring the local cache."""
    from transformers import AutoTokenizer

    return _from_pretrained(AutoTokenizer, name)


def load_encoder(name: str, *, attn_implementation: str = "eager",
                 gradient_checkpointing: bool = False):
    """Load a bare pretrained encoder, preferring the local cache.

    ``AutoModel`` drops the masked-LM head the checkpoint was pretrained with -
    the "UNEXPECTED keys" in the load report are exactly those, and discarding
    them is correct.
    """
    from transformers import AutoModel

    encoder = _from_pretrained(AutoModel, name, attn_implementation=attn_implementation)
    if gradient_checkpointing:
        encoder.gradient_checkpointing_enable()
    return encoder


def freeze_input_embeddings(encoder: nn.Module) -> int:
    """Stop gradient flowing into the word-embedding table.

    The table is most of each candidate's parameter count - 192 M of
    xlm-roberta-base's 278 M - yet only rows for tokens this corpus contains
    ever receive gradient. Freezing it drops their gradients and AdamW state
    (~12 bytes per parameter) and removes capacity to overfit a ~335-row fold.
    Position embeddings and the transformer body stay trainable.

    Returns:
        The number of parameters frozen.
    """
    weight = encoder.get_input_embeddings().weight
    weight.requires_grad_(False)
    return int(weight.numel())


def pool(hidden: torch.Tensor, attention_mask: torch.Tensor, mode: str = "cls") -> torch.Tensor:
    """Reduce token states to one vector per sequence.

    ``cls`` takes the first position. ``mean`` averages real tokens only, so
    padding never dilutes short articles.
    """
    if mode == "cls":
        return hidden[:, 0]
    if mode == "mean":
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    raise ValueError(f"unknown pooling mode: {mode!r}")


class MultiTaskModel(nn.Module):
    """One encoder, one linear head per enabled task, optional keyword fusion.

    Args:
        encoder: a HuggingFace encoder returning ``last_hidden_state``.
        hidden_size: the encoder's output width.
        head_sizes: ``{task: n_classes}`` for each enabled task.
        n_keyword_features: width of the keyword vector fused at the pooled
            layer; 0 disables fusion.
        dropout: applied inside each head.
        pooling: ``"cls"`` or ``"mean"``.
    """

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int,
        head_sizes: dict[str, int],
        *,
        n_keyword_features: int = 0,
        dropout: float = 0.1,
        pooling: str = "cls",
    ):
        super().__init__()
        unknown = set(head_sizes) - set(TASKS)
        if unknown:
            raise ValueError(f"unknown task heads: {sorted(unknown)}")
        if not head_sizes:
            raise ValueError("at least one head must be enabled")

        self.encoder = encoder
        self.pooling = pooling
        self.n_keyword_features = int(n_keyword_features)

        # Keyword features are standardised with the TRAINING fold's statistics,
        # stored as buffers so they travel with the model and are saved with it.
        self.register_buffer("kw_mean", torch.zeros(self.n_keyword_features))
        self.register_buffer("kw_std", torch.ones(self.n_keyword_features))

        in_dim = hidden_size + self.n_keyword_features
        self.heads = nn.ModuleDict(
            {task: ClassificationHead(in_dim, n, dropout) for task, n in head_sizes.items()}
        )

    @classmethod
    def from_config(cls, cfg: dict, n_keyword_features: int = 0,
                    encoder: nn.Module | None = None) -> "MultiTaskModel":
        """Build from a loaded config. Pass ``encoder`` to skip the download."""
        model_cfg = cfg["model"]
        if encoder is None:
            encoder = load_encoder(
                model_cfg["name"],
                attn_implementation=model_cfg.get("attn_implementation", "eager"),
                gradient_checkpointing=cfg["training"].get("gradient_checkpointing", False),
            )
        if model_cfg.get("freeze_embeddings", False):
            freeze_input_embeddings(encoder)
        head_sizes = {
            task: int(spec["n_classes"])
            for task, spec in cfg["heads"].items()
            if spec.get("enabled", True)
        }
        return cls(
            encoder,
            encoder.config.hidden_size,
            head_sizes,
            n_keyword_features=n_keyword_features,
            dropout=model_cfg.get("dropout", 0.1),
            pooling=model_cfg.get("pooling", "cls"),
        )

    def set_keyword_stats(self, mean, std) -> None:
        """Install standardisation statistics computed on the training fold.

        Zero-variance features (a rubric group that never fires in this fold)
        get std 1, so they standardise to a constant 0 instead of dividing by 0.
        """
        mean = torch.as_tensor(np.asarray(mean), dtype=torch.float32)
        std = torch.as_tensor(np.asarray(std), dtype=torch.float32)
        if mean.shape != self.kw_mean.shape or std.shape != self.kw_std.shape:
            raise ValueError(
                f"expected keyword stats of shape {tuple(self.kw_mean.shape)}, "
                f"got {tuple(mean.shape)} / {tuple(std.shape)}"
            )
        std = torch.where(std > 1e-6, std, torch.ones_like(std))
        self.kw_mean.copy_(mean)
        self.kw_std.copy_(std)

    @property
    def tasks(self) -> list[str]:
        return list(self.heads)

    def forward(self, input_ids, attention_mask, keyword_feats=None) -> dict[str, torch.Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = pool(out.last_hidden_state, attention_mask, self.pooling)

        if self.n_keyword_features:
            if keyword_feats is None:
                raise ValueError("model was built with keyword fusion but got no keyword_feats")
            # standardise in float32, then match the encoder's autocast dtype
            kw = (keyword_feats.float() - self.kw_mean) / self.kw_std
            pooled = torch.cat([pooled, kw.to(pooled.dtype)], dim=-1)

        return {task: head(pooled) for task, head in self.heads.items()}
