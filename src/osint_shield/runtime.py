"""Process-level setup: HuggingFace cache location, device selection, seeding.

Call :func:`configure_environment` **before** anything imports
``transformers`` or ``huggingface_hub``: both read ``HF_HOME`` once, at import
time, so setting it afterwards silently has no effect.
"""

from __future__ import annotations

import os
import random

import numpy as np

from .paths import CACHE


def configure_environment() -> None:
    """Point the HF cache at the project and disable known-bad Windows paths."""
    os.environ.setdefault("HF_HOME", str(CACHE / "huggingface"))
    # hf_xet stalls mid-"reconstructing file" on Windows; plain HTTP is reliable.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    # fast tokenizers fork worker threads that deadlock under DataLoader workers
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def get_device(prefer: str | None = None):
    """Return ``cuda`` when available, else ``cpu`` - or the explicit preference."""
    import torch

    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """Seed every RNG a training run touches."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
