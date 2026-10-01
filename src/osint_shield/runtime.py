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


def gpu_memory() -> tuple[float, float] | None:
    """``(free_gb, total_gb)`` on the current CUDA device, or ``None`` without one.

    On this laptop the GPU also drives the display, so "free" moves with
    whatever windows are open - check it immediately before a long run.
    """
    import torch

    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    return free / 1024**3, total / 1024**3


def is_cuda_oom(exc: BaseException) -> bool:
    """True for CUDA out-of-memory, whichever way it surfaces.

    The caching allocator raises ``torch.OutOfMemoryError``; a library such as
    cuBLAS failing to get workspace raises a generic ``AcceleratorError`` whose
    message says "out of memory".
    """
    import torch

    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()


def set_seed(seed: int) -> None:
    """Seed every RNG a training run touches."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
