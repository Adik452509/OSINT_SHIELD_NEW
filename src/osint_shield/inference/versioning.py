"""Model version strings.

Plan §6.3: carrying ``model_version`` on every record is what makes it possible
to tell later whether a shift in the output distribution came from the world or
from a redeployment. The version is derived from the weights themselves, so two
different models can never share one by accident.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ..training.final import MODEL_FILE


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, read in chunks so a 560 MB model never sits in memory twice."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def model_version(model_dir: Path, meta: dict) -> str:
    """``<model>-s<seed>-<first 8 hex of the weights' hash>``, e.g. ``mmbert-small-s42-3fa2b1c9``."""
    name = str(meta["model_name"]).split("/")[-1].lower()
    digest = file_sha256(Path(model_dir) / MODEL_FILE)[:8]
    return f"{name}-s{meta.get('seed', 'x')}-{digest}"
