"""Canonical project paths.

Everything resolves from the repository root so scripts work regardless of
the directory they are invoked from. Nothing here touches the filesystem
except :func:`ensure_dirs`.
"""

from __future__ import annotations

from pathlib import Path

# src/osint_shield/paths.py -> src/osint_shield -> src -> <repo root>
ROOT: Path = Path(__file__).resolve().parents[2]

CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
DATA_RAW = DATA / "raw"
DATA_PROCESSED = DATA / "processed"
DATA_EXTERNAL = DATA / "external"
RESOURCES = DATA / "resources"
DOCS = ROOT / "docs"
SCRIPTS = ROOT / "scripts"
RUNS = ROOT / "runs"
CACHE = ROOT / ".cache"

GOLD_CSV = DATA_RAW / "gold_dataset.csv"
TEST_CSV = DATA_RAW / "test.csv"
KEYWORDS_YAML = RESOURCES / "keywords.yaml"
TAXONOMY_YAML = RESOURCES / "taxonomy.yaml"

_WRITABLE = (DATA_PROCESSED, DATA_EXTERNAL, RUNS, CACHE)


def ensure_dirs() -> None:
    """Create the output directories this project writes to."""
    for d in _WRITABLE:
        d.mkdir(parents=True, exist_ok=True)


def resolve(path: str | Path) -> Path:
    """Resolve a config-relative path against the repository root.

    Absolute paths are returned unchanged, so a config may point anywhere.
    """
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
