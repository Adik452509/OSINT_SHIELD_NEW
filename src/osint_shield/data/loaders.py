"""Loading the annotated corpus into a normalised frame.

The raw CSV carries 37 columns of annotation bookkeeping. This module reduces
it to what training needs, and makes three corrections along the way:

1. Labels are mapped from the ``narrative`` **string** through
   ``taxonomy.yaml``, never from the corpus's own ``narrative_label`` column —
   those ids are non-contiguous (3 is unused, Radicalization) and would leave a
   permanently dead logit in the head.
2. ``has_body`` is materialised, because 46% of rows have no body text and the
   flag is confounded with severity strongly enough to be learnable on its own.
3. Original split membership is recovered from ``train/val/test.csv`` so
   ``test.csv`` can be held out exactly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import narrative_label_maps, severity_label_maps
from ..paths import DATA_RAW, GOLD_CSV

#: Columns kept from the raw CSV. Everything else is annotation bookkeeping.
KEEP_COLUMNS = [
    "id",
    "clean_headline",
    "clean_text",
    "original_language",
    "source_domain",
    "url",
    "published_at",
    "country",
    "state_region",
    "city",
    "narrative",
    "severity",
    "propaganda_present",
    "narrative_confidence",
    "severity_confidence",
    "label_status",
    "agree_narrative",
    "agree_severity",
    "prop_glorification",
    "prop_justification",
    "prop_victimhood",
    "prop_recruitment",
]

SPLIT_FILES = {"train": "train.csv", "val": "val.csv", "test": "test.csv"}


def load_split_ids(data_dir: Path | None = None) -> dict[str, set[str]]:
    """Read the article ids belonging to each original split."""
    root = data_dir or DATA_RAW
    ids: dict[str, set[str]] = {}
    for split, filename in SPLIT_FILES.items():
        path = root / filename
        if not path.exists():
            raise FileNotFoundError(f"missing split file: {path}")
        ids[split] = set(pd.read_csv(path, usecols=["id"], low_memory=False)["id"])
    return ids


def load_gold(
    csv_path: Path | None = None,
    *,
    data_dir: Path | None = None,
    attach_splits: bool = True,
) -> pd.DataFrame:
    """Load the full 1,064-row corpus as a normalised frame.

    Args:
        csv_path: override for ``data/raw/gold_dataset.csv``.
        data_dir: directory holding the split CSVs.
        attach_splits: add the ``split_orig`` column.

    Returns:
        One row per article with ``has_body``, ``y_narrative``, ``y_severity``,
        ``y_propaganda`` and (optionally) ``split_orig`` added.

    Raises:
        FileNotFoundError: if the corpus has not been placed in ``data/raw``.
        ValueError: if a narrative or severity value is outside the taxonomy.
    """
    path = csv_path or GOLD_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"corpus not found at {path}\n"
            "unzip data/raw/osint_dataset.zip into data/raw/ first"
        )

    raw = pd.read_csv(path, low_memory=False)
    missing = [c for c in KEEP_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"corpus is missing expected columns: {missing}")
    df = raw[KEEP_COLUMNS].copy()

    # 46% of rows have an empty body - see docs/DECISIONS.md D2.
    df["has_body"] = df["clean_text"].notna() & (
        df["clean_text"].fillna("").str.strip() != ""
    )

    narrative_to_id, _ = narrative_label_maps()
    severity_to_id, _ = severity_label_maps()

    unknown_narr = set(df["narrative"].unique()) - set(narrative_to_id)
    if unknown_narr:
        raise ValueError(
            f"narrative values absent from taxonomy.yaml: {sorted(unknown_narr)}"
        )
    unknown_sev = set(df["severity"].unique()) - set(severity_to_id)
    if unknown_sev:
        raise ValueError(f"severity values absent from taxonomy.yaml: {sorted(unknown_sev)}")

    df["y_narrative"] = df["narrative"].map(narrative_to_id).astype(int)
    df["y_severity"] = df["severity"].map(severity_to_id).astype(int)
    df["y_propaganda"] = df["propaganda_present"].astype(bool).astype(int)

    if attach_splits:
        split_ids = load_split_ids(data_dir)
        lookup = {aid: split for split, ids in split_ids.items() for aid in ids}
        df["split_orig"] = df["id"].map(lookup)
        if df["split_orig"].isna().any():
            n = int(df["split_orig"].isna().sum())
            raise ValueError(f"{n} gold rows belong to no split file")

    return df


def apply_data_mode(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Filter to the configured data mode.

    ``body_only`` keeps the 571 articles that have body text — the primary
    setting, because body-presence is confounded with severity. ``all`` keeps
    everything and is run to *measure* that confound.
    """
    if mode == "all":
        return df
    if mode == "body_only":
        return df[df["has_body"]].copy()
    raise ValueError(f"unknown data mode: {mode!r} (expected 'body_only' or 'all')")


def build_text(
    df: pd.DataFrame,
    *,
    sep: str = "</s>",
    no_body_marker: str = "[NO_BODY]",
) -> pd.Series:
    """Join headline and body into the model's input string.

    Headline-only rows are prefixed with ``no_body_marker`` so that, in
    ``mode="all"``, the model can distinguish the two collection regimes
    explicitly rather than inferring them from sequence length.

    Args:
        sep: the tokenizer's separator token — differs per model
            (``</s>`` for XLM-R, ``<eos>`` for mmBERT), so it is passed in
            rather than hardcoded.
    """
    headline = df["clean_headline"].fillna("").astype(str).str.strip()
    body = df["clean_text"].fillna("").astype(str).str.strip()
    joined = headline + f" {sep} " + body
    return joined.where(df["has_body"], no_body_marker + " " + headline)
