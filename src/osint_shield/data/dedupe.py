"""Duplicate detection over article headlines.

Two problems, handled differently:

**Exact duplicates** — the same article ingested twice (11 rows corpus-wide).
They carry no extra information, so one copy is kept.

**Near duplicates** — syndicated rewrites, ``0.85 <= similarity < 1.0``. The
text genuinely differs so both copies are kept, but they must land in the same
CV fold or the model is evaluated on something it trained on. 28 such pairs /
20 rows still leak if only exact dedupe is applied.

Similarity is cosine over character 3–5gram TF-IDF, which is robust to the
small rewrites syndication produces ("IAF chief visits France" vs "IAF Chief
visits France amid...") in a way word-level matching is not.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#: Google News and similar aggregators append " - Publisher" / " | Publisher".
_SOURCE_SUFFIX = re.compile(r"\s+[-|]\s+[^-|]{2,45}$")
_NON_ALNUM = re.compile(r"[^a-z0-9 ]")
_WS = re.compile(r"\s+")

#: Above this many rows the dense similarity matrix stops being reasonable.
_DENSE_LIMIT = 20_000


def strip_source_suffix(headline: str) -> str:
    """Remove a trailing ``" - Publisher"`` / ``" | Publisher"`` fragment."""
    return _SOURCE_SUFFIX.sub("", str(headline))


def normalise_headline(headline: str, *, strip_suffix: bool = True) -> str:
    """Lowercase, drop punctuation and collapse whitespace.

    Two headlines with the same normalised form are treated as exact
    duplicates.

    >>> normalise_headline("IAF Chief visits France! - News18")
    'iaf chief visits france'
    """
    text = str(headline)
    if strip_suffix:
        text = strip_source_suffix(text)
    # Collapse whitespace BEFORE stripping punctuation: the _NON_ALNUM class
    # keeps a literal space but deletes tabs and newlines outright, which would
    # weld "drill\tbegins" into "drillbegins" and break the match.
    text = _WS.sub(" ", text.lower())
    return _WS.sub(" ", _NON_ALNUM.sub("", text)).strip()


def components_from_similarity(sim: np.ndarray, threshold: float) -> np.ndarray:
    """Connected components of the graph whose edges are pairs >= ``threshold``.

    Transitivity is intentional and is the reason this is a graph problem
    rather than a pairwise one: if A~B and B~C but A~C falls below the
    threshold, all three still share a fold. The corpus's six near-identical
    "Pralay missile" articles chain together exactly this way, with several
    pairs below 0.85.

    Args:
        sim: square similarity matrix. The diagonal is ignored.
        threshold: edge cut-off.

    Returns:
        Integer component id per row, dense from 0.
    """
    adjacency = np.asarray(sim, dtype=float).copy()
    np.fill_diagonal(adjacency, 0.0)
    _, labels = connected_components(csr_matrix(adjacency >= threshold), directed=False)
    return labels.astype(int)


def similarity_groups(
    headlines: list[str],
    *,
    threshold: float = 0.85,
    strip_suffix: bool = True,
) -> np.ndarray:
    """Cluster headlines by character-ngram cosine similarity.

    .. warning::
       TF-IDF weights are **relative to the set passed in**. The same pair of
       headlines scores differently in a 3-document call than in a
       1,064-document one, so always pass the whole corpus at once and never
       compare similarity values across differently-sized calls.

    Args:
        headlines: raw headlines, one per row - the entire corpus.
        threshold: cosine similarity at which two headlines are "the same
            story". 0.85 was chosen by inspecting the pairs it admits.
        strip_suffix: remove aggregator source suffixes before comparing.

    Returns:
        Integer group id per row, dense from 0.

    Raises:
        ValueError: if the corpus is too large for a dense similarity matrix.
    """
    n = len(headlines)
    if n > _DENSE_LIMIT:
        raise ValueError(
            f"{n} rows exceeds the dense similarity limit ({_DENSE_LIMIT}); "
            "switch to a blocked or ANN approach before scaling up"
        )
    if n == 0:
        return np.zeros(0, dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)

    cleaned = [strip_source_suffix(h) if strip_suffix else str(h) for h in headlines]
    # min_df=1: with few rows, min_df=2 can empty the vocabulary entirely.
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
    matrix = vec.fit_transform(cleaned)

    return components_from_similarity(cosine_similarity(matrix), threshold)


def add_duplicate_columns(
    df: pd.DataFrame,
    *,
    headline_col: str = "clean_headline",
    threshold: float = 0.85,
    strip_suffix: bool = True,
) -> pd.DataFrame:
    """Return a copy of ``df`` with duplicate bookkeeping columns added.

    Adds:
        ``norm_headline``  normalised form used for exact matching.
        ``exact_group``    id shared by rows with identical ``norm_headline``.
        ``is_exact_dup``   True for every copy after the first.
        ``dupe_group``     similarity cluster id — the grouping key for folds.
    """
    out = df.copy()
    out["norm_headline"] = out[headline_col].map(
        lambda h: normalise_headline(h, strip_suffix=strip_suffix)
    )
    out["exact_group"] = out.groupby("norm_headline", sort=False).ngroup()
    out["is_exact_dup"] = out.duplicated("norm_headline", keep="first")
    out["dupe_group"] = similarity_groups(
        out[headline_col].tolist(), threshold=threshold, strip_suffix=strip_suffix
    )
    return out


def duplicate_summary(df: pd.DataFrame, *, split_col: str | None = None) -> dict:
    """Summarise duplication for the data report.

    Args:
        df: frame already passed through :func:`add_duplicate_columns`.
        split_col: if given, also count clusters spanning two original splits.
    """
    sizes = df["dupe_group"].value_counts()
    summary = {
        "n_rows": int(len(df)),
        "n_exact_dups": int(df["is_exact_dup"].sum()),
        "n_groups": int(df["dupe_group"].nunique()),
        "n_redundant_rows": int(len(df) - df["dupe_group"].nunique()),
        "largest_group": int(sizes.max()) if len(sizes) else 0,
        "n_multi_row_groups": int((sizes > 1).sum()),
    }
    if split_col is not None:
        spanning = df.groupby("dupe_group")[split_col].nunique()
        summary["n_groups_spanning_splits"] = int((spanning > 1).sum())
    return summary
