"""Rubric keyword matching and feature extraction.

The rubric in ``data/resources/keywords.yaml`` is the scheme the annotator
used to label the corpus. It is used here in three places, none of which
silently changes the fine-tuning objective:

1. **Fusion features** appended to the pooled ``[CLS]`` vector, as a measured
   ablation. Features are computed over the *full untruncated* document,
   which is the point: at ``max_length=512`` only 17.3% of bodied articles
   are seen in full, so these carry signal from the truncated tail.
2. **Propaganda candidate mining** - the tractable route from 7 labelled
   positives to a real training set.
3. Ingestion query terms and LLM evidence-extraction constraints.

Matching is case-insensitive and word-boundary anchored, so ``aid`` does not
fire inside ``said`` and ``FIR`` does not fire inside ``FIRST``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np
import yaml

from ..paths import KEYWORDS_YAML

#: Rubric sections that map to label families.
LABEL_SECTIONS = ("severity", "propaganda", "narrative")


def _compile(term: str) -> re.Pattern[str]:
    r"""Compile one rubric term into a word-boundary anchored pattern.

    ``\b`` is unreliable next to non-word characters, so terms that start or
    end with punctuation (``counter-terrorism``, ``Indo-US``) get a lookaround
    guard on the affected side instead.
    """
    escaped = re.escape(term)
    left = r"\b" if term[:1].isalnum() else r"(?<!\w)"
    right = r"\b" if term[-1:].isalnum() else r"(?!\w)"
    return re.compile(left + escaped + right, re.IGNORECASE)


@dataclass(frozen=True)
class Rubric:
    """The keyword rubric, indexed for matching.

    Attributes:
        groups: ``"narr_Security" -> ["airstrike", ...]`` for every group in
            every labelled section.
        low_precision: terms measured as noisy on this corpus.
    """

    groups: dict[str, list[str]]
    low_precision: frozenset[str] = field(default_factory=frozenset)

    @property
    def group_names(self) -> list[str]:
        """Group keys in a stable order - the column order of the feature matrix."""
        return list(self.groups)

    @property
    def flat_terms(self) -> list[tuple[str, str]]:
        """``(group, term)`` pairs in stable order."""
        return [(g, t) for g, terms in self.groups.items() for t in terms]

    def terms_for(self, group: str, *, exclude_low_precision: bool = False) -> list[str]:
        """Terms in one group, optionally dropping the measured-noisy ones."""
        terms = self.groups[group]
        if exclude_low_precision:
            return [t for t in terms if t.lower() not in self.low_precision]
        return list(terms)


_PREFIX = {"severity": "sev", "propaganda": "prop", "narrative": "narr"}


@lru_cache(maxsize=4)
def load_rubric(path: str | None = None) -> Rubric:
    """Load and cache the rubric from YAML."""
    src = KEYWORDS_YAML if path is None else path
    with open(src, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    groups: dict[str, list[str]] = {}
    for section in LABEL_SECTIONS:
        for group, terms in (raw.get(section) or {}).items():
            groups[f"{_PREFIX[section]}_{group}"] = list(terms)

    low = frozenset(t.lower() for t in (raw.get("low_precision") or []))
    return Rubric(groups=groups, low_precision=low)


class KeywordMatcher:
    """Matches rubric terms against text and builds fusion features.

    Args:
        rubric: the rubric to match with; loaded from YAML when omitted.
        exclude_low_precision: drop terms measured as noisy on this corpus.
            Default ``False`` so fusion features stay faithful to the rubric;
            set ``True`` for propaganda mining, where precision is what matters.

    Example:
        >>> m = KeywordMatcher()
        >>> hits = m.hits("DRDO conducts missile test near the border")
        >>> hits["narr_Security"]
        ['DRDO', 'missile test']
    """

    def __init__(self, rubric: Rubric | None = None, *, exclude_low_precision: bool = False):
        self.rubric = rubric or load_rubric()
        self.exclude_low_precision = exclude_low_precision
        self._terms: dict[str, list[str]] = {
            g: self.rubric.terms_for(g, exclude_low_precision=exclude_low_precision)
            for g in self.rubric.group_names
        }
        self._patterns: dict[str, list[tuple[str, re.Pattern[str]]]] = {
            g: [(t, _compile(t)) for t in terms] for g, terms in self._terms.items()
        }

    # ---------------------------------------------------------------- matching

    def hits(self, text: str) -> dict[str, list[str]]:
        """Return the matched terms per group. Groups with no match are omitted."""
        if not text:
            return {}
        found: dict[str, list[str]] = {}
        for group, pats in self._patterns.items():
            matched = [term for term, pat in pats if pat.search(text)]
            if matched:
                found[group] = matched
        return found

    def counts(self, text: str) -> dict[str, int]:
        """Total occurrences per group (a term matching twice counts twice)."""
        if not text:
            return {g: 0 for g in self._patterns}
        return {
            group: sum(len(pat.findall(text)) for _, pat in pats)
            for group, pats in self._patterns.items()
        }

    # ---------------------------------------------------------------- features

    @property
    def group_feature_names(self) -> list[str]:
        """Column names for :meth:`group_features`."""
        return [f"grp_{g}" for g in self._patterns]

    @property
    def per_keyword_feature_names(self) -> list[str]:
        """Column names for :meth:`per_keyword_features`."""
        return [f"kw_{g}|{t}" for g, terms in self._terms.items() for t in terms]

    def group_features(
        self, texts: Sequence[str], *, transform: str = "log1p"
    ) -> np.ndarray:
        """Dense ``(n_texts, n_groups)`` matrix of per-group match counts.

        Args:
            texts: documents, passed **untruncated** - that is the whole point
                of these features.
            transform: ``"log1p"`` (default, damps long documents),
                ``"binary"``, or ``"raw"``.

        Returns:
            float32 array with columns ordered as :attr:`group_feature_names`.
        """
        rows = [list(self.counts(t or "").values()) for t in texts]
        mat = np.asarray(rows, dtype=np.float32).reshape(len(texts), len(self._patterns))
        return _apply_transform(mat, transform)

    def per_keyword_features(
        self, texts: Sequence[str], *, transform: str = "binary"
    ) -> np.ndarray:
        """Dense ``(n_texts, n_terms)`` matrix, one column per rubric term.

        More expressive than :meth:`group_features` and correspondingly more
        prone to overfitting a ~500-row training fold.
        """
        pats = [(g, t, p) for g, ps in self._patterns.items() for t, p in ps]
        rows = [[len(p.findall(txt or "")) for _, _, p in pats] for txt in texts]
        mat = np.asarray(rows, dtype=np.float32).reshape(len(texts), len(pats))
        return _apply_transform(mat, transform)

    def mark(self, text: str, template: str = "<{group}> {term} </{group}>") -> str:
        """Wrap matched terms in inline markers (the ``markers`` fusion mode).

        Unlike the feature modes this only reaches terms that survive
        truncation, so it is tested as a separate ablation arm.
        """
        if not text:
            return text
        for group, pats in self._patterns.items():
            short = group.split("_", 1)[1]
            for term, pat in pats:
                text = pat.sub(
                    lambda m: template.format(group=short, term=m.group(0)), text
                )
        return text


def _apply_transform(mat: np.ndarray, transform: str) -> np.ndarray:
    if transform == "log1p":
        return np.log1p(mat, dtype=np.float32)
    if transform == "binary":
        return (mat > 0).astype(np.float32)
    if transform == "raw":
        return mat
    raise ValueError(f"unknown transform: {transform!r}")


def build_features(
    texts: Iterable[str],
    *,
    mode: str = "group_counts",
    transform: str = "log1p",
    exclude_low_precision: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Build a keyword feature matrix for one of the fusion ablation arms.

    Args:
        texts: documents, untruncated.
        mode: ``"off"``, ``"group_counts"``, or ``"per_keyword"``.
        transform: see :meth:`KeywordMatcher.group_features`.
        exclude_low_precision: drop measured-noisy terms.

    Returns:
        ``(matrix, column_names)``. ``mode="off"`` yields a zero-width matrix
        so callers can concatenate unconditionally.
    """
    texts = list(texts)
    if mode == "off":
        return np.zeros((len(texts), 0), dtype=np.float32), []

    m = KeywordMatcher(exclude_low_precision=exclude_low_precision)
    if mode == "group_counts":
        return m.group_features(texts, transform=transform), m.group_feature_names
    if mode == "per_keyword":
        return m.per_keyword_features(texts, transform=transform), m.per_keyword_feature_names
    raise ValueError(f"unknown keyword mode: {mode!r}")
