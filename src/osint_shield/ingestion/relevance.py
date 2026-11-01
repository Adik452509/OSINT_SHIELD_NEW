"""Is this defence & security news at all? The gate in front of the classifier.

The classifier has no "not defence news" class: shown a story about fuel prices
it still answers Security, Political, ... So general-outlet feeds are filtered
first, and only articles that pass are classified (D22).

    gate = RelevanceGate()
    gate.check("Army chief reviews LoC security")   # -> (True, ['Army', 'LoC'])

Terms come from ``data/resources/relevance.yaml`` plus the rubric in
``keywords.yaml`` (low-precision terms excluded). An article passes when

* a term appears in the **headline**, or
* at least :data:`MIN_TEXT_TERMS` **different** terms appear in the text.

One passing mention deep in a long article is not enough: on the first live
pass that admitted "Mbappé's apology to China" (``borders``) and a Durga Puja
feature (``cultural``). The stricter rule cut live admissions by 10%, mostly
such items, at a cost of 0.5 points of gold recall (99.7% -> 99.2%) - D22.
"""

from __future__ import annotations

import re
from functools import lru_cache

import yaml

from ..keywords import KeywordMatcher
from ..paths import RELEVANCE_YAML

#: Matched terms kept per article - enough to audit a decision.
MAX_REPORTED = 5

#: Distinct terms needed when the headline itself has none.
MIN_TEXT_TERMS = 2


def _alternation(terms: list[str], *, ignore_case: bool) -> re.Pattern[str] | None:
    """One regex for many terms, longest first so "fighter jets" beats "fighter"."""
    if not terms:
        return None
    body = "|".join(re.escape(t) for t in sorted(set(terms), key=len, reverse=True))
    return re.compile(rf"(?<!\w)(?:{body})(?!\w)", re.IGNORECASE if ignore_case else 0)


@lru_cache(maxsize=2)
def _load(path: str) -> tuple[list[str], list[str]]:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return list(raw.get("terms") or []), list(raw.get("acronyms") or [])


class RelevanceGate:
    """Decides whether an article is in scope.

    Args:
        path: the relevance YAML; defaults to ``data/resources/relevance.yaml``.
        matcher: rubric matcher; a precision-filtered one is built when omitted.
    """

    def __init__(self, path=None, *, matcher: KeywordMatcher | None = None):
        terms, acronyms = _load(str(path or RELEVANCE_YAML))
        self._words = _alternation(terms, ignore_case=True)
        self._acronyms = _alternation(acronyms, ignore_case=False)
        self._rubric = matcher or KeywordMatcher(exclude_low_precision=True)

    def terms(self, text: str) -> list[str]:
        """Distinct matched terms in order of appearance; "Army" and "army" count once."""
        if not text:
            return []
        found: list[str] = []
        for pat in (self._words, self._acronyms):
            if pat is not None:
                found += [m.group(0) for m in pat.finditer(text)]
        found += [t for terms in self._rubric.hits(text).values() for t in terms]
        unique: dict[str, str] = {}
        for term in found:
            unique.setdefault(term.lower(), term)
        return list(unique.values())

    def check(self, headline: str, text: str = "") -> tuple[bool, list[str]]:
        """Return ``(passes, matched_terms)``; at most :data:`MAX_REPORTED` terms."""
        found = self.terms(f"{headline}\n{text}")
        passes = bool(self.terms(headline)) or len(found) >= MIN_TEXT_TERMS
        return passes, found[:MAX_REPORTED]
