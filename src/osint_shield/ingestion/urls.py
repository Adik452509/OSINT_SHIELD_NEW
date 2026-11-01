"""Canonical URLs and the keys used to spot an article seen before.

Two keys, as the roadmap asks (M10): the URL hash catches the same link polled
again; the normalised headline catches the same story arriving by a second
route - typically a Google News item for an article already taken from the
publisher's own feed.

Near-duplicates (different outlets rewriting one story) are NOT dropped: they
are separate reports, and the M2 similarity method needs the whole corpus at
once anyway (TF-IDF weights are relative to the set passed in).
"""

from __future__ import annotations

import hashlib
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from ..data.dedupe import normalise_headline

#: Query parameters that only track the click, never change the page.
_TRACKING_PREFIXES = ("utm_",)
_TRACKING_KEYS = frozenset({"fbclid", "gclid", "ref", "ref_src", "cmpid", "ito", "from",
                            "source", "pfrom", "s_cid", "icid", "mc_cid", "mc_eid", "ocid"})

#: Shorter headlines are too generic to dedupe on ("Live updates", "Top news").
MIN_HEADLINE_WORDS = 4


def canonical_url(url: str) -> str:
    """Lower-case scheme and host, drop tracking parameters, fragment and trailing slash.

    >>> canonical_url("HTTPS://WWW.Example.com/a/b/?utm_source=x&id=7#top")
    'https://www.example.com/a/b?id=7'
    """
    parts = urlsplit(url.strip())
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
             if not k.lower().startswith(_TRACKING_PREFIXES) and k.lower() not in _TRACKING_KEYS]
    path = parts.path.rstrip("/") if parts.path not in ("", "/") else ""
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path,
                       urlencode(query), ""))


def url_hash(url: str) -> str:
    """16 hex characters of SHA-1 over the canonical URL."""
    return hashlib.sha1(canonical_url(url).encode("utf-8")).hexdigest()[:16]


def article_id(url: str) -> str:
    """Stable article id, derived from the URL so re-polling cannot mint a new one."""
    return "a_" + url_hash(url)


def headline_key(headline: str) -> str | None:
    """Dedupe key for a headline, or ``None`` when it is too short to trust.

    Reuses M2's normalisation, which strips an aggregator's " - Publisher" tail -
    exactly what makes a Google News title match the publisher's own.
    """
    key = normalise_headline(headline, strip_suffix=True)
    return key if len(key.split()) >= MIN_HEADLINE_WORDS else None


def domain(url: str) -> str:
    """Host without a leading ``www.``."""
    host = urlsplit(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host
