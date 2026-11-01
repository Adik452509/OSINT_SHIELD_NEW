"""The news sources polled by the live pipeline, read from ``configs/live.yaml``.

Two kinds:

* ``rss`` - a publisher's own feed. Some (defence.in, Naval News, Indian Express)
  carry the full article text, which spares fetching the page.
* Google News keyword searches, expanded from ``google_news.queries`` into one
  feed per query. Wide coverage, but headline-only: the link is a Google
  redirect, so the body has to be fetched separately (M10b).

Order matters: sources are polled in list order and the first copy of a story
wins, so publisher feeds come before Google News.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlencode

GOOGLE_NEWS_SEARCH = "https://news.google.com/rss/search"


@dataclass(frozen=True)
class Source:
    """One feed to poll.

    Attributes:
        name: short id, stored with every article (``thehindu_national``).
        url: the feed URL.
        kind: ``rss`` or ``google_news``.
        gate: apply the relevance gate. Off for specialist defence outlets, whose
            every item is in scope even when no listed word appears
            ("Belgium transfers minehunter BNS Bellis").
        language: ISO code stored with the articles.
    """

    name: str
    url: str
    kind: str = "rss"
    gate: bool = True
    language: str = "en"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def google_news_url(query: str, *, params: dict, window: str | None = None) -> str:
    """Search-feed URL for one query; ``window="1d"`` keeps only the last day."""
    q = f"{query} when:{window}" if window else query
    return f"{GOOGLE_NEWS_SEARCH}?{urlencode({'q': q, **params})}"


def load_sources(live_cfg: dict) -> list[Source]:
    """Build the source list from the ``live`` section of the config.

    Raises:
        ValueError: two sources share a name - articles would be misattributed.
    """
    sources = [Source(name=s["name"], url=s["url"], kind=s.get("kind", "rss"),
                      gate=bool(s.get("gate", True)), language=s.get("language", "en"))
               for s in live_cfg.get("sources") or []]

    gn = live_cfg.get("google_news") or {}
    if gn.get("enabled", False):
        params = dict(gn.get("params") or {})
        for query in gn.get("queries") or []:
            sources.append(Source(
                name=f"gnews_{_slug(query)}",
                url=google_news_url(query, params=params, window=gn.get("window")),
                kind="google_news", gate=bool(gn.get("gate", True)),
                language=gn.get("language", "en")))

    names = [s.name for s in sources]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"duplicate source names in live config: {dupes}")
    return sources
