"""Download a feed and turn its entries into :class:`FeedItem` records.

Parsing is separate from downloading so tests run on saved XML, with no network.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from .sources import Source
from .urls import domain

#: WordPress feeds end every item with this line; it is not article text.
_WP_FOOTER = re.compile(r"\s*The post .{1,400}? appeared first on .{1,120}?\.?\s*$", re.DOTALL)
_WS = re.compile(r"[ \t ]+")


@dataclass
class FeedItem:
    """One entry, cleaned.

    Attributes:
        text: the article text the feed carried - the full article for some
            publishers, a one-line teaser for most, empty for Google News.
        is_body: ``text`` is long enough to count as the article body. A teaser
            must not: routing treats "has a body" and "headline-only" differently
            (D17), and a 150-character teaser is neither.
    """

    source: str
    kind: str
    url: str
    headline: str
    text: str
    is_body: bool
    published_at: str | None
    source_domain: str
    language: str


def html_to_text(markup: str) -> str:
    """Strip tags, decode entities, one paragraph per line."""
    if not markup:
        return ""
    if "<" not in markup:
        text = html.unescape(markup)
    else:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(markup, "html.parser")
        for tag in soup(["script", "style", "noscript", "iframe"]):
            tag.decompose()
        text = soup.get_text("\n")
    lines = (_WS.sub(" ", line).strip() for line in text.splitlines())
    text = "\n".join(line for line in lines if line)
    return _WP_FOOTER.sub("", text).strip()


def _iso_utc(struct) -> str | None:
    """feedparser's parsed dates are UTC ``time.struct_time``."""
    if not struct:
        return None
    try:
        return datetime(*struct[:6], tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError):
        return None


def parse_feed(data: bytes, source: Source, *, min_body_chars: int = 1000,
               max_items: int = 200) -> list[FeedItem]:
    """Parse RSS/Atom bytes into items, skipping entries without a title or link.

    Raises:
        ValueError: the bytes are not a feed at all (an HTML error page, say).
    """
    import feedparser

    parsed = feedparser.parse(data)
    # feedparser accepts an HTML page without complaint and returns no entries;
    # an unrecognised format (empty `version`) is the reliable sign. Without this
    # a site that starts blocking us would look like a quiet news day.
    if not parsed.entries and (parsed.bozo or not parsed.get("version")):
        reason = parsed.get("bozo_exception") or "no RSS/Atom format recognised"
        raise ValueError(f"not a feed: {reason}")

    items = []
    for entry in parsed.entries[:max_items]:
        headline = html_to_text(entry.get("title", "")).replace("\n", " ")
        url = (entry.get("link") or "").strip()
        if not headline or not url:
            continue

        if source.kind == "google_news":
            # the "description" only repeats the headline and the publisher name
            text = ""
            origin = (entry.get("source") or {}).get("href") or url
        else:
            content = entry.get("content") or []
            raw = content[0].get("value", "") if content else entry.get("summary", "")
            text = html_to_text(raw)
            origin = url

        items.append(FeedItem(
            source=source.name, kind=source.kind, url=url, headline=headline,
            text=text, is_body=len(text) >= min_body_chars,
            published_at=_iso_utc(entry.get("published_parsed") or entry.get("updated_parsed")),
            source_domain=domain(origin), language=source.language))
    return items


class HttpFetcher:
    """Downloads feeds. One instance per run, reusing connections.

    Args:
        timeout: seconds per request.
        user_agent: identifies the pipeline to the sites it polls.
    """

    def __init__(self, *, timeout: float = 20.0, user_agent: str = "Mozilla/5.0"):
        import httpx

        self._client = httpx.Client(timeout=timeout, follow_redirects=True,
                                    headers={"User-Agent": user_agent})

    def __call__(self, url: str) -> bytes:
        resp = self._client.get(url)
        resp.raise_for_status()
        return resp.content

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
