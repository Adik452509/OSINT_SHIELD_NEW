"""One polling pass: every source -> parse -> dedupe -> relevance gate -> store.

    with HttpFetcher() as fetch:
        report = run_ingest(store, load_sources(cfg["live"]), fetch, RelevanceGate())

For each item, in order:

0. **Blocked publisher domain** (``live.blocked_domains``) -> skip.
1. **Same URL already stored** -> skip.
2. **Same headline already stored** (normalised, " - Publisher" tail removed) ->
   skip; but if the stored copy has no body and this one does, the body is
   attached to the stored copy. A Google News headline polled first is thereby
   upgraded when the publisher's own feed brings the full text.
3. **Relevance gate** (sources with ``gate: true``) -> stored either way, as
   ``relevant`` or ``irrelevant``, so it is judged once, not every 15 minutes.

Nothing here classifies: new relevant articles wait with ``status='new'`` (M10b).
"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Callable

from ..storage import LiveStore, utc_now
from .feeds import parse_feed
from .relevance import RelevanceGate
from .sources import Source
from .urls import article_id, headline_key, url_hash


@dataclass
class SourceReport:
    """What one source produced in one pass."""

    source: str
    fetched: int = 0
    new: int = 0
    relevant: int = 0
    with_body: int = 0
    dup_url: int = 0
    dup_headline: int = 0
    upgraded: int = 0
    blocked: int = 0
    error: str | None = None


@dataclass
class IngestReport:
    started_at: str
    finished_at: str = ""
    sources: list[SourceReport] = field(default_factory=list)

    def totals(self) -> dict:
        keys = ("fetched", "new", "relevant", "with_body", "dup_url", "dup_headline", "upgraded",
                "blocked")
        out = {k: sum(getattr(s, k) for s in self.sources) for k in keys}
        out["failed_sources"] = sum(s.error is not None for s in self.sources)
        return out

    def to_dict(self) -> dict:
        return {"started_at": self.started_at, "finished_at": self.finished_at,
                "totals": self.totals(), "sources": [asdict(s) for s in self.sources]}


def is_blocked(domain: str, blocked: frozenset[str]) -> bool:
    """``domain`` is on the list, or is a subdomain of an entry on it."""
    return any(domain == b or domain.endswith("." + b) for b in blocked)


def ingest_source(store: LiveStore, source: Source, fetch: Callable[[str], bytes],
                  gate: RelevanceGate, *, min_body_chars: int = 1000, max_items: int = 200,
                  blocked_domains: frozenset[str] = frozenset(),
                  now: Callable[[], str] = utc_now) -> SourceReport:
    """Poll one source. A failing source is reported, never raised: one dead feed
    must not stop the other twenty."""
    rep = SourceReport(source.name)
    try:
        items = parse_feed(fetch(source.url), source, min_body_chars=min_body_chars,
                           max_items=max_items)
    except Exception as exc:  # noqa: BLE001 - network, HTTP and parse errors alike
        rep.error = f"{type(exc).__name__}: {exc}"[:300]
        return rep

    rep.fetched = len(items)
    hashes = [url_hash(it.url) for it in items]
    seen = store.known_url_hashes(hashes)

    for item, h in zip(items, hashes):
        if is_blocked(item.source_domain, blocked_domains):
            rep.blocked += 1
            continue
        if h in seen:
            rep.dup_url += 1
            continue
        seen.add(h)

        key = headline_key(item.headline)
        if key is not None:
            existing = store.find_by_headline_key(key)
            if existing is not None:
                rep.dup_headline += 1
                if existing["body"] is None and item.is_body:
                    store.attach_body(existing["id"], item.text, "feed")
                    rep.upgraded += 1
                continue

        if source.gate:
            relevant, terms = gate.check(item.headline, item.text)
        else:
            relevant, terms = True, []

        if not relevant:
            fetch_status, status = "skipped", "irrelevant"
        else:
            fetch_status, status = ("ok" if item.is_body else "pending"), "new"

        try:
            store.insert_article({
                "id": article_id(item.url), "url": item.url, "url_hash": h,
                "headline": item.headline, "headline_key": key,
                "body": item.text if item.is_body else None,
                "body_source": "feed" if item.is_body else None,
                "feed_text": None if item.is_body else (item.text or None),
                "fetch_status": fetch_status, "source": source.name,
                "source_kind": source.kind, "source_domain": item.source_domain,
                "language": item.language, "published_at": item.published_at,
                "collected_at": now(), "relevant": relevant, "gate_terms": terms,
                "status": status,
            })
        except sqlite3.IntegrityError:
            rep.dup_url += 1            # another process stored it a moment ago
            continue

        rep.new += 1
        rep.relevant += int(relevant)
        rep.with_body += int(relevant and item.is_body)
    return rep


def run_ingest(store: LiveStore, sources: list[Source], fetch: Callable[[str], bytes],
               gate: RelevanceGate, *, min_body_chars: int = 1000, max_items: int = 200,
               blocked_domains=(),
               on_source: Callable[[SourceReport], None] | None = None,
               record: bool = True, now: Callable[[], str] = utc_now) -> IngestReport:
    """Poll every source in order and record the pass in ``ingest_runs``.

    Args:
        blocked_domains: publisher domains never stored (and their subdomains).
        on_source: called after each source - the script prints progress with it.
        record: write the run to ``ingest_runs``; the live pipeline records its
            whole cycle instead.
        now: the clock stamped on articles - the pipeline passes its own, so its
            "give up waiting" rule compares like with like.
    """
    blocked = frozenset(d.lower().removeprefix("www.") for d in blocked_domains)
    report = IngestReport(started_at=now())
    for source in sources:
        rep = ingest_source(store, source, fetch, gate, min_body_chars=min_body_chars,
                            max_items=max_items, blocked_domains=blocked, now=now)
        report.sources.append(rep)
        if on_source is not None:
            on_source(rep)
    report.finished_at = now()
    if record:
        store.record_run(report.started_at, report.finished_at, report.to_dict())
    return report
