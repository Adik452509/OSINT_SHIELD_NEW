"""The live pipeline: one cycle = ingest -> fetch bodies -> classify -> explain.

    pipe = LivePipeline.from_config(load_config("live.yaml"))
    report = pipe.run_cycle()          # scripts/11_live.py run calls this every 15 min

Each stage works on what the database says is waiting, so a cycle that is
interrupted, or a stage that fails, loses nothing: the next cycle picks up the
same articles. A failing stage is reported and the others still run.

Stage rules (D22, D23):

* **fetch** - publisher pages first (cheap, plentiful), then a small, spaced
  budget of Google News decodes; a 429 from Google pauses decoding for
  ``cooldown_minutes``, remembered across restarts. An article whose body has
  not arrived within ``give_up_after_minutes`` is classified headline-only.
* **classify** - only once the fetch is settled, so an article is not
  classified headline-only minutes before its body arrives. One batch per
  cycle (D21: the batched path, fp32).
* **explain** - escalated articles with a body, High severity first, up to
  ``max_per_cycle``; if Ollama is not running the stage is skipped and retried
  next cycle.
"""

from __future__ import annotations

import os
import socket
import time
from datetime import datetime, timedelta, timezone
from typing import Callable

from ..ingestion.fetch import FetchError
from ..ingestion.gnews import DecodeError, GoogleRateLimited
from ..ingestion.ingest import is_blocked, run_ingest
from ..ingestion.urls import domain, url_hash
from ..llm.client import OllamaUnavailable
from ..llm.explain import EXPLAIN_PROMPT_VERSION, explain
from ..storage import LiveStore

GNEWS_PAUSE_KEY = "gnews_paused_until"
_FMT = "%Y-%m-%dT%H:%M:%SZ"


def _fmt(t: datetime) -> str:
    return t.strftime(_FMT)


def _release_gpu_cache() -> None:
    """Hand cached encoder memory back before Llama loads on the same 6 GB card."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class LivePipeline:
    """Runs pipeline cycles against one :class:`LiveStore`.

    Every collaborator is passed in, so tests run the whole cycle with fakes;
    :meth:`from_config` builds the real ones.

    Args:
        store: the live database.
        live_cfg: the ``live`` section of ``configs/live.yaml``.
        sources, gate, feed_fetch: as for :func:`run_ingest`.
        page_fetcher: has ``fetch(url, headline) -> text`` raising ``FetchError``.
        decoder: has ``decode(google_news_url) -> publisher_url``.
        classifier_factory: returns an ``ArticleClassifier``; called once, lazily,
            because loading the model takes seconds and GPU memory.
        llm: an ``OllamaClient`` (has ``model`` and ``generate_json``).
        clock: current UTC time; tests move it.
        log: progress messages, for the script.
    """

    def __init__(self, store: LiveStore, live_cfg: dict, *, sources, gate, feed_fetch,
                 page_fetcher, decoder, classifier_factory: Callable, llm,
                 clock: Callable[[], datetime] | None = None,
                 log: Callable[[str], None] | None = None):
        self.store = store
        self.cfg = live_cfg
        self.sources, self.gate, self.feed_fetch = sources, gate, feed_fetch
        self.page_fetcher, self.decoder, self.llm = page_fetcher, decoder, llm
        self._classifier_factory = classifier_factory
        self._classifier = None
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.log = log or (lambda msg: None)
        #: identifies this pipeline in the cycle lease
        self.owner = f"{socket.gethostname()}:{os.getpid()}:{id(self):x}"
        #: a cycle that crashed without releasing its lease blocks others this long
        self.lease_ttl = timedelta(minutes=45)

        fetch = live_cfg.get("fetch") or {}
        self.max_pages = int(fetch.get("max_pages_per_cycle", 120))
        self.max_attempts = int(fetch.get("max_attempts", 2))
        self.give_up_after = timedelta(minutes=float(fetch.get("give_up_after_minutes", 60)))
        self.unfetchable = frozenset(fetch.get("unfetchable_domains") or ())

        dec = (live_cfg.get("google_news") or {}).get("decode") or {}
        self.decode_enabled = bool(dec.get("enabled", True))
        self.max_decodes = int(dec.get("max_per_cycle", 10))
        self.cooldown = timedelta(minutes=float(dec.get("cooldown_minutes", 60)))

        ex = live_cfg.get("explain") or {}
        self.explain_enabled = bool(ex.get("enabled", True))
        self.max_explain = int(ex.get("max_per_cycle", 40))
        self.explain_chars = int(ex.get("max_chars", 6000))
        self.explain_seconds = float(ex.get("max_seconds_per_cycle", 300))

    # ------------------------------------------------------------ construction
    @classmethod
    def from_config(cls, cfg: dict, *, log=None, classifier=None) -> "LivePipeline":
        """Build the real pipeline from a loaded ``live.yaml`` (which extends serve.yaml).

        Args:
            classifier: an already-loaded ``ArticleClassifier`` to share - the API
                passes its own, so the model is not loaded twice on a 6 GB card.
        """
        from ..ingestion import HttpFetcher, RelevanceGate, load_sources
        from ..ingestion.fetch import PageFetcher
        from ..ingestion.gnews import GoogleNewsDecoder
        from ..inference import ArticleClassifier
        from ..llm import OllamaClient
        from ..paths import resolve

        live = cfg["live"]
        http = live.get("http") or {}
        timeout = float(http.get("timeout_s", 20))
        agent = http.get("user_agent", "Mozilla/5.0")
        fetch = live.get("fetch") or {}
        dec = (live.get("google_news") or {}).get("decode") or {}
        ex = live.get("explain") or {}
        return cls(
            LiveStore(resolve(live["db_path"])), live,
            sources=load_sources(live), gate=RelevanceGate(),
            feed_fetch=HttpFetcher(timeout=timeout, user_agent=agent),
            page_fetcher=PageFetcher(timeout=timeout, user_agent=agent,
                                     min_chars=int(fetch.get("min_page_chars", 400)),
                                     delay_s=float(fetch.get("delay_s", 0.5))),
            decoder=GoogleNewsDecoder(timeout=timeout,
                                      user_agent=dec.get("user_agent", agent),
                                      delay_s=float(dec.get("delay_s", 6))),
            classifier_factory=(lambda: classifier) if classifier is not None
            else (lambda: ArticleClassifier.from_config(cfg)),
            llm=OllamaClient(model=ex.get("model"), timeout=float(ex.get("timeout_s", 120))),
            log=log)

    def close(self) -> None:
        for part in (self.feed_fetch, self.page_fetcher, self.decoder):
            close = getattr(part, "close", None)
            if close is not None:
                close()

    @property
    def classifier(self):
        if self._classifier is None:
            self.log("loading the classifier ...")
            self._classifier = self._classifier_factory()
        return self._classifier

    # ------------------------------------------------------------------ cycle
    def run_cycle(self) -> dict:
        """Run all four stages; returns (and stores) the cycle report.

        Holds the ``cycle`` lease for the duration: if another process (the CLI
        loop, or the dashboard's scheduler) is mid-cycle, this one is skipped -
        reported with ``skipped``, not recorded.
        """
        now = self.clock()
        holder = self.store.acquire_lease("cycle", self.owner, now=_fmt(now),
                                          until=_fmt(now + self.lease_ttl))
        if holder is not None:
            return {"skipped": f"another cycle is running ({holder})", "errors": []}
        try:
            return self._run_stages()
        finally:
            self.store.release_lease("cycle", self.owner)

    def _run_stages(self) -> dict:
        t0 = time.monotonic()
        report: dict = {"started_at": _fmt(self.clock()), "errors": []}
        for name, stage in (("ingest", self.stage_ingest), ("fetch", self.stage_fetch),
                            ("classify", self.stage_classify), ("explain", self.stage_explain)):
            self.log(f"{name} ...")
            try:
                report[name] = stage()
            except Exception as exc:  # noqa: BLE001 - one broken stage must not stop the rest
                report[name] = {}
                report["errors"].append(f"{name}: {type(exc).__name__}: {exc}"[:400])
        report["finished_at"] = _fmt(self.clock())
        report["seconds"] = round(time.monotonic() - t0, 1)
        report["totals"] = (report.get("ingest") or {}).get("totals", {})
        self.store.record_run(report["started_at"], report["finished_at"], report)
        return report

    # ----------------------------------------------------------------- stages
    def stage_ingest(self) -> dict:
        ing = self.cfg.get("ingest") or {}
        rep = run_ingest(self.store, self.sources, self.feed_fetch, self.gate,
                         min_body_chars=int(ing.get("min_body_chars", 1000)),
                         max_items=int(ing.get("max_items_per_source", 200)),
                         blocked_domains=self.cfg.get("blocked_domains") or (),
                         record=False, now=lambda: _fmt(self.clock()))
        failed = [s.source for s in rep.sources if s.error]
        return {"totals": rep.totals(), "failed_sources": failed}

    def stage_fetch(self) -> dict:
        now = self.clock()
        out = {"pages_ok": 0, "pages_failed": 0, "pages_skipped": 0, "gnews_decoded": 0,
               "gnews_failed": 0, "duplicates": 0, "gnews_paused_until": None,
               "expired": self.store.expire_pending(_fmt(now - self.give_up_after))}

        for row in self.store.pending_fetch("rss", self.max_pages):
            self._fetch_page(row, row["url"], out)

        paused = self.store.get_state(GNEWS_PAUSE_KEY)
        if not self.decode_enabled:
            out["gnews_paused_until"] = "disabled"
            return out
        if paused and paused > _fmt(now):
            out["gnews_paused_until"] = paused
            return out

        for row in self.store.pending_fetch("google_news", self.max_decodes):
            target = row["resolved_url"]
            if not target:
                try:
                    target = self.decoder.decode(row["url"])
                except GoogleRateLimited:
                    until = _fmt(self.clock() + self.cooldown)
                    self.store.set_state(GNEWS_PAUSE_KEY, until)
                    out["gnews_paused_until"] = until
                    break
                except DecodeError as exc:
                    out["gnews_failed"] += 1
                    self.store.record_fetch_failure(
                        row["id"], f"decode: {exc}",
                        give_up=row["fetch_attempts"] + 1 >= self.max_attempts)
                    continue
                out["gnews_decoded"] += 1
                other = self.store.find_by_url_hash(url_hash(target))
                if other is not None and other != row["id"]:
                    self.store.mark_duplicate(row["id"], other, target)
                    out["duplicates"] += 1
                    continue
            self._fetch_page(row, target, out, resolved=target)
        return out

    def _fetch_page(self, row: dict, url: str, out: dict, *, resolved: str | None = None):
        if is_blocked(domain(url), self.unfetchable):
            self.store.record_fetch_failure(row["id"], "site refuses automated access",
                                            give_up=True, resolved_url=resolved)
            out["pages_skipped"] += 1
            return
        try:
            text = self.page_fetcher.fetch(url, row["headline"])
        except FetchError as exc:
            self.store.record_fetch_failure(
                row["id"], str(exc), resolved_url=resolved,
                give_up=row["fetch_attempts"] + 1 >= self.max_attempts)
            out["pages_failed"] += 1
            return
        self.store.attach_body(row["id"], text, "page", resolved_url=resolved)
        out["pages_ok"] += 1

    def stage_classify(self, batch_size: int = 64) -> dict:
        rows = self.store.to_classify()
        out = {"classified": 0, "with_body": 0, "high": 0, "escalated": 0, "needs_review": 0}
        if not rows:
            return out
        stamp = _fmt(self.clock())
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            records = self.classifier.classify([
                {"id": r["id"], "headline": r["headline"], "body": r["body"],
                 "published_at": r["published_at"]} for r in chunk])
            self.store.save_predictions(records, stamp)
            out["classified"] += len(records)
            out["with_body"] += sum(r["has_body"] for r in records)
            out["high"] += sum(r["severity"]["label"] == "High" for r in records)
            out["escalated"] += sum(r["escalated"] for r in records)
            out["needs_review"] += sum(r["needs_review"] for r in records)
        _release_gpu_cache()
        return out

    def stage_explain(self) -> dict:
        """Explain up to ``max_per_cycle`` articles, stopping early at the time budget:
        Llama's speed varies (4-5 s typical, one 13 s stall measured), and a cycle
        must finish well inside the poll interval."""
        out = {"ok": 0, "unverified": 0, "failed": 0, "quotes_dropped": 0, "skipped": None,
               "seconds_each": None}
        if not self.explain_enabled:
            out["skipped"] = "disabled"
            return out
        t0, done = time.monotonic(), 0
        for row in self.store.to_explain(self.max_explain):
            if time.monotonic() - t0 > self.explain_seconds:
                out["skipped"] = "time budget used - the rest wait for the next cycle"
                break
            try:
                exp = explain(self.llm, row["headline"], row["body"],
                              max_chars=self.explain_chars)
            except OllamaUnavailable:
                out["skipped"] = "Ollama is not running"
                break
            self.store.save_explanation(
                row["id"], llm_model=self.llm.model, prompt_version=EXPLAIN_PROMPT_VERSION,
                status=exp.status, summary=exp.summary, evidence=exp.evidence,
                error=exp.error, created_at=_fmt(self.clock()))
            out[exp.status] += 1
            out["quotes_dropped"] += exp.dropped
            done += 1
        if done:
            out["seconds_each"] = round((time.monotonic() - t0) / done, 1)
        return out
