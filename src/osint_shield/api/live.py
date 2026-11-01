"""Dashboard endpoints over the live database (M10c). Mounted at ``/live``.

    GET  /live/summary                    cards: counts, views, last cycle
    GET  /live/articles?view=alerts&...   one page of a view (filters below)
    GET  /live/articles/{id}              one article, with body and review history
    POST /live/articles/{id}/review       correct a label, or "not defence news"
    GET  /live/trend?days=7               per-day counts for the chart
    GET  /live/sources                    per-source counts for filters
    GET  /live/status                     scheduler, recent cycles, Google pause
    POST /live/scheduler/start|stop       the background schedule
    POST /live/run-now                    one cycle now

Views follow the alert policy (D23): alerts, unconfirmed, analysis, review,
dismissed, all - see ``storage/dashboard.py``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, model_validator

from ..config import narrative_label_maps
from ..storage import LiveStore, utc_now
from ..storage.dashboard import VIEWS, AlertPolicy, DashboardQueries

NARRATIVES = tuple(narrative_label_maps()[0])


class ReviewIn(BaseModel):
    """An analyst's verdict. Give a corrected label, or mark the article out of scope."""

    narrative: Literal[NARRATIVES] | None = None  # type: ignore[valid-type]
    severity: Literal["High", "Low"] | None = None
    not_relevant: bool = Field(default=False, description="not defence news")
    note: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def _says_something(self):
        if not (self.narrative or self.severity or self.not_relevant):
            raise ValueError("give a narrative, a severity, or not_relevant=true")
        if self.not_relevant and (self.narrative or self.severity):
            raise ValueError("an article that is not defence news has no labels to correct")
        return self


class ScheduleIn(BaseModel):
    every_minutes: float | None = Field(default=None, ge=1, le=24 * 60)


def _ago(**kw) -> str:
    return (datetime.now(timezone.utc) - timedelta(**kw)).strftime("%Y-%m-%dT%H:%M:%SZ")


def live_router(store: LiveStore, live_cfg: dict, *, get_runner=lambda: None,
                model_version=lambda: None) -> APIRouter:
    """Build the ``/live`` routes.

    Args:
        get_runner: returns the ``CycleRunner``, or None - then the operations
            routes answer 503 (the API can serve a database another process fills).
            A callable, because the runner is created at startup, after the routes.
        model_version: callable returning the loaded model's version.
    """
    q = DashboardQueries(store, AlertPolicy.from_config(live_cfg))
    router = APIRouter(prefix="/live", tags=["live"])

    def _runner():
        runner = get_runner()
        if runner is None:
            raise HTTPException(503, "no scheduler in this server process")
        return runner

    @router.get("/summary")
    def summary():
        stats = store.stats()
        last = q.recent_cycles(1)
        return {
            "model_version": model_version(),
            "articles_stored": stats["articles"],
            "relevant": stats["relevant"],
            "classified": stats["predictions"],
            "waiting_for_text": stats["pending_fetch"],
            "views": q.counts(),
            "last_24h": q.counts(since=_ago(hours=24)),
            **q.breakdown(),
            "explanations": stats["explanations"],
            "last_cycle": last[0] if last else None,
            "google_news_paused_until": store.get_state("gnews_paused_until"),
        }

    @router.get("/articles")
    def articles(view: Literal[VIEWS] = "all",  # type: ignore[valid-type]
                 narrative: str | None = None,
                 severity: Literal["High", "Low"] | None = None,
                 source: str | None = Query(None, description="source name or domain"),
                 q_: str | None = Query(None, alias="q", description="text in the headline"),
                 since_hours: float | None = Query(None, gt=0),
                 sort: Literal["newest", "p_high"] = "newest",
                 limit: int = Query(50, ge=1, le=500),
                 offset: int = Query(0, ge=0)):
        return q.articles(view=view, narrative=narrative, severity=severity, source=source,
                          q=q_, since=_ago(hours=since_hours) if since_hours else None,
                          sort=sort, limit=limit, offset=offset)

    @router.get("/articles/{article_id}")
    def article(article_id: str):
        rec = q.article(article_id)
        if rec is None:
            raise HTTPException(404, "no classified article with that id")
        return rec

    @router.post("/articles/{article_id}/review")
    def review(article_id: str, verdict: ReviewIn):
        if store.get_prediction(article_id) is None:
            raise HTTPException(404, "no classified article with that id")
        store.save_review(article_id, narrative=verdict.narrative, severity=verdict.severity,
                          not_relevant=verdict.not_relevant, note=verdict.note,
                          reviewed_at=utc_now())
        return q.article(article_id)

    @router.get("/trend")
    def trend(days: int = Query(7, ge=1, le=90)):
        return {"days": days, "series": q.trend(since=_ago(days=days))}

    @router.get("/sources")
    def sources():
        return {"sources": q.sources(), "narratives": list(NARRATIVES)}

    @router.get("/status")
    def status():
        runner = get_runner()
        return {"scheduler": runner.status() if runner is not None else None,
                "recent_cycles": q.recent_cycles(10),
                "google_news_paused_until": store.get_state("gnews_paused_until"),
                "waiting_for_text": store.stats()["pending_fetch"]}

    @router.post("/scheduler/start")
    def start(body: ScheduleIn | None = None):
        r = _runner()
        if not r.start(body.every_minutes if body else None):
            raise HTTPException(409, "the scheduler is already running")
        return r.status()

    @router.post("/scheduler/stop")
    def stop():
        r = _runner()
        if not r.stop():
            raise HTTPException(409, "the scheduler is not running")
        return r.status()

    @router.post("/run-now", status_code=202)
    def run_now():
        r = _runner()
        if not r.run_now():
            raise HTTPException(409, "a cycle is already running")
        return r.status()

    return router
