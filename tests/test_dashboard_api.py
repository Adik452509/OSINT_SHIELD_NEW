"""M10c: the dashboard API - views and the alert policy, reviews, operations, the lease."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from osint_shield.api.main import create_app  # noqa: E402
from osint_shield.live import CycleRunner  # noqa: E402
from osint_shield.storage import LiveStore  # noqa: E402
from tests.test_routing import clf  # noqa: E402

LIVE_CFG = {"sources": [{"name": "19fortyfive", "url": "https://www.19fortyfive.com/feed/"},
                        {"name": "pub", "url": "https://pub.example/feed"}],
            "alerts": {"analysis_sources": ["19fortyfive"], "require_body": True}}


def ago(**kw) -> str:
    return (datetime.now(timezone.utc) - timedelta(**kw)).strftime("%Y-%m-%dT%H:%M:%SZ")


def seed(store, aid, headline, *, domain="pub.example", has_body=True, severity="Low",
         p_high=0.2, narrative="Security", needs_review=False, collected=None):
    collected = collected or ago(hours=1)
    store.insert_article({
        "id": aid, "url": f"https://{domain}/{aid}", "url_hash": aid, "headline": headline,
        "headline_key": None, "body": "Article text." if has_body else None,
        "body_source": "page" if has_body else None, "feed_text": None,
        "fetch_status": "ok" if has_body else "failed", "source": "pub", "source_kind": "rss",
        "source_domain": domain, "language": "en", "published_at": collected,
        "collected_at": collected, "relevant": True, "gate_terms": ["army"], "status": "new"})
    store.save_predictions([{
        "id": aid, "model_version": "test-v1", "has_body": has_body,
        "narrative": {"label": narrative, "confidence": 0.9, "source": "encoder"},
        "severity": {"label": severity,
                     "confidence": p_high if severity == "High" else 1 - p_high},
        "escalated": severity == "High", "escalation_reasons": [],
        "needs_review": needs_review,
        "review_reasons": ["low_confidence"] if needs_review else []}], collected)


@pytest.fixture
def store(tmp_path):
    s = LiveStore(tmp_path / "live.db")
    seed(s, "a1", "Militants attack army post in Kupwara", severity="High", p_high=0.9)
    seed(s, "a2", "Army chief reviews LoC security", has_body=False, severity="High",
         p_high=0.7, needs_review=True)
    seed(s, "a3", "The F-35 summed up in one word", domain="19fortyfive.com",
         severity="High", p_high=0.8)
    seed(s, "a4", "Police arrest two with country-made guns", narrative="Investigation",
         needs_review=True)
    seed(s, "a5", "Navy commissions new frigate")
    seed(s, "a6", "Old story about a border meeting", collected=ago(days=3))
    s.save_explanation("a5", llm_model="fake", prompt_version="explain-v1", status="ok",
                       summary="The Navy commissioned a frigate.",
                       evidence=["The Navy commissioned a frigate on Monday."], error=None,
                       created_at=ago(minutes=30))
    return s


class FakeRunner:
    def __init__(self):
        self.scheduled, self.busy, self.every_minutes = False, False, 15.0

    def status(self):
        return {"scheduled": self.scheduled, "busy": self.busy,
                "every_minutes": self.every_minutes}

    def start(self, every=None):
        if self.scheduled:
            return False
        self.scheduled = True
        self.every_minutes = every or self.every_minutes
        return True

    def stop(self):
        if not self.scheduled:
            return False
        self.scheduled = False
        return True

    def run_now(self):
        if self.busy:
            return False
        self.busy = True
        return True

    def close(self):
        pass


@pytest.fixture
def client(store):
    with TestClient(create_app(clf(), serve_cfg={"live": LIVE_CFG}, store=store,
                               runner=FakeRunner())) as c:
        yield c


def ids(resp) -> list[str]:
    return sorted(item["id"] for item in resp.json()["items"])


# ---------------------------------------------------------------- views
@pytest.mark.parametrize("view, expected", [
    ("alerts", ["a1"]),                     # High + body + not analysis
    ("unconfirmed", ["a2"]),                # High, headline-only
    ("analysis", ["a3"]),                   # analysis outlet, whatever the severity
    ("review", ["a2", "a4"]),
    ("all", ["a1", "a2", "a3", "a4", "a5", "a6"]),
    ("dismissed", []),
])
def test_views_apply_the_alert_policy(client, view, expected):
    assert ids(client.get("/live/articles", params={"view": view})) == expected


def test_records_carry_category_explanation_and_labels(client):
    items = {i["id"]: i for i in client.get("/live/articles").json()["items"]}
    assert [items[k]["category"] for k in ("a1", "a2", "a3", "a5")] == \
        ["alert", "unconfirmed", "analysis", "normal"]
    assert items["a1"]["is_alert"] and items["a1"]["severity"]["p_high"] == 0.9
    assert items["a5"]["explanation"]["evidence"] == ["The Navy commissioned a frigate on Monday."]
    assert items["a1"]["explanation"] is None and items["a1"]["review"] is None


def test_filters_and_paging(client):
    assert ids(client.get("/live/articles", params={"narrative": "Investigation"})) == ["a4"]
    assert ids(client.get("/live/articles", params={"q": "frigate"})) == ["a5"]
    assert ids(client.get("/live/articles", params={"source": "19fortyfive.com"})) == ["a3"]
    assert "a6" not in ids(client.get("/live/articles", params={"since_hours": 24}))
    page = client.get("/live/articles", params={"limit": 2, "offset": 2}).json()
    assert page["total"] == 6 and len(page["items"]) == 2
    top = client.get("/live/articles", params={"sort": "p_high", "limit": 1}).json()
    assert top["items"][0]["id"] == "a1"


def test_unknown_view_is_rejected(client):
    assert client.get("/live/articles", params={"view": "everything"}).status_code == 422


def test_article_detail_has_body_and_404(client):
    rec = client.get("/live/articles/a5").json()
    assert rec["body"] == "Article text." and rec["review_history"] == []
    assert client.get("/live/articles/nope").status_code == 404


# -------------------------------------------------------------- reviews
def test_a_severity_correction_moves_an_article_into_alerts(client, store):
    resp = client.post("/live/articles/a4/review", json={"severity": "High", "note": "shots fired"})
    assert resp.status_code == 200
    rec = resp.json()
    assert rec["effective"]["severity"] == "High" and rec["category"] == "alert"
    assert rec["severity"]["label"] == "Low"                  # the model's answer is kept
    assert store.get_prediction("a4")["severity"] == "Low"
    assert ids(client.get("/live/articles", params={"view": "alerts"})) == ["a1", "a4"]
    assert "a4" not in ids(client.get("/live/articles", params={"view": "review"}))


def test_not_defence_news_dismisses_an_article(client):
    client.post("/live/articles/a1/review", json={"not_relevant": True})
    assert ids(client.get("/live/articles", params={"view": "alerts"})) == []
    assert ids(client.get("/live/articles", params={"view": "dismissed"})) == ["a1"]
    assert "a1" not in ids(client.get("/live/articles"))
    history = client.get("/live/articles/a1").json()["review_history"]
    assert history[0]["not_relevant"] is True


def test_the_latest_review_wins(client):
    client.post("/live/articles/a5/review", json={"narrative": "Political"})
    client.post("/live/articles/a5/review", json={"narrative": "Civilian"})
    rec = client.get("/live/articles/a5").json()
    assert rec["effective"]["narrative"] == "Civilian" and len(rec["review_history"]) == 2


@pytest.mark.parametrize("body", [
    {},                                                  # says nothing
    {"not_relevant": True, "narrative": "Security"},     # contradictory
    {"narrative": "Aliens"},                             # not a class
    {"severity": "Medium"},
])
def test_bad_reviews_are_rejected(client, body):
    assert client.post("/live/articles/a5/review", json=body).status_code == 422


def test_reviewing_an_unknown_article_is_404(client):
    assert client.post("/live/articles/nope/review",
                       json={"severity": "Low"}).status_code == 404


# ------------------------------------------------------- summary, trend
def test_summary(client):
    s = client.get("/live/summary").json()
    assert s["model_version"] == "test-v1" and s["classified"] == 6
    assert s["views"]["alerts"] == 1 and s["views"]["review"] == 2
    assert s["last_24h"]["all"] == 5                       # a6 is three days old
    assert s["by_severity"] == {"High": 3, "Low": 3}


def test_trend_counts_per_day(client):
    series = client.get("/live/trend", params={"days": 7}).json()["series"]
    assert sum(d["classified"] for d in series) == 6
    assert sum(d["alerts"] for d in series) == 1


def test_sources_and_narratives_for_filters(client):
    body = client.get("/live/sources").json()
    assert "Investigation" in body["narratives"]
    assert sum(s["classified"] for s in body["sources"]) == 6


# ------------------------------------------------------------ operations
def test_scheduler_controls(client):
    assert client.post("/live/scheduler/start", json={"every_minutes": 30}).status_code == 200
    assert client.post("/live/scheduler/start").status_code == 409
    assert client.get("/live/status").json()["scheduler"]["every_minutes"] == 30
    assert client.post("/live/run-now").status_code == 202
    assert client.post("/live/run-now").status_code == 409          # busy
    assert client.post("/live/scheduler/stop").status_code == 200
    assert client.post("/live/scheduler/stop").status_code == 409


def test_without_a_scheduler_operations_answer_503(store):
    with TestClient(create_app(clf(), store=store)) as c:
        assert c.get("/live/status").json()["scheduler"] is None
        assert c.post("/live/run-now").status_code == 503


def test_the_dashboard_dev_server_may_call_the_api(client):
    resp = client.get("/live/summary", headers={"Origin": "http://localhost:5173"})
    assert resp.headers["access-control-allow-origin"] == "http://localhost:5173"
    other = client.get("/live/summary", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in other.headers


def test_classify_still_works_alongside(client):
    assert client.post("/classify/one", json={"headline": "Army exercise"}).status_code == 200


# ---------------------------------------------------------------- lease
def test_lease_blocks_a_second_owner_until_released_or_expired(tmp_path):
    s = LiveStore(tmp_path / "l.db")
    t0, t1 = "2026-11-01T10:00:00Z", "2026-11-01T10:45:00Z"
    assert s.acquire_lease("cycle", "cli", now=t0, until=t1) is None
    assert s.acquire_lease("cycle", "api", now=t0, until=t1) == "cli"
    assert s.acquire_lease("cycle", "cli", now=t0, until=t1) is None          # re-entrant
    assert s.acquire_lease("cycle", "api", now="2026-11-01T11:00:00Z",
                           until="2026-11-01T11:45:00Z") is None             # expired
    s.release_lease("cycle", "cli")                                          # not the owner
    assert s.acquire_lease("cycle", "cli", now=t0, until=t1) == "api"
    s.release_lease("cycle", "api")
    assert s.acquire_lease("cycle", "cli", now=t0, until=t1) is None


def test_a_pipeline_skips_its_cycle_while_another_holds_the_lease(tmp_path):
    pytest.importorskip("feedparser")
    from tests.test_live import make_pipe

    s = LiveStore(tmp_path / "p.db")
    s.acquire_lease("cycle", "other-process", now="2000-01-01T00:00:00Z",
                    until="2999-01-01T00:00:00Z")
    rep = make_pipe(s).run_cycle()
    assert rep["skipped"].startswith("another cycle is running (other-process)")
    assert s.stats()["runs"] == 0
    s.release_lease("cycle", "other-process")
    assert "skipped" not in make_pipe(s).run_cycle()


# --------------------------------------------------------------- runner
class SlowPipeline:
    def __init__(self, fail=False):
        self.cycles, self.fail = 0, fail

    def run_cycle(self):
        time.sleep(0.2)
        self.cycles += 1
        if self.fail:
            raise RuntimeError("boom")
        return {"cycle": self.cycles}

    def close(self):
        pass


def _wait(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while not cond() and time.monotonic() < end:
        time.sleep(0.02)
    return cond()


def test_runner_runs_one_cycle_at_a_time():
    pipe = SlowPipeline()
    r = CycleRunner(lambda: pipe)
    assert r.run_now() is True
    assert _wait(lambda: r.busy)
    assert r.run_now() is False
    assert _wait(lambda: not r.busy)
    assert r.last_report == {"cycle": 1} and r.last_error is None
    r.close()


def test_runner_schedule_starts_with_a_cycle_and_stops():
    pipe = SlowPipeline()
    r = CycleRunner(lambda: pipe, every_minutes=60)
    assert r.start() is True and r.start() is False
    assert _wait(lambda: pipe.cycles == 1 and r.next_run_at is not None)
    assert r.stop() is True and r.stop() is False
    r.close()
    assert not r.scheduled


def test_runner_survives_a_failing_cycle():
    r = CycleRunner(lambda: SlowPipeline(fail=True))
    r.run_now()
    assert _wait(lambda: r.last_error is not None and not r.busy)
    assert r.last_error.startswith("RuntimeError: boom")
    r.close()
