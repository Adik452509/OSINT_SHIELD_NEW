"""The HTTP API, driven through FastAPI's TestClient with a fake classifier."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from osint_shield.api.main import create_app  # noqa: E402
from tests.test_routing import clf  # noqa: E402


@pytest.fixture
def client():
    with TestClient(create_app(clf(), max_articles=3)) as c:
        yield c


def test_root_redirects_to_the_docs(client):
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code in (302, 307)
    assert resp.headers["location"] == "/docs"


def test_health_reports_the_model_version(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_version"] == "test-v1"
    assert body["escalate_below"] == 0.70


def test_classify_returns_one_record_per_article_in_order(client):
    resp = client.post("/classify", json={"articles": [
        {"id": "a", "headline": "Army exercise", "body": "text"},
        {"id": "b", "headline": "Floods displace civilians"},
    ]})
    assert resp.status_code == 200
    body = resp.json()
    assert [r["id"] for r in body["results"]] == ["a", "b"]
    assert body["model_version"] == "test-v1"
    assert body["results"][1]["needs_review"] is True          # headline-only


def test_single_and_batch_paths_agree(client):
    article = {"id": "x", "headline": "Army exercise", "body": "text"}
    one = client.post("/classify/one", json=article).json()
    batch = client.post("/classify", json={"articles": [article]}).json()["results"][0]
    assert one == batch


def test_empty_batch_is_rejected(client):
    assert client.post("/classify", json={"articles": []}).status_code == 422


def test_missing_headline_is_rejected(client):
    assert client.post("/classify/one", json={"body": "no headline"}).status_code == 422


def test_oversized_batch_is_refused(client):
    arts = [{"headline": f"h{i}"} for i in range(4)]
    assert client.post("/classify", json={"articles": arts}).status_code == 413
