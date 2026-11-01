"""M10b: page fetching, Google News decoding, LLM explanations and the pipeline cycle.

No network, no GPU, no Ollama: every outside party is a fake.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from osint_shield.ingestion import RelevanceGate, Source
from osint_shield.ingestion.fetch import FetchError, PageFetcher, matches_headline
from osint_shield.ingestion.gnews import (
    DecodeError,
    GoogleNewsDecoder,
    GoogleRateLimited,
    article_token,
    batch_payload,
    parse_decoded,
    parse_signature,
)
from osint_shield.live import LivePipeline
from osint_shield.llm import OllamaError, OllamaUnavailable, check_answer, explain
from osint_shield.storage import LiveStore
from tests.test_ingestion import FakeFetch, rss
from tests.test_routing import clf

BODY = ("Three soldiers were injured when militants opened fire on an army post in Kupwara on "
        "Monday. Security forces have launched a search operation in the area. " * 6)


# ------------------------------------------------------------------- fakes
class Resp:
    def __init__(self, status=200, text="", url="https://x/", ctype="text/html"):
        self.status_code, self.text, self.url = status, text, url
        self.headers = {"content-type": ctype}

    def iter_text(self):
        for i in range(0, len(self.text), 50):
            yield self.text[i:i + 50]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeHttp:
    """``get`` / ``stream`` / ``post`` answer from queues of responses."""

    def __init__(self, get=(), post=()):
        self.gets, self.posts = list(get), list(post)
        self.calls = []

    def get(self, url, **kw):
        self.calls.append(("GET", url))
        return self.gets.pop(0)

    def stream(self, method, url, **kw):
        self.calls.append(("STREAM", url))
        return self.gets.pop(0)

    def post(self, url, **kw):
        self.calls.append(("POST", url))
        return self.posts.pop(0)

    def close(self):
        pass


class FakePages:
    """``fetch(url, headline)`` -> text, or raises the FetchError you give it."""

    def __init__(self, pages):
        self.pages, self.calls = pages, []

    def fetch(self, url, headline):
        self.calls.append(url)
        page = self.pages.get(url, FetchError("http 404"))
        if isinstance(page, Exception):
            raise page
        return page


class FakeDecoder:
    def __init__(self, mapping):
        self.mapping, self.calls = mapping, []

    def decode(self, url):
        self.calls.append(url)
        out = self.mapping[url]
        if isinstance(out, Exception):
            raise out
        return out


class FakeLLM:
    model = "fake-llm"

    def __init__(self, answer=None, error=None):
        self.answer, self.error, self.calls = answer, error, 0

    def generate_json(self, prompt, schema, **kw):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.answer


class Clock:
    def __init__(self):
        self.t = datetime(2026, 11, 1, 10, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.t

    def advance(self, **kw):
        self.t += timedelta(**kw)


# ------------------------------------------------------------- page fetching
def test_headline_match_accepts_the_article_and_rejects_another_page():
    hl = "Militants attack army post in Kupwara, three soldiers injured - The Hindu"
    assert matches_headline(hl, "", BODY)
    assert not matches_headline(hl, "Accept cookies", "We use cookies to improve your visit.")


def test_page_fetcher_turns_bad_pages_into_fetch_errors():
    pytest.importorskip("trafilatura")
    page = (f"<html><head><title>Army post attacked in Kupwara</title></head><body>"
            f"<article><h1>Army post attacked</h1><p>{BODY}</p></article></body></html>")
    hl = "Militants attack army post in Kupwara, soldiers injured"
    cases = [(Resp(403), "http 403"),
             (Resp(text="<html><body><p>Subscribe to read.</p></body></html>"), "too short"),
             (Resp(text="%PDF", ctype="application/pdf"), "not a web page")]
    for resp, expected in cases:
        f = PageFetcher(client=FakeHttp(get=[resp]), delay_s=0)
        with pytest.raises(FetchError, match=expected):
            f.fetch("https://p/1", hl)

    ok = PageFetcher(client=FakeHttp(get=[Resp(text=page)]), delay_s=0)
    assert "Kupwara" in ok.fetch("https://p/1", hl)
    wrong = PageFetcher(client=FakeHttp(get=[Resp(text=page)]), delay_s=0)
    with pytest.raises(FetchError, match="does not match"):
        wrong.fetch("https://p/1", "Cricket board announces new tournament schedule for women")


# ----------------------------------------------------------- Google News decode
GN = "https://news.google.com/rss/articles/CBMiAbC123?oc=5"
GN_PAGE = '<html><c-wiz data-n-a-sg="SIG42" data-n-a-ts="1793511912"></c-wiz></html>'
GN_ANSWER = ')]}\'\n[["wrb.fr","Fbv4je","[\\"garturlres\\",\\"https://www.thehindu.com/news/a1.ece\\",1]"]]'


def test_google_news_helpers():
    assert article_token(GN) == "CBMiAbC123"
    assert parse_signature(GN_PAGE) == ("SIG42", 1793511912)
    assert parse_signature("<html></html>") is None
    assert "CBMiAbC123" in batch_payload("CBMiAbC123", 1, "S").replace("%22", '"')
    assert parse_decoded(GN_ANSWER) == "https://www.thehindu.com/news/a1.ece"
    assert parse_decoded("nothing here") is None


def test_decoder_resolves_a_link():
    http = FakeHttp(get=[Resp(text=GN_PAGE)], post=[Resp(text=GN_ANSWER)])
    assert (GoogleNewsDecoder(client=http, delay_s=0).decode(GN)
            == "https://www.thehindu.com/news/a1.ece")
    assert [c[0] for c in http.calls] == ["STREAM", "POST"]


@pytest.mark.parametrize("page", [Resp(429), Resp(200, url="https://www.google.com/sorry/index")])
def test_decoder_reports_rate_limiting(page):
    with pytest.raises(GoogleRateLimited):
        GoogleNewsDecoder(client=FakeHttp(get=[page]), delay_s=0).decode(GN)


def test_decoder_reports_a_changed_layout():
    with pytest.raises(DecodeError):
        GoogleNewsDecoder(client=FakeHttp(get=[Resp(text="<html>new layout</html>")]),
                          delay_s=0).decode(GN)


# ----------------------------------------------------------------- explanations
QUOTE = "Three soldiers were injured when militants opened fire on an army post in Kupwara"


def test_only_quotes_found_in_the_article_survive():
    exp = check_answer({"evidence": [QUOTE, "Five terrorists were killed in the gunfight", "army"],
                        "summary": "Militants attacked an army post."}, "Headline", BODY)
    assert exp.status == "ok" and exp.evidence == [QUOTE] and exp.dropped == 2


def test_quotes_survive_curly_quotes_and_spacing():
    body = "The minister said “we will respond firmly” to any   provocation at the border."
    exp = check_answer({"evidence": ['said "we will respond firmly" to any provocation'],
                        "summary": "s"}, "h", body)
    assert exp.status == "ok"


def test_summary_without_any_real_quote_is_unverified_not_evidence():
    exp = check_answer({"evidence": ["An invented sentence about a battle."],
                        "summary": "Something happened."}, "h", BODY)
    assert exp.status == "unverified" and exp.evidence == []


def test_empty_answer_is_a_failure():
    assert check_answer({"evidence": [], "summary": ""}, "h", BODY).status == "failed"


def test_explain_separates_ollama_down_from_a_bad_answer():
    with pytest.raises(OllamaUnavailable):
        explain(FakeLLM(error=OllamaUnavailable("down")), "h", BODY)
    assert explain(FakeLLM(error=OllamaError("no valid JSON")), "h", BODY).status == "failed"


# ------------------------------------------------------------------- migration
def test_a_version_1_database_is_upgraded_in_place(tmp_path):
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE articles (id TEXT PRIMARY KEY, url TEXT NOT NULL, url_hash TEXT "
                "NOT NULL UNIQUE, headline TEXT NOT NULL, headline_key TEXT, body TEXT, "
                "body_source TEXT, feed_text TEXT, fetch_status TEXT NOT NULL, fetch_attempts "
                "INTEGER NOT NULL DEFAULT 0, source TEXT NOT NULL, source_kind TEXT NOT NULL, "
                "source_domain TEXT, language TEXT, published_at TEXT, collected_at TEXT NOT NULL, "
                "relevant INTEGER NOT NULL, gate_terms TEXT, status TEXT NOT NULL)")
    con.execute("INSERT INTO articles (id, url, url_hash, headline, fetch_status, source, "
                "source_kind, collected_at, relevant, status) VALUES "
                "('a_1', 'u', 'h', 'Army exercise', 'pending', 's', 'rss', 't', 1, 'new')")
    con.execute("PRAGMA user_version=1")
    con.commit()
    con.close()

    store = LiveStore(path)
    row = store.get_article("a_1")
    assert row["headline"] == "Army exercise" and row["resolved_url"] is None
    con = sqlite3.connect(path)
    assert con.execute("PRAGMA user_version").fetchone()[0] == 3
    con.close()


# ---------------------------------------------------------------------- pipeline
PUB, GNEWS = Source("pub", "https://pub/feed"), Source("gn", "https://gn/feed", kind="google_news")


def feeds():
    return FakeFetch({
        PUB.url: rss(
            {"title": "Militants attack army post in Kupwara", "link": "https://pub/1",
             "desc": "Teaser."},
            {"title": "Navy frigate shadows PLA vessel in Indian Ocean",
             "link": "https://www.scmp.com/n/2", "desc": "Teaser."}),
        GNEWS.url: rss(
            {"title": "Army chief reviews LoC security posture - Other",
             "link": "https://news.google.com/rss/articles/T1"},
            {"title": "BSF foils infiltration bid along Jammu border - Pub2",
             "link": "https://news.google.com/rss/articles/T2"}),
    })


def make_pipe(store, *, decoder=None, llm=None, clock=None, factory=None, cfg=None):
    pages = FakePages({"https://pub/1": BODY, "https://other.example/7": BODY})
    return LivePipeline(
        store, cfg or {"fetch": {"unfetchable_domains": ["scmp.com"], "max_attempts": 2,
                                 "give_up_after_minutes": 60}},
        sources=[PUB, GNEWS], gate=RelevanceGate(), feed_fetch=feeds(), page_fetcher=pages,
        decoder=decoder or FakeDecoder({
            "https://news.google.com/rss/articles/T1": "https://other.example/7",
            "https://news.google.com/rss/articles/T2": "https://pub/1"}),   # = pub/1: duplicate
        classifier_factory=factory or (lambda: clf(severity=(0.2, 0.8))),  # all High
        llm=llm or FakeLLM({"evidence": [QUOTE, "An invented sentence."], "summary": "S."}),
        clock=clock or Clock())


@pytest.fixture
def store(tmp_path):
    pytest.importorskip("feedparser")
    return LiveStore(tmp_path / "live.db")


def _by_url(store, url):
    from osint_shield.ingestion import article_id

    return article_id(url)


def test_a_full_cycle(store):
    rep = make_pipe(store).run_cycle()
    assert rep["errors"] == []
    f = rep["fetch"]
    assert (f["pages_ok"], f["pages_skipped"], f["duplicates"], f["gnews_decoded"]) == (2, 1, 1, 2)

    # SCMP blocks us -> classified headline-only; the Google copy of pub/1 is a duplicate
    scmp = store.get_article(_by_url(store, "https://www.scmp.com/n/2"))
    assert (scmp["fetch_status"], scmp["status"]) == ("failed", "classified")
    dup = store.get_article(_by_url(store, "https://news.google.com/rss/articles/T2"))
    assert dup["status"] == "duplicate" and store.get_prediction(dup["id"]) is None

    # the decoded Google item got its publisher URL and the page text
    gn = store.get_article(_by_url(store, "https://news.google.com/rss/articles/T1"))
    assert gn["resolved_url"] == "https://other.example/7" and gn["body_source"] == "page"

    assert rep["classify"]["classified"] == 3 and rep["classify"]["with_body"] == 2
    # explained: escalated AND bodied only; the invented quote is dropped
    assert (rep["explain"]["ok"], rep["explain"]["quotes_dropped"]) == (2, 2)
    exp = store.get_explanation(gn["id"])
    assert json.loads(exp["evidence"]) == [QUOTE] and exp["llm_model"] == "fake-llm"
    assert store.get_explanation(scmp["id"]) is None
    assert store.stats()["runs"] == 1


def test_a_second_cycle_has_nothing_left_to_do(store):
    pipe = make_pipe(store)
    pipe.run_cycle()
    rep = pipe.run_cycle()
    assert rep["classify"]["classified"] == 0 and rep["explain"]["ok"] == 0
    assert pipe.llm.calls == 2


def test_google_rate_limit_pauses_decoding_across_restarts(store):
    clock = Clock()
    limited = FakeDecoder({"https://news.google.com/rss/articles/T1": GoogleRateLimited("429"),
                           "https://news.google.com/rss/articles/T2": GoogleRateLimited("429")})
    rep = make_pipe(store, decoder=limited, clock=clock).run_cycle()
    assert rep["fetch"]["gnews_paused_until"] == "2026-11-01T11:00:00Z"
    assert len(limited.calls) == 1                         # stopped at the first 429

    clock.advance(minutes=30)                              # a fresh pipeline: a restart
    later = FakeDecoder({})
    make_pipe(store, decoder=later, clock=clock).run_cycle()
    assert later.calls == []                               # still paused


def test_articles_stop_waiting_for_a_body(store):
    clock = Clock()
    off = {"fetch": {"give_up_after_minutes": 60, "unfetchable_domains": ["scmp.com"]},
           "google_news": {"decode": {"enabled": False}}}
    pipe = make_pipe(store, clock=clock, cfg=off)
    first = pipe.run_cycle()
    gn_id = _by_url(store, "https://news.google.com/rss/articles/T1")
    assert store.get_article(gn_id)["fetch_status"] == "pending"
    assert store.get_prediction(gn_id) is None             # not classified headline-only yet
    assert first["classify"]["classified"] == 2            # the two publisher items

    clock.advance(minutes=61)
    second = pipe.run_cycle()
    assert second["fetch"]["expired"] == 2
    assert store.get_prediction(gn_id)["has_body"] == 0


def test_ollama_down_skips_explanations_and_retries_next_cycle(store):
    llm = FakeLLM(error=OllamaUnavailable("connection refused"))
    pipe = make_pipe(store, llm=llm)
    rep = pipe.run_cycle()
    assert rep["explain"]["skipped"] == "Ollama is not running" and llm.calls == 1
    assert store.stats()["explanations"] == {}

    llm.error, llm.answer = None, {"evidence": [QUOTE], "summary": "S."}
    assert pipe.run_cycle()["explain"]["ok"] == 2


def test_explaining_stops_at_the_time_budget(store):
    cfg = {"fetch": {"unfetchable_domains": ["scmp.com"]},
           "explain": {"max_seconds_per_cycle": -1}}          # already spent
    pipe = make_pipe(store, cfg=cfg)
    rep = pipe.run_cycle()
    assert rep["explain"]["ok"] == 0 and rep["explain"]["skipped"].startswith("time budget")
    assert pipe.llm.calls == 0


def test_a_broken_stage_does_not_stop_the_cycle(store):
    def broken():
        raise RuntimeError("CUDA out of memory")

    rep = make_pipe(store, factory=broken).run_cycle()
    assert any(e.startswith("classify: RuntimeError") for e in rep["errors"])
    assert rep["fetch"]["pages_ok"] == 2 and rep["explain"]["ok"] == 0
    assert store.stats()["runs"] == 1
