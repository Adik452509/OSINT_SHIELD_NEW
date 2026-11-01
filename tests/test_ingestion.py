"""Live ingestion (M10a): URLs, dedupe keys, relevance gate, feed parsing, a polling pass.

No network: feeds are built in memory and served by a fake fetcher.
"""

from __future__ import annotations

import json

import pytest

from osint_shield.config import load_config
from osint_shield.ingestion import (
    RelevanceGate,
    Source,
    canonical_url,
    headline_key,
    html_to_text,
    load_sources,
    run_ingest,
    url_hash,
)
from osint_shield.storage import LiveStore

LONG = ("The Indian Army conducted a large exercise along the northern border this week. " * 20)


# ------------------------------------------------------------------------- feeds
def rss(*items: dict, title: str = "Test feed") -> bytes:
    """A minimal RSS 2.0 document. Item keys: title, link, desc, content, pub, source_url."""
    out = []
    for it in items:
        parts = [f"<title>{it['title']}</title>"] if "title" in it else []
        if "link" in it:
            parts.append(f"<link>{it['link']}</link>")
        if "desc" in it:
            parts.append(f"<description><![CDATA[{it['desc']}]]></description>")
        if "content" in it:
            parts.append(f"<content:encoded><![CDATA[{it['content']}]]></content:encoded>")
        if "pub" in it:
            parts.append(f"<pubDate>{it['pub']}</pubDate>")
        if "source_url" in it:
            parts.append(f'<source url="{it["source_url"]}">Publisher</source>')
        out.append("<item>" + "".join(parts) + "</item>")
    return ('<?xml version="1.0" encoding="UTF-8"?><rss version="2.0" '
            'xmlns:content="http://purl.org/rss/1.0/modules/content/"><channel>'
            f"<title>{title}</title>{''.join(out)}</channel></rss>").encode("utf-8")


class FakeFetch:
    """url -> bytes, or an exception to raise. Counts calls."""

    def __init__(self, pages: dict):
        self.pages, self.calls = pages, []

    def __call__(self, url):
        self.calls.append(url)
        page = self.pages[url]
        if isinstance(page, Exception):
            raise page
        return page


@pytest.fixture
def store(tmp_path):
    return LiveStore(tmp_path / "live.db")


@pytest.fixture(scope="module")
def gate():
    return RelevanceGate()


# -------------------------------------------------------------------------- urls
def test_canonical_url_drops_tracking_fragment_and_case():
    assert (canonical_url("HTTPS://WWW.Example.com/a/b/?utm_source=x&id=7#top")
            == "https://www.example.com/a/b?id=7")


def test_url_hash_ignores_tracking_parameters():
    assert url_hash("https://x.com/story?utm_medium=rss") == url_hash("https://x.com/story")
    assert url_hash("https://x.com/story?id=1") != url_hash("https://x.com/story?id=2")


def test_headline_key_matches_google_news_title_to_publisher_title():
    publisher = "Army chief reviews security along the LoC"
    gnews = "Army chief reviews security along the LoC - The Hindu"
    assert headline_key(publisher) == headline_key(gnews) is not None


def test_short_headlines_are_not_deduplicated():
    assert headline_key("Live updates") is None


# -------------------------------------------------------------------------- gate
@pytest.mark.parametrize("text", [
    "Army chief reviews LoC security",
    "Three jawans injured in IED blast in Chhattisgarh",
    "Rafale Marine carries out first landing on INS Vikrant",
    "CRPF officers get first promotion in 15 years",
])
def test_gate_admits_defence_news(gate, text):
    assert gate.check(text)[0]


@pytest.mark.parametrize("text", [
    "ATF prices up by Rs 1.37 per litre; commercial LPG rates rise",
    "Railway Board approves Bengaluru-Mumbai Vande Bharat sleeper train",
    "Farmer earns Rs 5 lac from new crop",           # 'lac' is not LAC
])
def test_gate_rejects_unrelated_news(gate, text):
    assert not gate.check(text)[0]


def test_one_passing_mention_in_the_text_is_not_enough(gate):
    assert not gate.check("The lasting price of an apology to China",
                          "The striker's remarks about closed borders upset fans.")[0]
    # the same word twice is still one term
    assert not gate.check("Football star apologises", "Borders, borders, borders.")[0]


def test_two_different_terms_in_the_text_are_enough(gate):
    assert gate.check("Tanker reaches Odisha port safely",
                      "The vessel was hit by shrapnel after a drone attack near Hormuz.")[0]


def test_a_political_veteran_is_not_defence_news(gate):
    assert not gate.check("Big upsets in Bihar MLC polls as BJP veteran loses")[0]


def test_gate_reports_what_matched(gate):
    ok, terms = gate.check("BSF foils infiltration bid along the border")
    assert ok and {"BSF", "infiltration", "border"} <= set(terms)


def test_gate_falls_back_to_the_rubric(tmp_path):
    empty = tmp_path / "relevance.yaml"
    empty.write_text("terms: []\nacronyms: []\n", encoding="utf-8")
    g = RelevanceGate(empty)
    assert g.check("Agency conducts missile test off the coast")[0]   # rubric term
    assert not g.check("Monsoon arrives early this year")[0]


# ---------------------------------------------------------------------- parsing
def test_html_to_text_strips_tags_and_wordpress_footer():
    text = html_to_text("<p>First &amp; second.</p><script>x()</script><p>Third</p>"
                        "<p>The post Some Title appeared first on Naval News.</p>")
    assert text == "First & second.\nThird"


def test_parse_feed_full_text_teaser_and_skips():
    pytest.importorskip("feedparser")
    pytest.importorskip("bs4")
    from osint_shield.ingestion import parse_feed

    data = rss(
        {"title": "Navy commissions new frigate", "link": "https://n.com/1",
         "content": f"<p>{LONG}</p>", "pub": "Sun, 01 Nov 2026 10:20:00 +0530"},
        {"title": "Army exercise begins", "link": "https://n.com/2", "desc": "Short teaser."},
        {"title": "No link here"},
    )
    items = parse_feed(data, Source("n", "https://n.com/feed"))
    assert [i.url for i in items] == ["https://n.com/1", "https://n.com/2"]
    full, teaser = items
    assert full.is_body and full.text.startswith("The Indian Army")
    assert full.published_at == "2026-11-01T04:50:00Z"            # converted to UTC
    assert not teaser.is_body and teaser.text == "Short teaser."
    assert full.source_domain == "n.com"


def test_parse_google_news_keeps_headline_only_and_publisher_domain():
    pytest.importorskip("feedparser")
    from osint_shield.ingestion import parse_feed

    data = rss({"title": "BSF foils infiltration bid - The Hindu",
                "link": "https://news.google.com/rss/articles/CBMiXyz",
                "desc": '<a href="x">BSF foils infiltration bid</a> The Hindu',
                "source_url": "https://www.thehindu.com"})
    (item,) = parse_feed(data, Source("g", "u", kind="google_news"))
    assert item.text == "" and not item.is_body
    assert item.source_domain == "thehindu.com"


def test_parse_feed_rejects_an_html_error_page():
    pytest.importorskip("feedparser")
    from osint_shield.ingestion import parse_feed

    with pytest.raises(ValueError):
        parse_feed(b"<html><body>403 Forbidden</body></html>", Source("x", "u"))


# ---------------------------------------------------------------------- sources
def test_google_news_queries_expand_into_sources():
    srcs = load_sources({"sources": [{"name": "pub", "url": "https://p/feed"}],
                         "google_news": {"enabled": True, "window": "1d",
                                         "params": {"hl": "en-IN"},
                                         "queries": ["Indian Army", "BSF border"]}})
    assert [s.name for s in srcs] == ["pub", "gnews_indian_army", "gnews_bsf_border"]
    assert srcs[1].kind == "google_news"
    assert "q=Indian+Army+when%3A1d" in srcs[1].url and "hl=en-IN" in srcs[1].url


def test_duplicate_source_names_are_refused():
    with pytest.raises(ValueError):
        load_sources({"sources": [{"name": "a", "url": "u1"}, {"name": "a", "url": "u2"}]})


def test_live_config_loads_and_inherits_the_model():
    cfg = load_config("live.yaml")
    assert cfg["model_dir"]                                  # from serve.yaml
    sources = load_sources(cfg["live"])
    assert len({s.name for s in sources}) == len(sources)
    gnews_first = next(i for i, s in enumerate(sources) if s.kind == "google_news")
    assert all(s.kind == "rss" for s in sources[:gnews_first])   # publishers first


# ------------------------------------------------------------------------ store
def test_store_starts_empty_with_schema_version(store):
    import sqlite3

    assert store.stats()["articles"] == 0
    con = sqlite3.connect(store.path)
    assert con.execute("PRAGMA user_version").fetchone()[0] == 1
    con.close()


# ----------------------------------------------------------------------- ingest
@pytest.fixture
def feeds():
    pytest.importorskip("feedparser")
    pytest.importorskip("bs4")
    pub = Source("pub", "https://pub/feed")
    gnews = Source("gn", "https://gn/feed", kind="google_news")
    special = Source("spec", "https://spec/feed", gate=False)
    pages = {
        pub.url: rss(
            {"title": "BSF foils infiltration bid along Jammu border", "link": "https://pub/1",
             "content": f"<p>{LONG}</p>"},
            {"title": "ATF prices rise for the third month running", "link": "https://pub/2",
             "desc": "Fuel costs climb."},
            {"title": "Army begins winter exercise in Ladakh sector", "link": "https://pub/3",
             "desc": "Teaser only."},
        ),
        gnews.url: rss(
            {"title": "BSF foils infiltration bid along Jammu border - Pub",
             "link": "https://news.google.com/a1"},
            {"title": "NIA arrests two in Pahalgam attack probe - Other",
             "link": "https://news.google.com/a2"},
        ),
        special.url: rss({"title": "Belgium transfers minehunter to Bulgaria navy fleet",
                          "link": "https://spec/1", "content": f"<p>{LONG}</p>"}),
    }
    return [pub, gnews, special], FakeFetch(pages)


def test_a_polling_pass_dedupes_gates_and_stores(store, gate, feeds):
    sources, fetch = feeds
    report = run_ingest(store, sources, fetch, gate)
    by = {s.source: s for s in report.sources}

    assert (by["pub"].new, by["pub"].relevant, by["pub"].with_body) == (3, 2, 1)
    assert (by["gn"].new, by["gn"].dup_headline) == (1, 1)    # first item is pub/1 again
    assert by["spec"].relevant == 1                           # gate off

    fuel = store.get_article(_id("https://pub/2"))
    assert (fuel["relevant"], fuel["status"], fuel["fetch_status"]) == (0, "irrelevant", "skipped")
    teaser = store.get_article(_id("https://pub/3"))
    assert teaser["body"] is None and teaser["fetch_status"] == "pending"
    assert teaser["feed_text"] == "Teaser only."
    full = store.get_article(_id("https://pub/1"))
    assert full["fetch_status"] == "ok" and full["body_source"] == "feed"
    assert "BSF" in json.loads(full["gate_terms"])


def test_a_second_pass_stores_nothing_new(store, gate, feeds):
    sources, fetch = feeds
    run_ingest(store, sources, fetch, gate)
    again = run_ingest(store, sources, fetch, gate).totals()
    # 5 stored links; the Google News copy of pub/1 was never stored, so it is
    # caught by its headline again
    assert (again["new"], again["dup_url"], again["dup_headline"]) == (0, 5, 1)
    assert store.stats()["runs"] == 2


def test_publisher_full_text_upgrades_an_earlier_google_news_headline(store, gate):
    pytest.importorskip("feedparser")
    gn, pub = Source("gn", "https://gn/feed", kind="google_news"), Source("pub", "https://pub/f")
    fetch = FakeFetch({
        gn.url: rss({"title": "Army begins winter exercise in Ladakh sector - Pub",
                     "link": "https://news.google.com/x"}),
        pub.url: rss({"title": "Army begins winter exercise in Ladakh sector",
                      "link": "https://pub/9", "content": f"<p>{LONG}</p>"}),
    })
    run_ingest(store, [gn], fetch, gate)
    rep = run_ingest(store, [pub], fetch, gate).sources[0]
    assert (rep.new, rep.upgraded) == (0, 1)
    row = store.get_article(_id("https://news.google.com/x"))
    assert row["body"].startswith("The Indian Army")
    assert (row["fetch_status"], row["status"]) == ("ok", "new")


def test_a_truncated_teaser_is_not_a_body():
    pytest.importorskip("feedparser")
    from osint_shield.ingestion import parse_feed

    teaser = "Army " + "x" * 495                                   # SCMP-style 500-char cut
    (item,) = parse_feed(rss({"title": "Army exercise begins", "link": "https://s/1",
                              "desc": teaser}), Source("s", "u"))
    assert not item.is_body


def test_blocked_domains_and_their_subdomains_are_skipped(store, gate):
    pytest.importorskip("feedparser")
    gn = Source("gn", "https://gn/feed", kind="google_news")
    fetch = FakeFetch({gn.url: rss(
        {"title": "Army chief visits forward posts in Sukna - Shop", "link": "https://g/1",
         "source_url": "https://shop.ssbcrack.com"},
        {"title": "Army chief visits forward posts in Tezpur sector", "link": "https://g/2",
         "source_url": "https://www.ssbcrack.com"})})
    rep = run_ingest(store, [gn], fetch, gate,
                     blocked_domains=["shop.ssbcrack.com"]).sources[0]
    assert (rep.blocked, rep.new) == (1, 1)                         # parent domain kept


def test_one_failing_source_does_not_stop_the_rest(store, gate):
    pytest.importorskip("feedparser")
    bad, good = Source("bad", "https://bad/feed"), Source("good", "https://good/feed")
    fetch = FakeFetch({bad.url: ConnectionError("refused"),
                       good.url: rss({"title": "Navy rescues crew of sinking vessel off Goa",
                                      "link": "https://good/1"})})
    report = run_ingest(store, [bad, good], fetch, gate)
    assert report.sources[0].error.startswith("ConnectionError")
    assert report.sources[1].new == 1
    assert report.totals()["failed_sources"] == 1


def _id(url):
    from osint_shield.ingestion import article_id

    return article_id(url)
