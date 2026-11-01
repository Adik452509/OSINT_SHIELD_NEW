"""SQLite store for the live pipeline.

One file (``data/live/osint_live.db``, gitignored), standard library only. The
scheduler writes while the API reads, so the database runs in WAL mode and every
operation opens its own short-lived connection: SQLite connections must not be
shared across the threads FastAPI serves requests on.

Tables
------
articles      one row per article ever seen - relevant or not, so a story
              rejected at 10:00 is not re-judged at 10:15.
predictions   the classifier's record per article (M10b).
explanations  the LLM's summary + verbatim evidence for escalated ones (M10b).
reviews       analyst corrections from the dashboard (M10c) - how rare classes grow.
ingest_runs   one row per poll, for the dashboard's operations view.

Article lifecycle (``articles.status``):

    irrelevant                      failed the relevance gate; never classified
    new -> classified -> explained  (M10b)

``fetch_status`` tracks the body separately: ``ok`` (from the feed or the
page), ``pending`` (to fetch, M10b), ``failed``, ``skipped`` (irrelevant).
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id              TEXT PRIMARY KEY,
    url             TEXT NOT NULL,
    url_hash        TEXT NOT NULL UNIQUE,
    headline        TEXT NOT NULL,
    headline_key    TEXT,
    body            TEXT,
    body_source     TEXT,
    feed_text       TEXT,
    fetch_status    TEXT NOT NULL,
    fetch_attempts  INTEGER NOT NULL DEFAULT 0,
    source          TEXT NOT NULL,
    source_kind     TEXT NOT NULL,
    source_domain   TEXT,
    language        TEXT,
    published_at    TEXT,
    collected_at    TEXT NOT NULL,
    relevant        INTEGER NOT NULL,
    gate_terms      TEXT,
    status          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_articles_headline_key ON articles(headline_key);
CREATE INDEX IF NOT EXISTS ix_articles_status ON articles(status);
CREATE INDEX IF NOT EXISTS ix_articles_collected ON articles(collected_at);

CREATE TABLE IF NOT EXISTS predictions (
    article_id          TEXT PRIMARY KEY REFERENCES articles(id),
    model_version       TEXT NOT NULL,
    has_body            INTEGER NOT NULL,
    narrative           TEXT NOT NULL,
    narrative_conf      REAL,
    narrative_source    TEXT NOT NULL,
    severity            TEXT NOT NULL,
    severity_conf       REAL NOT NULL,
    p_high              REAL NOT NULL,
    escalated           INTEGER NOT NULL,
    escalation_reasons  TEXT NOT NULL,
    needs_review        INTEGER NOT NULL,
    review_reasons      TEXT NOT NULL,
    classified_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS explanations (
    article_id      TEXT PRIMARY KEY REFERENCES articles(id),
    llm_model       TEXT NOT NULL,
    prompt_version  TEXT NOT NULL,
    status          TEXT NOT NULL,
    summary         TEXT,
    evidence        TEXT,
    error           TEXT,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reviews (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id   TEXT NOT NULL REFERENCES articles(id),
    narrative    TEXT,
    severity     TEXT,
    note         TEXT,
    reviewed_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_reviews_article ON reviews(article_id);

CREATE TABLE IF NOT EXISTS ingest_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at   TEXT NOT NULL,
    finished_at  TEXT NOT NULL,
    report       TEXT NOT NULL
);
"""

#: Columns an article row is inserted with.
ARTICLE_COLUMNS = ("id", "url", "url_hash", "headline", "headline_key", "body", "body_source",
                   "feed_text", "fetch_status", "source", "source_kind", "source_domain",
                   "language", "published_at", "collected_at", "relevant", "gate_terms",
                   "status")


def utc_now() -> str:
    """Current time as ``YYYY-MM-DDTHH:MM:SSZ`` - the one timestamp format stored."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class LiveStore:
    """All reads and writes of the live database.

    Args:
        path: the SQLite file; created, with its folder, on first use.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as con:
            con.execute("PRAGMA journal_mode=WAL")
            version = con.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, SCHEMA_VERSION):
                raise RuntimeError(f"{self.path} has schema v{version}; this code expects "
                                   f"v{SCHEMA_VERSION}")
            con.executescript(_SCHEMA)
            con.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    @contextmanager
    def _conn(self):
        con = sqlite3.connect(self.path, timeout=10)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys=ON")
        try:
            yield con
            con.commit()
        except BaseException:
            con.rollback()
            raise
        finally:
            con.close()

    # ------------------------------------------------------------- dedupe keys
    def known_url_hashes(self, hashes: list[str]) -> set[str]:
        """Which of ``hashes`` are already stored."""
        if not hashes:
            return set()
        found: set[str] = set()
        with self._conn() as con:
            for i in range(0, len(hashes), 500):      # SQLite's parameter limit
                chunk = hashes[i:i + 500]
                rows = con.execute(f"SELECT url_hash FROM articles WHERE url_hash IN "
                                   f"({','.join('?' * len(chunk))})", chunk)
                found.update(r[0] for r in rows)
        return found

    def find_by_headline_key(self, key: str) -> sqlite3.Row | None:
        """The earliest stored article with this headline key, if any."""
        with self._conn() as con:
            return con.execute("SELECT id, body, status FROM articles WHERE headline_key = ? "
                               "ORDER BY collected_at LIMIT 1", (key,)).fetchone()

    # ------------------------------------------------------------------ writes
    def insert_article(self, row: dict) -> None:
        """Insert one article. ``gate_terms`` may be a list; it is stored as JSON."""
        values = dict(row)
        if not isinstance(values.get("gate_terms"), (str, type(None))):
            values["gate_terms"] = json.dumps(values["gate_terms"])
        values["relevant"] = int(bool(values["relevant"]))
        with self._conn() as con:
            con.execute(f"INSERT INTO articles ({','.join(ARTICLE_COLUMNS)}) VALUES "
                        f"({','.join('?' * len(ARTICLE_COLUMNS))})",
                        [values.get(c) for c in ARTICLE_COLUMNS])

    def attach_body(self, article_id: str, body: str, body_source: str) -> None:
        """Give a headline-only article its body and send it back to be (re)classified."""
        with self._conn() as con:
            con.execute("UPDATE articles SET body = ?, body_source = ?, fetch_status = 'ok', "
                        "status = CASE WHEN status = 'irrelevant' THEN status ELSE 'new' END "
                        "WHERE id = ?", (body, body_source, article_id))

    def record_run(self, started_at: str, finished_at: str, report: dict) -> int:
        with self._conn() as con:
            cur = con.execute("INSERT INTO ingest_runs (started_at, finished_at, report) "
                              "VALUES (?, ?, ?)", (started_at, finished_at, json.dumps(report)))
            return int(cur.lastrowid)

    # ------------------------------------------------------------------- reads
    def get_article(self, article_id: str) -> dict | None:
        with self._conn() as con:
            row = con.execute("SELECT * FROM articles WHERE id = ?", (article_id,)).fetchone()
        return dict(row) if row else None

    def stats(self) -> dict:
        """Counts for ``11_live.py stats`` and, later, the dashboard."""
        with self._conn() as con:
            one = lambda sql: con.execute(sql).fetchone()[0]  # noqa: E731
            by_source = con.execute(
                "SELECT source, COUNT(*) AS n, SUM(relevant) AS relevant, "
                "SUM(relevant AND body IS NOT NULL) AS with_body FROM articles "
                "GROUP BY source ORDER BY n DESC").fetchall()
            last = con.execute("SELECT finished_at FROM ingest_runs "
                               "ORDER BY id DESC LIMIT 1").fetchone()
            return {
                "articles": one("SELECT COUNT(*) FROM articles"),
                "relevant": one("SELECT COUNT(*) FROM articles WHERE relevant = 1"),
                "relevant_with_body": one("SELECT COUNT(*) FROM articles "
                                          "WHERE relevant = 1 AND body IS NOT NULL"),
                "pending_fetch": one("SELECT COUNT(*) FROM articles "
                                     "WHERE fetch_status = 'pending'"),
                "by_status": {r[0]: r[1] for r in con.execute(
                    "SELECT status, COUNT(*) FROM articles GROUP BY status")},
                "by_source": [dict(r) for r in by_source],
                "runs": one("SELECT COUNT(*) FROM ingest_runs"),
                "last_run": last[0] if last else None,
            }
