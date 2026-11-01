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
ingest_runs   one row per pipeline cycle, for the dashboard's operations view.
state         small key/value settings that must survive a restart
              (e.g. when Google's rate limit lifts).

Article lifecycle (``articles.status``):

    irrelevant                      failed the relevance gate; never classified
    duplicate                       a Google News item that resolved to an article
                                    already stored under its publisher URL
    new -> classified -> explained  explained only if escalated and bodied

``fetch_status`` tracks the body separately: ``ok`` (from the feed or the
page), ``pending`` (still to fetch), ``failed`` (classified headline-only),
``skipped`` (irrelevant or duplicate).
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 2

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
    status          TEXT NOT NULL,
    resolved_url    TEXT,
    fetch_error     TEXT
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

CREATE TABLE IF NOT EXISTS state (
    key    TEXT PRIMARY KEY,
    value  TEXT
);
"""

#: v1 -> v2 (M10b): where a Google News link really points, and why a fetch failed.
_MIGRATE_1_TO_2 = (
    "ALTER TABLE articles ADD COLUMN resolved_url TEXT",
    "ALTER TABLE articles ADD COLUMN fetch_error TEXT",
)

#: Columns an article row is inserted with.
ARTICLE_COLUMNS = ("id", "url", "url_hash", "headline", "headline_key", "body", "body_source",
                   "feed_text", "fetch_status", "source", "source_kind", "source_domain",
                   "language", "published_at", "collected_at", "relevant", "gate_terms",
                   "status")


def utc_now() -> str:
    """Current time as ``YYYY-MM-DDTHH:MM:SSZ`` - the one timestamp format stored."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def p_high(record: dict) -> float:
    """Probability of High severity from a classifier record (binary head)."""
    sev = record["severity"]
    conf = float(sev["confidence"])
    return conf if sev["label"] == "High" else 1.0 - conf


class LiveStore:
    """All reads and writes of the live database.

    Args:
        path: the SQLite file; created, with its folder, on first use. An older
            schema is upgraded in place - collected articles are kept.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as con:
            con.execute("PRAGMA journal_mode=WAL")
            version = con.execute("PRAGMA user_version").fetchone()[0]
            if version == 1:
                for sql in _MIGRATE_1_TO_2:
                    con.execute(sql)
            elif version not in (0, SCHEMA_VERSION):
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

    def find_by_url_hash(self, url_hash: str) -> str | None:
        """Id of the article stored under this URL hash, if any."""
        with self._conn() as con:
            row = con.execute("SELECT id FROM articles WHERE url_hash = ?", (url_hash,)).fetchone()
        return row[0] if row else None

    # ---------------------------------------------------------------- ingest
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

    def attach_body(self, article_id: str, body: str, body_source: str,
                    resolved_url: str | None = None) -> None:
        """Give an article its body and send it back to be (re)classified."""
        with self._conn() as con:
            con.execute("UPDATE articles SET body = ?, body_source = ?, fetch_status = 'ok', "
                        "fetch_error = NULL, resolved_url = COALESCE(?, resolved_url), "
                        "status = CASE WHEN status IN ('irrelevant', 'duplicate') THEN status "
                        "ELSE 'new' END WHERE id = ?",
                        (body, body_source, resolved_url, article_id))

    def record_run(self, started_at: str, finished_at: str, report: dict) -> int:
        with self._conn() as con:
            cur = con.execute("INSERT INTO ingest_runs (started_at, finished_at, report) "
                              "VALUES (?, ?, ?)", (started_at, finished_at, json.dumps(report)))
            return int(cur.lastrowid)

    # ----------------------------------------------------------------- fetch
    def pending_fetch(self, kind: str, limit: int) -> list[dict]:
        """Relevant articles still waiting for a body, newest first."""
        with self._conn() as con:
            rows = con.execute(
                "SELECT id, url, headline, source_domain, resolved_url, fetch_attempts "
                "FROM articles WHERE fetch_status = 'pending' AND source_kind = ? "
                "ORDER BY collected_at DESC LIMIT ?", (kind, limit)).fetchall()
        return [dict(r) for r in rows]

    def record_fetch_failure(self, article_id: str, error: str, *, give_up: bool,
                             resolved_url: str | None = None) -> None:
        """Count a failed attempt; ``give_up`` ends the wait (classified headline-only)."""
        with self._conn() as con:
            con.execute("UPDATE articles SET fetch_attempts = fetch_attempts + 1, "
                        "fetch_error = ?, resolved_url = COALESCE(?, resolved_url), "
                        "fetch_status = CASE WHEN ? THEN 'failed' ELSE fetch_status END "
                        "WHERE id = ?", (error[:300], resolved_url, int(give_up), article_id))

    def mark_duplicate(self, article_id: str, of_id: str, resolved_url: str) -> None:
        with self._conn() as con:
            con.execute("UPDATE articles SET status = 'duplicate', fetch_status = 'skipped', "
                        "resolved_url = ?, fetch_error = ? WHERE id = ?",
                        (resolved_url, f"same article as {of_id}", article_id))

    def expire_pending(self, collected_before: str) -> int:
        """Stop waiting for bodies of articles collected before this time."""
        with self._conn() as con:
            cur = con.execute("UPDATE articles SET fetch_status = 'failed', fetch_error = "
                              "COALESCE(fetch_error, 'not fetched in time') "
                              "WHERE fetch_status = 'pending' AND collected_at < ?",
                              (collected_before,))
            return cur.rowcount

    # -------------------------------------------------------------- classify
    def to_classify(self, limit: int = 1000) -> list[dict]:
        """Relevant articles that are new and whose body fetch is settled."""
        with self._conn() as con:
            rows = con.execute(
                "SELECT id, headline, body, published_at FROM articles WHERE relevant = 1 "
                "AND status = 'new' AND fetch_status != 'pending' "
                "ORDER BY collected_at LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def save_predictions(self, records: list[dict], classified_at: str) -> None:
        """Store classifier records (M9 contract). Reclassifying drops a stale explanation."""
        with self._conn() as con:
            for r in records:
                con.execute(
                    "INSERT OR REPLACE INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (r["id"], r["model_version"], int(r["has_body"]),
                     r["narrative"]["label"], r["narrative"]["confidence"],
                     r["narrative"]["source"], r["severity"]["label"],
                     float(r["severity"]["confidence"]), round(p_high(r), 4),
                     int(r["escalated"]), json.dumps(r["escalation_reasons"]),
                     int(r["needs_review"]), json.dumps(r["review_reasons"]), classified_at))
                con.execute("DELETE FROM explanations WHERE article_id = ?", (r["id"],))
                con.execute("UPDATE articles SET status = 'classified' WHERE id = ?", (r["id"],))

    # --------------------------------------------------------------- explain
    def to_explain(self, limit: int) -> list[dict]:
        """Escalated, bodied, classified articles without an explanation; High first."""
        with self._conn() as con:
            rows = con.execute(
                "SELECT a.id, a.headline, a.body FROM articles a "
                "JOIN predictions p ON p.article_id = a.id "
                "LEFT JOIN explanations e ON e.article_id = a.id "
                "WHERE a.status = 'classified' AND p.escalated = 1 AND p.has_body = 1 "
                "AND e.article_id IS NULL ORDER BY p.p_high DESC, a.collected_at DESC LIMIT ?",
                (limit,)).fetchall()
        return [dict(r) for r in rows]

    def save_explanation(self, article_id: str, *, llm_model: str, prompt_version: str,
                         status: str, summary: str | None, evidence: list[str],
                         error: str | None, created_at: str) -> None:
        with self._conn() as con:
            con.execute("INSERT OR REPLACE INTO explanations VALUES (?,?,?,?,?,?,?,?)",
                        (article_id, llm_model, prompt_version, status, summary,
                         json.dumps(evidence), error, created_at))
            con.execute("UPDATE articles SET status = 'explained' WHERE id = ?", (article_id,))

    # ----------------------------------------------------------------- state
    def get_state(self, key: str) -> str | None:
        with self._conn() as con:
            row = con.execute("SELECT value FROM state WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_state(self, key: str, value: str | None) -> None:
        with self._conn() as con:
            con.execute("INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)", (key, value))

    # ------------------------------------------------------------------- reads
    def get_article(self, article_id: str) -> dict | None:
        with self._conn() as con:
            row = con.execute("SELECT * FROM articles WHERE id = ?", (article_id,)).fetchone()
        return dict(row) if row else None

    def get_prediction(self, article_id: str) -> dict | None:
        with self._conn() as con:
            row = con.execute("SELECT * FROM predictions WHERE article_id = ?",
                              (article_id,)).fetchone()
        return dict(row) if row else None

    def get_explanation(self, article_id: str) -> dict | None:
        with self._conn() as con:
            row = con.execute("SELECT * FROM explanations WHERE article_id = ?",
                              (article_id,)).fetchone()
        return dict(row) if row else None

    def stats(self) -> dict:
        """Counts for ``11_live.py stats`` and, later, the dashboard."""
        with self._conn() as con:
            one = lambda sql: con.execute(sql).fetchone()[0]  # noqa: E731
            group = lambda sql: {r[0]: r[1] for r in con.execute(sql)}  # noqa: E731
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
                "by_status": group("SELECT status, COUNT(*) FROM articles GROUP BY status"),
                "by_fetch_status": group("SELECT fetch_status, COUNT(*) FROM articles "
                                         "WHERE relevant = 1 GROUP BY fetch_status"),
                "body_source": group("SELECT body_source, COUNT(*) FROM articles "
                                     "WHERE relevant = 1 AND body IS NOT NULL "
                                     "GROUP BY body_source"),
                "predictions": one("SELECT COUNT(*) FROM predictions"),
                "by_narrative": group("SELECT narrative, COUNT(*) FROM predictions "
                                      "GROUP BY narrative ORDER BY 2 DESC"),
                "high_severity": one("SELECT COUNT(*) FROM predictions WHERE severity = 'High'"),
                "escalated": one("SELECT COUNT(*) FROM predictions WHERE escalated = 1"),
                "needs_review": one("SELECT COUNT(*) FROM predictions WHERE needs_review = 1"),
                "explanations": group("SELECT status, COUNT(*) FROM explanations "
                                      "GROUP BY status"),
                "by_source": [dict(r) for r in by_source],
                "runs": one("SELECT COUNT(*) FROM ingest_runs"),
                "last_run": last[0] if last else None,
            }
