"""Read models for the dashboard (M10c): article lists, summary, trend.

Everything here reads; the only write the dashboard makes is a review
(:meth:`LiveStore.save_review`).

**Effective labels.** An analyst's latest review overrides the model's label for
display and filtering; the prediction itself is never changed, so model and
analyst can still be compared later.

**Views** - the alert policy (D23, user choice):

    alerts       effective severity High, has a body, not an analysis outlet
    unconfirmed  effective severity High, headline-only, not an analysis outlet
    analysis     from an analysis outlet (19FortyFive, The Diplomat), any severity
    review       flagged needs_review by routing and not reviewed yet
    dismissed    reviewed as "not defence news"
    all          everything classified except dismissed
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .db import LiveStore

VIEWS = ("all", "alerts", "unconfirmed", "analysis", "review", "dismissed")

_SELECT = """
SELECT a.id, a.headline, a.url, a.resolved_url, a.source, a.source_kind, a.source_domain,
       a.published_at, a.collected_at, a.body_source, a.gate_terms,
       p.model_version, p.has_body, p.narrative, p.narrative_conf, p.narrative_source,
       p.severity, p.p_high, p.escalated, p.escalation_reasons, p.needs_review,
       p.review_reasons, p.classified_at,
       e.status AS ex_status, e.summary AS ex_summary, e.evidence AS ex_evidence,
       e.llm_model AS ex_model,
       r.narrative AS r_narrative, r.severity AS r_severity, r.not_relevant AS r_not_relevant,
       r.note AS r_note, r.reviewed_at AS r_reviewed_at,
       COALESCE(r.narrative, p.narrative) AS eff_narrative,
       COALESCE(r.severity, p.severity) AS eff_severity
FROM predictions p
JOIN articles a ON a.id = p.article_id
LEFT JOIN explanations e ON e.article_id = a.id
LEFT JOIN latest_review r ON r.article_id = a.id
"""

_NOT_DISMISSED = "COALESCE(r.not_relevant, 0) = 0"


@dataclass(frozen=True)
class AlertPolicy:
    """Which High-severity articles count as alerts (``live.alerts`` in live.yaml)."""

    analysis_domains: frozenset[str] = frozenset()
    require_body: bool = True

    @classmethod
    def from_config(cls, live_cfg: dict) -> "AlertPolicy":
        """Analysis outlets are named by source; matched by domain, so the same
        outlet arriving through Google News is recognised too."""
        from ..ingestion.urls import domain

        alerts = live_cfg.get("alerts") or {}
        names = set(alerts.get("analysis_sources") or ())
        domains = {domain(s["url"]) for s in live_cfg.get("sources") or [] if s["name"] in names}
        return cls(frozenset(domains), bool(alerts.get("require_body", True)))


class DashboardQueries:
    def __init__(self, store: LiveStore, policy: AlertPolicy):
        self.store = store
        self.policy = policy

    # ------------------------------------------------------------- SQL pieces
    def _analysis_sql(self) -> tuple[str, list]:
        domains = sorted(self.policy.analysis_domains)
        if not domains:
            return "0", []
        return f"a.source_domain IN ({','.join('?' * len(domains))})", domains

    def _view_sql(self, view: str) -> tuple[str, list]:
        analysis, params = self._analysis_sql()
        high = "COALESCE(r.severity, p.severity) = 'High'"
        body = "p.has_body = 1" if self.policy.require_body else "1"
        if view == "all":
            return _NOT_DISMISSED, []
        if view == "alerts":
            return f"{_NOT_DISMISSED} AND {high} AND {body} AND NOT ({analysis})", params
        if view == "unconfirmed":
            return (f"{_NOT_DISMISSED} AND {high} AND p.has_body = 0 AND NOT ({analysis})",
                    params)
        if view == "analysis":
            return f"{_NOT_DISMISSED} AND ({analysis})", params
        if view == "review":
            return "p.needs_review = 1 AND r.id IS NULL", []
        if view == "dismissed":
            return "COALESCE(r.not_relevant, 0) = 1", []
        raise ValueError(f"unknown view {view!r}; choose from {VIEWS}")

    # ------------------------------------------------------------------ rows
    def _record(self, row) -> dict:
        analysis = row["source_domain"] in self.policy.analysis_domains
        high = row["eff_severity"] == "High"
        dismissed = bool(row["r_not_relevant"])
        if dismissed:
            category = "dismissed"
        elif analysis:
            category = "analysis"
        elif high and (row["has_body"] or not self.policy.require_body):
            category = "alert"
        elif high:
            category = "unconfirmed"
        else:
            category = "normal"
        reviewed = row["r_reviewed_at"] is not None
        return {
            "id": row["id"],
            "headline": row["headline"],
            "url": row["resolved_url"] or row["url"],
            "source": row["source"],
            "source_kind": row["source_kind"],
            "source_domain": row["source_domain"],
            "published_at": row["published_at"],
            "collected_at": row["collected_at"],
            "has_body": bool(row["has_body"]),
            "body_source": row["body_source"],
            "matched_terms": json.loads(row["gate_terms"] or "[]"),
            "narrative": {"label": row["narrative"], "confidence": row["narrative_conf"],
                          "source": row["narrative_source"]},
            "severity": {"label": row["severity"], "p_high": row["p_high"]},
            "effective": {"narrative": row["eff_narrative"], "severity": row["eff_severity"]},
            "category": category,
            "is_alert": category == "alert",
            "escalated": bool(row["escalated"]),
            "escalation_reasons": json.loads(row["escalation_reasons"]),
            "needs_review": bool(row["needs_review"]),
            "review_reasons": json.loads(row["review_reasons"]),
            "model_version": row["model_version"],
            "classified_at": row["classified_at"],
            "explanation": None if row["ex_status"] is None else {
                "status": row["ex_status"], "summary": row["ex_summary"],
                "evidence": json.loads(row["ex_evidence"] or "[]"), "model": row["ex_model"]},
            "review": None if not reviewed else {
                "narrative": row["r_narrative"], "severity": row["r_severity"],
                "not_relevant": dismissed, "note": row["r_note"],
                "reviewed_at": row["r_reviewed_at"]},
        }

    def articles(self, *, view: str = "all", narrative: str | None = None,
                 severity: str | None = None, source: str | None = None,
                 q: str | None = None, since: str | None = None, sort: str = "newest",
                 limit: int = 50, offset: int = 0) -> dict:
        """One page of a view, newest first (or by P(High) with ``sort="p_high"``)."""
        where, params = self._view_sql(view)
        clauses = [where]
        if narrative:
            clauses.append("COALESCE(r.narrative, p.narrative) = ?")
            params.append(narrative)
        if severity:
            clauses.append("COALESCE(r.severity, p.severity) = ?")
            params.append(severity)
        if source:
            clauses.append("(a.source = ? OR a.source_domain = ?)")
            params += [source, source]
        if q:
            clauses.append("a.headline LIKE ?")
            params.append(f"%{q}%")
        if since:
            clauses.append("a.collected_at >= ?")
            params.append(since)
        cond = " AND ".join(f"({c})" for c in clauses)
        order = ("p.p_high DESC" if sort == "p_high"
                 else "COALESCE(a.published_at, a.collected_at) DESC")
        with self.store._conn() as con:
            total = con.execute(f"SELECT COUNT(*) FROM ({_SELECT} WHERE {cond})",
                                params).fetchone()[0]
            rows = con.execute(f"{_SELECT} WHERE {cond} ORDER BY {order}, a.id LIMIT ? OFFSET ?",
                               [*params, limit, offset]).fetchall()
        return {"view": view, "total": total, "limit": limit, "offset": offset,
                "items": [self._record(r) for r in rows]}

    def article(self, article_id: str) -> dict | None:
        """One classified article in full, with its body and review history."""
        with self.store._conn() as con:
            row = con.execute(f"{_SELECT} WHERE a.id = ?", (article_id,)).fetchone()
            if row is None:
                return None
            body = con.execute("SELECT body, feed_text, fetch_error FROM articles WHERE id = ?",
                               (article_id,)).fetchone()
            history = con.execute("SELECT narrative, severity, not_relevant, note, reviewed_at "
                                  "FROM reviews WHERE article_id = ? ORDER BY id",
                                  (article_id,)).fetchall()
        rec = self._record(row)
        rec.update(body=body["body"], feed_text=body["feed_text"], fetch_error=body["fetch_error"],
                   review_history=[{**dict(h), "not_relevant": bool(h["not_relevant"])}
                                   for h in history])
        return rec

    def counts(self, since: str | None = None) -> dict[str, int]:
        """Size of every view, optionally only for articles collected since a time."""
        out = {}
        for view in VIEWS:
            where, params = self._view_sql(view)
            if since:
                where, params = f"({where}) AND a.collected_at >= ?", [*params, since]
            with self.store._conn() as con:
                out[view] = con.execute(f"SELECT COUNT(*) FROM ({_SELECT} WHERE {where})",
                                        params).fetchone()[0]
        return out

    def breakdown(self) -> dict:
        """Effective narrative and severity counts over everything not dismissed."""
        with self.store._conn() as con:
            rows = con.execute(f"SELECT eff_narrative, eff_severity, COUNT(*) FROM "
                               f"({_SELECT} WHERE {_NOT_DISMISSED}) GROUP BY 1, 2").fetchall()
        narr: dict[str, int] = {}
        sev: dict[str, int] = {}
        for n, s, c in rows:
            narr[n] = narr.get(n, 0) + c
            sev[s] = sev.get(s, 0) + c
        return {"by_narrative": dict(sorted(narr.items(), key=lambda kv: -kv[1])),
                "by_severity": sev}

    def trend(self, since: str) -> list[dict]:
        """Per UTC day: classified, High, alerts."""
        alert_where, params = self._view_sql("alerts")
        with self.store._conn() as con:
            rows = con.execute(
                f"SELECT substr(a.collected_at, 1, 10) AS day, COUNT(*) AS classified, "
                f"SUM(COALESCE(r.severity, p.severity) = 'High') AS high, "
                f"SUM(CASE WHEN {alert_where} THEN 1 ELSE 0 END) AS alerts "
                f"FROM predictions p JOIN articles a ON a.id = p.article_id "
                f"LEFT JOIN latest_review r ON r.article_id = a.id "
                f"WHERE {_NOT_DISMISSED} AND a.collected_at >= ? GROUP BY day ORDER BY day",
                [*params, since]).fetchall()
        return [dict(r) for r in rows]

    def sources(self) -> list[dict]:
        with self.store._conn() as con:
            rows = con.execute(
                "SELECT a.source, a.source_kind, COUNT(*) AS stored, SUM(a.relevant) AS relevant, "
                "COUNT(p.article_id) AS classified FROM articles a "
                "LEFT JOIN predictions p ON p.article_id = a.id "
                "GROUP BY a.source, a.source_kind ORDER BY stored DESC").fetchall()
        return [dict(r) for r in rows]

    def recent_cycles(self, n: int = 10) -> list[dict]:
        with self.store._conn() as con:
            rows = con.execute("SELECT report FROM ingest_runs ORDER BY id DESC LIMIT ?",
                               (n,)).fetchall()
        return [json.loads(r[0]) for r in rows]
