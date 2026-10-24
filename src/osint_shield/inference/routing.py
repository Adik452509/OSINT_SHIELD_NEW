"""From raw articles to the plan's output records (§6.3), with the M7b/M8 rules.

    clf = ArticleClassifier.from_config(load_config("serve.yaml"))
    clf.classify([{"headline": "...", "body": "..."}])

Routing, per article:

* **narrative** - the encoder; except for **headline-only** articles, where the
  keyword rubric decides whenever one of its narrative terms appears (D17,
  P = 0.98 over the encoder on 415 development headlines).
* **severity** - the encoder.
* **propaganda** - ``not_reliably_assessed`` (D20).
* **escalated** (send to the LLM) - severity High, or, for articles WITH a body,
  narrative confidence below ``escalate_below``. Headline-only confidence is
  never used: the encoder is over-confident there (D17).
* **needs_review** (analyst queue, plan §6.4) - every headline-only article and
  every low-confidence one. The review queue is how the rare classes grow.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..keywords import KeywordMatcher

PROPAGANDA_STATUS = "not_reliably_assessed"


@dataclass
class RoutingConfig:
    """How predictions are routed. Read from ``configs/serve.yaml`` → ``routing``."""

    escalate_below: float = 0.70
    headline_only_needs_review: bool = True


def make_id(headline: str, body: str) -> str:
    """Stable id for an article that arrived without one."""
    return "rt_" + hashlib.sha1(f"{headline}\n{body}".encode("utf-8")).hexdigest()[:10]


@dataclass
class ArticleClassifier:
    """Turns articles into output records.

    Args:
        predictor: anything with ``predict_proba(df)`` and ``labels`` - normally
            :class:`osint_shield.inference.Predictor`.
        model_version: stamped on every record.
        routing: escalation and review settings.
    """

    predictor: object
    model_version: str
    routing: RoutingConfig = field(default_factory=RoutingConfig)

    def __post_init__(self):
        self._matcher = KeywordMatcher()

    @classmethod
    def from_config(cls, serve_cfg: dict, *, device=None) -> "ArticleClassifier":
        """Load the deployed model named in ``serve.yaml``."""
        from ..paths import resolve
        from ..runtime import get_device
        from .predict import Predictor
        from .versioning import model_version

        model_dir = resolve(serve_cfg["model_dir"])
        if device is None:
            choice = serve_cfg.get("device", "auto")
            device = get_device(None if choice == "auto" else choice)
        predictor = Predictor.from_dir(model_dir, device=device,
                                       batch_size=int(serve_cfg.get("batch_size", 16)),
                                       precision=serve_cfg.get("precision", "fp32"))
        return cls(predictor, model_version(model_dir, predictor.meta),
                   RoutingConfig(**serve_cfg.get("routing", {})))

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _missing(value) -> bool:
        return value is None or (isinstance(value, float) and math.isnan(value))

    @classmethod
    def _normalise(cls, articles) -> list[dict]:
        """Clean the input into plain Python values.

        Routing decisions are made on these, never on a DataFrame: pandas 3
        turns ``None`` into ``NaN`` on the way into a frame, so ``is None``
        checks after a round trip silently fail - every headline-only article
        then looks like it has a body.
        """
        items = []
        for a in (articles.to_dict("records") if isinstance(articles, pd.DataFrame)
                  else articles):
            headline = "" if cls._missing(a.get("headline")) else str(a["headline"]).strip()
            if not headline:
                raise ValueError("every article needs a non-empty headline")
            body = a.get("body")
            body = body.strip() if isinstance(body, str) else ""
            given_id = a.get("id")
            published = a.get("published_at")
            items.append({
                "id": make_id(headline, body) if cls._missing(given_id) or not given_id
                else str(given_id),
                "headline": headline,
                "body": body,
                "published_at": None if cls._missing(published) else str(published),
            })
        return items

    def _rubric_narrative(self, headline: str) -> str | None:
        """The rubric's narrative label, or None when none of its terms fires.

        Ties go to the first class in rubric order - the same rule as
        ``KeywordRuleBaseline``, which M7b measured.
        """
        counts = {g[len("narr_"):]: n for g, n in self._matcher.counts(headline).items()
                  if g.startswith("narr_")}
        if not counts or max(counts.values()) == 0:
            return None
        return max(counts, key=lambda k: counts[k])

    # ------------------------------------------------------------------- main
    def classify(self, articles) -> list[dict]:
        """One output record per article, in input order."""
        items = self._normalise(articles)
        if not items:
            return []
        # the DataFrame exists only to feed the model
        probs = self.predictor.predict_proba(pd.DataFrame({
            "clean_headline": [a["headline"] for a in items],
            "clean_text": [a["body"] or None for a in items],
        }))
        names = self.predictor.labels
        t = float(self.routing.escalate_below)

        records = []
        for i, item in enumerate(items):
            has_body = bool(item["body"])
            p_n, p_s = np.asarray(probs["narrative"][i]), np.asarray(probs["severity"][i])
            narrative = {"label": names["narrative"][int(p_n.argmax())],
                         "confidence": round(float(p_n.max()), 4), "source": "encoder"}
            severity = {"label": names["severity"][int(p_s.argmax())],
                        "confidence": round(float(p_s.max()), 4)}

            if not has_body:
                rubric = self._rubric_narrative(item["headline"])
                if rubric is not None:
                    # the rubric has no probability; don't invent one
                    narrative = {"label": rubric, "confidence": None, "source": "rubric"}

            escalation, review = [], []
            if severity["label"] == "High":
                escalation.append("severity_high")
            if has_body and narrative["confidence"] < t:
                escalation.append("low_confidence")
                review.append("low_confidence")
            if not has_body and self.routing.headline_only_needs_review:
                review.append("headline_only")

            records.append({
                "id": item["id"],
                "headline": item["headline"],
                "published_at": item["published_at"],
                "has_body": has_body,
                "narrative": narrative,
                "severity": severity,
                "propaganda": {"status": PROPAGANDA_STATUS},
                "escalated": bool(escalation),
                "escalation_reasons": escalation,
                "summary": None,      # filled by the LLM stage (M10)
                "evidence": None,     # filled by the LLM stage (M10)
                "model_version": self.model_version,
                "needs_review": bool(review),
                "review_reasons": review,
            })
        return records
