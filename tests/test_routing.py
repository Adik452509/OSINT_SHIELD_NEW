"""ArticleClassifier routing rules (D17, D20, plan §6.2-6.4) - fake predictor, no model."""

from __future__ import annotations

import numpy as np
import pytest

from osint_shield.inference import PROPAGANDA_STATUS, ArticleClassifier, RoutingConfig, make_id

NARR = {0: "Security", 1: "Political", 2: "Civilian", 3: "Other", 4: "Investigation"}


class FakePredictor:
    """Returns the probabilities you give it, one row per article."""

    labels = {"narrative": NARR, "severity": {0: "Low", 1: "High"},
              "propaganda": {0: "No", 1: "Yes"}}
    device = "cpu"

    def __init__(self, narrative, severity):
        self.narrative = np.asarray(narrative, dtype=float)
        self.severity = np.asarray(severity, dtype=float)
        self.calls = 0

    def predict_proba(self, df):
        self.calls += 1
        n = len(df)
        return {"narrative": np.tile(self.narrative, (n, 1)),
                "severity": np.tile(self.severity, (n, 1)),
                "propaganda": np.tile([0.9, 0.1], (n, 1))}


def clf(narrative=(0.9, 0.05, 0.02, 0.02, 0.01), severity=(0.8, 0.2), t=0.70):
    return ArticleClassifier(FakePredictor(narrative, severity), model_version="test-v1",
                             routing=RoutingConfig(escalate_below=t))


BODIED = {"headline": "Army holds annual exercise", "body": "Troops trained for a week."}


def test_confident_low_severity_article_is_neither_escalated_nor_reviewed():
    r = clf().classify([BODIED])[0]
    assert r["narrative"] == {"label": "Security", "confidence": 0.9, "source": "encoder"}
    assert r["severity"]["label"] == "Low"
    assert r["escalated"] is False and r["needs_review"] is False


def test_high_severity_escalates():
    r = clf(severity=(0.3, 0.7)).classify([BODIED])[0]
    assert r["escalated"] is True
    assert r["escalation_reasons"] == ["severity_high"]


def test_low_confidence_escalates_and_queues_for_review():
    r = clf(narrative=(0.5, 0.3, 0.1, 0.05, 0.05)).classify([BODIED])[0]
    assert "low_confidence" in r["escalation_reasons"]
    assert "low_confidence" in r["review_reasons"]


def test_threshold_is_respected():
    narrative = (0.75, 0.15, 0.05, 0.03, 0.02)
    assert clf(narrative=narrative, t=0.70).classify([BODIED])[0]["escalated"] is False
    assert clf(narrative=narrative, t=0.80).classify([BODIED])[0]["escalated"] is True


def test_headline_only_uses_the_rubric_when_a_keyword_fires():
    """D17: hybrid narrative for headline-only articles."""
    r = clf(narrative=(0.1, 0.8, 0.05, 0.03, 0.02)).classify(
        [{"headline": "Floods displace thousands of civilians"}])[0]
    assert r["narrative"] == {"label": "Civilian", "confidence": None, "source": "rubric"}


def test_headline_only_falls_back_to_the_encoder_without_a_keyword():
    r = clf().classify([{"headline": "Weather update for the weekend"}])[0]
    assert r["narrative"]["source"] == "encoder"
    assert r["narrative"]["label"] == "Security"


def test_headline_only_is_always_reviewed_but_never_escalated_on_confidence():
    """D17: the encoder is over-confident on headlines - type routes it, not confidence."""
    r = clf(narrative=(0.4, 0.3, 0.1, 0.1, 0.1)).classify(
        [{"headline": "Weather update for the weekend"}])[0]
    assert r["needs_review"] is True and r["review_reasons"] == ["headline_only"]
    assert "low_confidence" not in r["escalation_reasons"]
    assert r["escalated"] is False


def test_headline_only_still_escalates_on_high_severity():
    r = clf(severity=(0.2, 0.8)).classify([{"headline": "Weather update"}])[0]
    assert r["escalation_reasons"] == ["severity_high"]


def test_propaganda_is_reported_as_not_assessed():
    """D20: the head is trained but not trustworthy - never a yes/no."""
    r = clf().classify([BODIED])[0]
    assert r["propaganda"] == {"status": PROPAGANDA_STATUS}


def test_every_record_carries_the_contract_fields():
    r = clf().classify([BODIED])[0]
    for key in ("id", "headline", "published_at", "narrative", "severity", "escalated",
                "summary", "evidence", "model_version", "needs_review"):
        assert key in r
    assert r["model_version"] == "test-v1"
    assert r["summary"] is None and r["evidence"] is None     # M10 fills these


def test_ids_are_kept_or_generated_deterministically():
    given = clf().classify([{**BODIED, "id": "mine-1"}])[0]
    assert given["id"] == "mine-1"
    a = clf().classify([BODIED])[0]["id"]
    b = clf().classify([BODIED])[0]["id"]
    assert a == b == make_id(BODIED["headline"], BODIED["body"])
    assert a.startswith("rt_")


def test_batch_preserves_input_order():
    arts = [{**BODIED, "id": f"a{i}"} for i in range(5)]
    assert [r["id"] for r in clf().classify(arts)] == ["a0", "a1", "a2", "a3", "a4"]


def test_whitespace_body_counts_as_headline_only():
    r = clf().classify([{"headline": "Weather update", "body": "   "}])[0]
    assert r["has_body"] is False


def test_published_at_passes_through_or_stays_none():
    """pandas 3 turns None into NaN inside a frame - the record must still say None."""
    with_date = clf().classify([{**BODIED, "published_at": "2026-10-24T06:00:00Z"}])[0]
    without = clf().classify([BODIED])[0]
    assert with_date["published_at"] == "2026-10-24T06:00:00Z"
    assert without["published_at"] is None


def test_dataframe_input_with_nan_fields_routes_correctly():
    """The regression behind M9's first test run: NaN bodies must mean headline-only."""
    import pandas as pd

    df = pd.DataFrame({"headline": ["Floods displace thousands of civilians", "Army exercise"],
                       "body": [None, "Troops trained."],
                       "published_at": [None, None]})
    first, second = clf().classify(df)
    assert first["has_body"] is False and first["narrative"]["source"] == "rubric"
    assert second["has_body"] is True
    assert first["published_at"] is None and first["id"].startswith("rt_")


def test_missing_headline_is_rejected():
    with pytest.raises(ValueError, match="headline"):
        clf().classify([{"headline": "  ", "body": "text"}])


def test_empty_input_does_not_call_the_model():
    c = clf()
    assert c.classify([]) == []
    assert c.predictor.calls == 0
