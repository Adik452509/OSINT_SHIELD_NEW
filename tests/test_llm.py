"""Ollama client and propaganda prompt - no server needed."""

from __future__ import annotations

import json

import pytest

from osint_shield.llm import (
    PROPAGANDA_SCHEMA,
    OllamaClient,
    OllamaError,
    evidence_found,
    propaganda_prompt,
)


# ----------------------------------------------------------------------- prompt
def test_prompt_states_the_own_voice_rule():
    p = propaganda_prompt("Army chief visits post", "Body text.")
    assert "OWN voice" in p
    assert "attribution" in p


def test_prompt_carries_the_article():
    p = propaganda_prompt("Six brave sons honoured", "The ceremony was held.")
    assert "Six brave sons honoured" in p
    assert "The ceremony was held." in p


def test_prompt_truncates_long_bodies_on_a_word_boundary():
    body = "word " * 2000
    p = propaganda_prompt("H", body, max_chars=100)
    article = p.split("ARTICLE:")[1]
    assert len(article) < 130
    assert article.rstrip().endswith("...")


def test_prompt_marks_headline_only_articles():
    assert "(headline only)" in propaganda_prompt("H", None)


def test_schema_constrains_types_to_the_four_techniques():
    enum = PROPAGANDA_SCHEMA["properties"]["types"]["items"]["enum"]
    assert set(enum) == {"glorification", "justification", "victimhood", "recruitment"}


def test_schema_puts_the_analysis_before_the_verdict():
    """Ollama emits fields in schema order: the model must reason before it decides."""
    keys = list(PROPAGANDA_SCHEMA["properties"])
    assert keys.index("analysis") < keys.index("propaganda")
    assert keys.index("own_voice") < keys.index("propaganda")
    assert set(PROPAGANDA_SCHEMA["required"]) == set(keys)


def test_screener_is_recall_oriented():
    """v1 told the model 'when unsure, answer no' and it answered no to everything.
    A screener must flag borderline cases; the human reviewer is the precision filter."""
    from osint_shield.llm import PROMPT_VERSION, PROPAGANDA_SYSTEM

    assert "when unsure, answer no" not in PROPAGANDA_SYSTEM.lower()
    assert "borderline" in PROPAGANDA_SYSTEM.lower()
    assert PROMPT_VERSION != "v1"


# --------------------------------------------------------------------- evidence
def test_evidence_found_ignores_case_quotes_and_spacing():
    assert evidence_found("brave  SONS", "Six ‘brave sons’ were honoured")


def test_evidence_not_found_when_invented():
    assert not evidence_found("glorious victory", "Six soldiers were honoured")


def test_empty_evidence_never_counts():
    assert not evidence_found("", "anything")
    assert not evidence_found("  ", "anything")


def test_evidence_checked_against_headline_and_body():
    assert evidence_found("shield in crisis", "Body text", "India became a shield in crisis")


# ----------------------------------------------------------------------- client
class FakeClient(OllamaClient):
    """Replays canned responses and records every payload sent."""

    def __init__(self, responses):
        super().__init__(host="http://test", model="llama3.2:3b")
        self.responses = list(responses)
        self.sent = []

    def _request(self, path, payload=None):
        self.sent.append((path, payload))
        return self.responses.pop(0)


def test_generate_json_parses_the_response():
    c = FakeClient([{"response": json.dumps({"propaganda": True, "types": [], "strength": 2,
                                             "evidence": "x"})}])
    assert c.generate_json("p", PROPAGANDA_SCHEMA)["strength"] == 2


def test_first_attempt_is_deterministic_and_sends_the_schema():
    c = FakeClient([{"response": "{}"}])
    c.generate_json("p", PROPAGANDA_SCHEMA, system="sys")
    _, payload = c.sent[0]
    assert payload["options"]["temperature"] == 0.0
    assert payload["format"] == PROPAGANDA_SCHEMA
    assert payload["system"] == "sys"
    assert payload["stream"] is False


def test_retry_changes_the_seed_after_invalid_json():
    """An identical deterministic retry would only repeat the failure."""
    c = FakeClient([{"response": "not json"}, {"response": "{\"propaganda\": false}"}])
    assert c.generate_json("p", PROPAGANDA_SCHEMA) == {"propaganda": False}
    seeds = [payload["options"]["seed"] for _, payload in c.sent]
    assert seeds == [0, 1]


def test_raises_after_exhausting_retries():
    c = FakeClient([{"response": "nope"}] * 3)
    with pytest.raises(OllamaError, match="no valid JSON"):
        c.generate_json("p", PROPAGANDA_SCHEMA, retries=2)


def test_models_lists_tags():
    c = FakeClient([{"models": [{"name": "llama3.2:3b"}, {"name": "qwen2.5-coder:7b"}]}])
    assert c.models() == ["llama3.2:3b", "qwen2.5-coder:7b"]


def test_unreachable_server_raises_a_clear_error():
    c = OllamaClient(host="http://127.0.0.1:9", timeout=1)
    with pytest.raises(OllamaError, match="cannot reach"):
        c.models()
