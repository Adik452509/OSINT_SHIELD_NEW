"""The LLM stage: a short summary and verbatim evidence for escalated articles.

Plan §6.3 leaves ``summary`` and ``evidence`` to the LLM. What Llama is asked,
and what it is NOT asked, follows from M8 (D19): local 3B models fill a form
without performing the judgement it encodes. So:

* Llama is **not** told the classifier's labels and is **not** asked to judge
  them - it would only rationalise them. It is asked for facts: 1-3 sentences
  copied from the article that state what happened, and a two-sentence summary.
* Every quote is **checked in code** against the article; a quote that is not
  there is dropped. An explanation with no surviving quote is stored as
  ``unverified`` and shown as such - never as evidence.
* The summary cannot be checked mechanically; the dashboard labels it as an
  LLM summary.

Only escalated articles **with a body** are explained: for a headline-only
article the "summary" would be a paraphrase of the headline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .client import OllamaClient, OllamaError, OllamaUnavailable
from .prompts import evidence_found

#: Bump whenever the prompt or schema changes; stored with every explanation.
EXPLAIN_PROMPT_VERSION = "explain-v1"

#: Quotes shorter than this prove nothing ("the army", "on Monday").
MIN_QUOTE_CHARS = 20

# Evidence before summary: Ollama emits fields in schema order, so the model
# picks the facts from the text before it writes about them.
EXPLAIN_SCHEMA = {
    "type": "object",
    "properties": {
        "evidence": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "summary": {"type": "string"},
    },
    "required": ["evidence", "summary"],
}

EXPLAIN_SYSTEM = (
    "You help a defence and security analyst triage news. Use only facts stated in the "
    "article. Never add background knowledge, opinions or predictions. Copy quotes exactly, "
    "character for character."
)

_PROMPT = """Read this news article.

Fields, in order:
- evidence: 1 to 3 sentences copied EXACTLY from the article that state the key facts:
  what happened, where, who was involved, and any casualties, damage or threat.
- summary: at most two sentences in your own words, using only facts the article states.

HEADLINE: {headline}
ARTICLE: {body}"""


@dataclass
class Explanation:
    status: str                       # ok | unverified | failed
    summary: str | None = None
    evidence: list[str] = field(default_factory=list)
    dropped: int = 0                  # quotes not found in the article
    error: str | None = None


def explain_prompt(headline: str, body: str, *, max_chars: int = 6000) -> str:
    """The prompt, with the body cut on a word boundary. The key facts of a news
    article are in its opening; 6,000 characters is ~1,500 tokens."""
    body = body.strip()
    if len(body) > max_chars:
        body = body[:max_chars].rsplit(" ", 1)[0] + " ..."
    return _PROMPT.format(headline=headline.strip(), body=body)


def check_answer(answer: dict, headline: str, body: str) -> Explanation:
    """Keep only quotes that really are in the article; grade the result."""
    quotes = [q.strip() for q in (answer.get("evidence") or []) if isinstance(q, str)]
    verified = [q for q in quotes
                if len(q) >= MIN_QUOTE_CHARS and evidence_found(q, headline, body)]
    summary = str(answer.get("summary") or "").strip()[:800] or None
    if not verified and not summary:
        return Explanation("failed", error="empty answer", dropped=len(quotes))
    return Explanation("ok" if verified else "unverified", summary, verified,
                       dropped=len(quotes) - len(verified))


def explain(client: OllamaClient, headline: str, body: str, *,
            max_chars: int = 6000) -> Explanation:
    """Ask the LLM, then verify.

    Raises:
        OllamaUnavailable: Ollama is not running - the caller retries later.
    """
    try:
        answer = client.generate_json(explain_prompt(headline, body, max_chars=max_chars),
                                      EXPLAIN_SCHEMA, system=EXPLAIN_SYSTEM)
    except OllamaUnavailable:
        raise
    except OllamaError as exc:
        return Explanation("failed", error=str(exc)[:300])
    return check_answer(answer, headline, body)
