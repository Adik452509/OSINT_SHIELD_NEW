"""Prompts and output schemas for the LLM.

The propaganda prompt condenses ``docs/PROPAGANDA_GUIDE.md``, so Llama and the
human reviewer apply the same definitions - in particular the own-voice rule
(D18): reported, attributed rhetoric is not propaganda.
"""

from __future__ import annotations

import re

PROPAGANDA_TYPES = ("glorification", "justification", "victimhood", "recruitment")

PROPAGANDA_SCHEMA = {
    "type": "object",
    "properties": {
        "propaganda": {"type": "boolean"},
        "types": {"type": "array", "items": {"type": "string", "enum": list(PROPAGANDA_TYPES)}},
        "strength": {"type": "integer", "minimum": 0, "maximum": 3},
        "evidence": {"type": "string"},
    },
    "required": ["propaganda", "types", "strength", "evidence"],
}

PROPAGANDA_SYSTEM = (
    "You are a careful media analyst who labels news articles for propaganda techniques. "
    "You judge ONLY the article's own voice: its headline, its narration and the framing "
    "the outlet chose. Loaded words inside an attributed quote or a 'X said' statement do "
    "not count unless the article itself adopts them. When unsure, answer no."
)

_PROMPT = """Label this news article for propaganda in its OWN voice.

Techniques:
- glorification: the outlet's own words exalt the military, national sacrifice or the country's
  role beyond neutral reporting ("brave sons", "martyred heroes", "glorious victory").
- justification: the outlet's own words use loaded or one-sided framing to justify a position or
  delegitimise an opponent (partisan slurs as fact, conspiracy or sabotage insinuations).
- victimhood: the outlet's own words frame a person or group as persecuted to serve a partisan
  narrative ("wrongly imprisoned", "silenced") rather than reporting a grievance.
- recruitment: the outlet's own words call the reader to act or join ("rise up", "join the cause").

NOT propaganda: neutral reporting of events or casualties; reporting what a politician, official
or group said, with attribution, even if their words are loaded; analysis that argues with
evidence in neutral language.

Answer with:
- propaganda: true or false
- types: the techniques used (empty if false)
- strength: 0 none, 1 faint or borderline, 2 clear, 3 strong and central to the article
- evidence: the exact words from the article's own voice that show it (empty if false)

HEADLINE: {headline}
ARTICLE: {body}"""


def propaganda_prompt(headline: str, body: str | None, *, max_chars: int = 2000) -> str:
    """Build the screening prompt. The body is cut at ``max_chars`` (on a word
    boundary): propaganda is usually signalled in the headline and opening."""
    body = (body or "").strip()
    if len(body) > max_chars:
        body = body[:max_chars].rsplit(" ", 1)[0] + " ..."
    return _PROMPT.format(headline=str(headline).strip(), body=body or "(headline only)")


def _normalise(text: str) -> str:
    text = re.sub(r"[‘’“”\"']", "", str(text).lower())
    return re.sub(r"\s+", " ", text).strip()


def evidence_found(evidence: str, *texts: str) -> bool:
    """Is the quoted evidence actually present in the article?

    Catches hallucinated quotes. Case, whitespace and quotation marks are
    ignored; an empty quote never counts as found.
    """
    needle = _normalise(evidence)
    if len(needle) < 3:
        return False
    return any(needle in _normalise(t) for t in texts if t)
