"""Prompts and output schemas for the LLM.

The propaganda prompt condenses ``docs/PROPAGANDA_GUIDE.md``, so Llama and the
human reviewer apply the same definitions - in particular the own-voice rule
(D18): reported, attributed rhetoric is not propaganda.
"""

from __future__ import annotations

import re

PROPAGANDA_TYPES = ("glorification", "justification", "victimhood", "recruitment")

#: Bump whenever the prompt or schema changes. Screening results from another
#: version are ignored on resume, so one run never mixes two prompts.
#:   v1         "when unsure, answer no" - answered false/0/"" on every one of
#:              the first 20 articles in 0.7 s: degenerate, cheapest valid output
#:   v2-recall  recall-oriented, analysis first - flagged ALL 19 calibration
#:              articles at strength 2, own_voice true on 18/19, and copied the
#:              prompt's own example phrases as "evidence": form-filling again
#:   v3-fewshot invented few-shot examples; no example phrases inside the
#:              definitions; `quoted_only` answered before the verdict and the
#:              own-voice rule enforced in code (see `screen_flag`)
PROMPT_VERSION = "v3-fewshot"

# Property order matters: Ollama's grammar emits fields in schema order, so the
# model reasons, then rules on attribution, before it commits to a verdict.
PROPAGANDA_SCHEMA = {
    "type": "object",
    "properties": {
        "analysis": {"type": "string"},
        "quoted_only": {"type": "boolean"},
        "propaganda": {"type": "boolean"},
        "types": {"type": "array", "items": {"type": "string", "enum": list(PROPAGANDA_TYPES)}},
        "strength": {"type": "integer", "minimum": 0, "maximum": 3},
        "evidence": {"type": "string"},
    },
    "required": ["analysis", "quoted_only", "propaganda", "types", "strength", "evidence"],
}

PROPAGANDA_SYSTEM = (
    "You screen news articles for propaganda techniques so that a human reviewer can check "
    "them. Flag anything that might qualify and use strength 1 for borderline cases - the "
    "reviewer makes the final decision. Judge only the article's own voice: its headline, its "
    "narration and the framing the outlet chose. Most news articles are NOT propaganda. "
    "Copy evidence exactly from the article - never from these instructions."
)

# Invented examples, deliberately not from the corpus: a corpus article here
# would leak its own answer into the screen.
_EXAMPLES = """Examples (invented, for illustration):

HEADLINE: Our fearless lions crush the enemy once again
ARTICLE: Troops repelled an incursion overnight. Two attackers were killed.
{{"analysis": "The headline itself calls soldiers 'fearless lions' and celebrates crushing the enemy: glorifying language in the outlet's own voice.", "quoted_only": false, "propaganda": true, "types": ["glorification"], "strength": 2, "evidence": "Our fearless lions crush the enemy"}}

HEADLINE: Opposition is selling out the nation, says minister
ARTICLE: The minister told reporters the opposition was "selling out the nation". The opposition rejected the claim as baseless.
{{"analysis": "The loaded phrase belongs to the minister and is attributed; the article reports it and gives the opposition's reply.", "quoted_only": true, "propaganda": false, "types": [], "strength": 0, "evidence": ""}}

HEADLINE: Army releases names of soldiers killed in border clash
ARTICLE: The Army on Monday released the names of four soldiers killed in last week's clash.
{{"analysis": "Neutral factual reporting of casualties; no loaded language.", "quoted_only": false, "propaganda": false, "types": [], "strength": 0, "evidence": ""}}"""

_PROMPT = """Screen this news article for propaganda in its OWN voice.

Techniques:
- glorification: the outlet's own words exalt the military, national sacrifice or the country's
  role with heroic or triumphal language beyond what neutral reporting needs.
- justification: the outlet's own words use loaded or one-sided framing to justify a position or
  delegitimise an opponent - partisan slurs stated as fact, conspiracy or sabotage insinuation.
- victimhood: the outlet's own words frame a person or group as persecuted to serve a partisan
  narrative, rather than reporting a grievance.
- recruitment: the outlet's own words call the reader to act, mobilise or join.

NOT propaganda: neutral reporting of events or casualties; reporting what a politician, official
or group said, with attribution, even when their words are loaded; analysis arguing with evidence
in neutral language.

Fields, in order:
- analysis: one or two sentences - what loaded language is there, and whose words are they?
- quoted_only: true if ALL the loaded language is inside quotes or attributed statements
- propaganda: true if the article's own voice might use a technique - borderline counts
- types: the techniques used (empty if false)
- strength: 0 none, 1 borderline, 2 clear, 3 strong and central to the article
- evidence: exact words copied from the article's own voice (empty if false)

""" + _EXAMPLES + """

Now screen this article.

HEADLINE: {headline}
ARTICLE: {body}"""


def screen_flag(answer: dict) -> bool:
    """The screen's decision, with the own-voice rule enforced in code.

    v2 showed the model will describe a quote as a quote and flag it anyway, so
    the rule is not left to the model: an answer whose loaded language is all
    quoted or attributed is never a flag, whatever its verdict field says.
    """
    return bool(answer.get("propaganda")) and not bool(answer.get("quoted_only"))


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
