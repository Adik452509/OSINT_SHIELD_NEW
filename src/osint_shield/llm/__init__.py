"""Local LLM access (Ollama) and prompts."""

from .client import OllamaClient, OllamaError, OllamaUnavailable
from .explain import EXPLAIN_PROMPT_VERSION, Explanation, check_answer, explain, explain_prompt
from .prompts import (
    PROMPT_VERSION,
    PROPAGANDA_SCHEMA,
    PROPAGANDA_SYSTEM,
    PROPAGANDA_TYPES,
    evidence_found,
    propaganda_prompt,
    screen_flag,
)

__all__ = [
    "EXPLAIN_PROMPT_VERSION",
    "Explanation",
    "check_answer",
    "explain",
    "explain_prompt",
    "screen_flag",
    "OllamaClient",
    "OllamaError",
    "OllamaUnavailable",
    "PROMPT_VERSION",
    "PROPAGANDA_SCHEMA",
    "PROPAGANDA_SYSTEM",
    "PROPAGANDA_TYPES",
    "evidence_found",
    "propaganda_prompt",
]
