"""Local LLM access (Ollama) and prompts."""

from .client import OllamaClient, OllamaError
from .prompts import (
    PROPAGANDA_SCHEMA,
    PROPAGANDA_SYSTEM,
    PROPAGANDA_TYPES,
    evidence_found,
    propaganda_prompt,
)

__all__ = [
    "OllamaClient",
    "OllamaError",
    "PROPAGANDA_SCHEMA",
    "PROPAGANDA_SYSTEM",
    "PROPAGANDA_TYPES",
    "evidence_found",
    "propaganda_prompt",
]
