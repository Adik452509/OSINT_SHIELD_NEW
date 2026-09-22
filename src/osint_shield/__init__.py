"""OSINT Shield - defence & security news classification.

A fine-tuned multilingual encoder classifies every incoming article; only
high-severity or low-confidence articles are escalated to a local LLM for
summary and reasoning.
"""

__version__ = "0.1.0"

from .config import (
    collapse_map,
    load_config,
    load_taxonomy,
    narrative_label_maps,
    severity_label_maps,
)
from .paths import ROOT

__all__ = [
    "__version__",
    "ROOT",
    "load_config",
    "load_taxonomy",
    "narrative_label_maps",
    "severity_label_maps",
    "collapse_map",
]
