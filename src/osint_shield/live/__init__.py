"""The live pipeline: ingest -> fetch bodies -> classify -> explain, every cycle."""

from .pipeline import LivePipeline

__all__ = ["LivePipeline"]
