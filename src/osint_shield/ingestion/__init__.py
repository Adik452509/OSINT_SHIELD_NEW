"""Live news ingestion: sources, feed parsing, dedupe keys and the relevance gate."""

from .feeds import FeedItem, HttpFetcher, html_to_text, parse_feed
from .ingest import IngestReport, SourceReport, ingest_source, run_ingest
from .relevance import RelevanceGate
from .sources import Source, google_news_url, load_sources
from .urls import article_id, canonical_url, headline_key, url_hash

__all__ = [
    "FeedItem", "HttpFetcher", "IngestReport", "RelevanceGate", "Source", "SourceReport",
    "article_id", "canonical_url", "google_news_url", "headline_key", "html_to_text",
    "ingest_source", "load_sources", "parse_feed", "run_ingest", "url_hash",
]
