"""Request and response models for the classification API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ArticleIn(BaseModel):
    """One article. Only the headline is required: Google-News items arrive with no body."""

    id: str | None = Field(default=None, description="your id; generated if absent")
    headline: str = Field(min_length=1)
    body: str | None = Field(default=None, description="article text, if available")
    published_at: str | None = None


class ClassifyRequest(BaseModel):
    articles: list[ArticleIn] = Field(min_length=1)


class Label(BaseModel):
    label: str
    confidence: float | None = Field(
        default=None, description="null when the keyword rubric decided (it has no probability)")
    source: str | None = Field(default=None, description="'encoder' or 'rubric'")


class ArticleOut(BaseModel):
    """The plan's output contract (§6.3) plus the routing reasons."""

    id: str
    headline: str
    published_at: str | None
    has_body: bool
    narrative: Label
    severity: Label
    propaganda: dict
    escalated: bool
    escalation_reasons: list[str]
    summary: str | None
    evidence: str | None
    model_version: str
    needs_review: bool
    review_reasons: list[str]


class ClassifyResponse(BaseModel):
    model_version: str
    elapsed_ms: float
    results: list[ArticleOut]


class Health(BaseModel):
    status: str
    model_version: str
    device: str
    escalate_below: float
