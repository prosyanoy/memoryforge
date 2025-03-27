"""Pydantic schemas shared by every pipeline stage."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def ensure_aware(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


class FactKind(str, Enum):
    preference = "preference"
    issue = "issue"
    contact_detail = "contact_detail"


class IssueStatus(str, Enum):
    open = "open"
    resolved = "resolved"


class Identifiers(BaseModel):
    """Raw (un-normalised) identifiers attached to an event. At least one is required."""

    user_id: str | None = Field(default=None, max_length=128)
    email: str | None = Field(default=None, max_length=254)
    phone: str | None = Field(default=None, max_length=32)

    @model_validator(mode="after")
    def _at_least_one(self) -> "Identifiers":
        if not any((self.user_id, self.email, self.phone)):
            raise ValueError("at least one of user_id, email, phone is required")
        return self


class ConversationEvent(BaseModel):
    """A single raw message from a conversation. `event_id` is the idempotency key."""

    event_id: str = Field(min_length=1, max_length=128)
    conversation_id: str = Field(min_length=1, max_length=128)
    timestamp: datetime
    role: Literal["customer", "agent"] = "customer"
    text: str = Field(min_length=1, max_length=8000)
    identifiers: Identifiers
    channel: str = Field(default="chat", max_length=32)
    correlation_id: str | None = Field(default=None, max_length=128)

    @field_validator("timestamp")
    @classmethod
    def _aware(cls, v: datetime) -> datetime:
        return ensure_aware(v)


class ExtractedFact(BaseModel):
    """One fact as emitted by the extractor (LLM output contract)."""

    kind: FactKind
    key: str
    value: str
    confidence: float = Field(ge=0.0, le=1.0)
    observed_at: datetime | None = None
    status: IssueStatus | None = None
    evidence: str | None = Field(default=None, max_length=280)

    @field_validator("key")
    @classmethod
    def _key(cls, v: str) -> str:
        v = v.strip().lower().replace("-", "_").replace(" ", "_")
        if not KEY_RE.match(v):
            raise ValueError(f"invalid fact key {v!r}")
        return v

    @field_validator("value")
    @classmethod
    def _value(cls, v: str) -> str:
        v = v.strip()
        if not v or len(v) > 500:
            raise ValueError("value must be 1-500 characters")
        return v

    @field_validator("observed_at")
    @classmethod
    def _aware(cls, v: datetime | None) -> datetime | None:
        return ensure_aware(v) if v else None

    @model_validator(mode="after")
    def _status_only_for_issues(self) -> "ExtractedFact":
        if self.kind is not FactKind.issue:
            self.status = None
        elif self.status is None:
            self.status = IssueStatus.open
        return self


class ExtractionResult(BaseModel):
    """Top-level JSON object the extractor must return."""

    facts: list[ExtractedFact] = Field(default_factory=list)


class MemoryRecord(BaseModel):
    """A fact as persisted for one customer."""

    id: int
    customer_id: str
    kind: FactKind
    key: str
    value: str
    confidence: float
    observed_at: datetime
    status: IssueStatus | None = None
    evidence: str | None = None
    source_event_id: str
    conversation_id: str
    prompt_version: str
    extractor: str
    created_at: datetime
    active: bool = True
    superseded_by: int | None = None


class ScoreBreakdown(BaseModel):
    similarity: float
    recency: float
    confidence: float


class ScoredMemory(BaseModel):
    memory: MemoryRecord
    score: float
    breakdown: ScoreBreakdown


class RetrievalResponse(BaseModel):
    customer_id: str
    query: str
    took_ms: float
    results: list[ScoredMemory]


def fact_text(kind: FactKind | str, key: str, value: str, status: IssueStatus | str | None = None) -> str:
    """Canonical text that is embedded for a fact."""
    kind = kind.value if isinstance(kind, FactKind) else kind
    suffix = f" ({status.value if isinstance(status, IssueStatus) else status})" if status else ""
    return f"{kind} {key.replace('_', ' ')}: {value}{suffix}"
