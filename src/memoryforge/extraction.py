"""Extraction stage: render the versioned prompt, call the LLM, validate against the Pydantic schema."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import structlog
from pydantic import ValidationError

from . import prompts
from .forgetting import ForgetPolicy
from .llm import LLMClient
from .schemas import ConversationEvent, ExtractedFact, ExtractionResult

log = structlog.get_logger()

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)


class ExtractionError(Exception):
    """The model never produced schema-valid JSON."""

    def __init__(self, message: str, attempts: int, raw: str):
        super().__init__(message)
        self.attempts = attempts
        self.raw = raw


@dataclass
class ExtractionOutcome:
    facts: list[ExtractedFact]
    prompt_version: str
    extractor: str
    attempts: int = 1
    redactions: list[str] = field(default_factory=list)


def parse_llm_json(raw: str) -> ExtractionResult:
    """Tolerate markdown fences and surrounding prose; insist on schema-valid JSON."""
    text = _FENCE_RE.sub("", raw.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in model output")
    return ExtractionResult.model_validate(json.loads(text[start : end + 1]))


class Extractor:
    def __init__(self, llm: LLMClient, policy: ForgetPolicy, prompt_version: str = prompts.DEFAULT_VERSION,
                 attempts: int = 2):
        prompts.render(prompt_version, message="x", event_time="")  # fail fast on a bad version
        self.llm = llm
        self.policy = policy
        self.prompt_version = prompt_version
        self.attempts = max(1, attempts)

    def extract(self, event: ConversationEvent) -> ExtractionOutcome:
        safe_text, redactions = self.policy.redact(event.text)
        if redactions:
            log.info("pii_redacted", categories=redactions)
        rendered = prompts.render(self.prompt_version, message=safe_text, event_time=event.timestamp.isoformat())
        user = rendered.user
        raw, last_err = "", None
        for attempt in range(1, self.attempts + 1):
            raw = self.llm.generate(system=rendered.system, prompt=user, json_schema=rendered.json_schema)
            try:
                result = parse_llm_json(raw)
                break
            except (ValueError, ValidationError) as exc:  # JSONDecodeError is a ValueError
                last_err = exc
                log.warning("extraction_invalid_output", attempt=attempt, error=str(exc)[:300])
                user = rendered.user + (
                    f"\n\nYour previous reply was invalid ({str(exc)[:200]}). Reply with ONLY valid JSON."
                )
        else:
            raise ExtractionError(f"model output failed schema validation: {last_err}", self.attempts, raw[:1000])

        facts = [self._normalise(f, event) for f in result.facts]
        return ExtractionOutcome(facts, self.prompt_version, self.llm.name, attempt, redactions)

    @staticmethod
    def _normalise(fact: ExtractedFact, event: ConversationEvent) -> ExtractedFact:
        """Models invent timestamps: default to the event time and never allow the future."""
        if fact.observed_at is None or fact.observed_at > event.timestamp:
            return fact.model_copy(update={"observed_at": event.timestamp})
        return fact
