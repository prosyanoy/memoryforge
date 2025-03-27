"""The 'what to forget' rule set.

Rules, in the order they are applied:
  1. pii_redaction  - high-risk PII (card numbers, SSNs, secrets, IBANs) is redacted from the text
                      *before* the extractor sees it, so it never reaches an LLM or the store.
  2. pii_fact       - any extracted fact that still mentions such data, or whose key names a
                      sensitive concept (ssn, password, card, ...), is dropped.
  3. invalid_contact- contact details that do not parse (bad email/phone) are dropped.
  4. low_confidence - facts below the confidence threshold are dropped.
  5. superseded     - a newer fact for the same (kind, key) replaces older ones; older observations
                      are kept as inactive history, and a late-arriving older observation is dropped.
  6. expired        - retention sweep removes resolved issues and any fact past its TTL.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .identity import normalize_email, normalize_phone
from .schemas import ExtractedFact, FactKind, MemoryRecord

RULESET_VERSION = "2026.1"

_CARD_RE = re.compile(r"(?<![\w-])(?:\d[ -]?){12,18}\d(?![\w-])")
_SSN_RE = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_SECRET_RE = re.compile(r"(?i)\b(password|passcode|pin|cvv|cvc|secret)\b\s*(?:is|are|:|=)\s*\S+?(?=[.,;!?]*(?:\s|$))")
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}(?:\s?[A-Z0-9]{1,4})?\b")
_MARKER_RE = re.compile(r"\[REDACTED:[a-z]+\]")
_SENSITIVE_KEY_RE = re.compile(
    r"(ssn|social_security|card|credit|cvv|cvc|password|passcode|pin$|iban|passport|licen[sc]e|"
    r"birth|dob|medical|diagnos|health|salary)"
)


def luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
    return total % 10 == 0


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Replace high-risk PII with [REDACTED:<category>] markers. Returns (text, categories)."""
    found: list[str] = []

    def card(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and luhn_ok(digits):
            found.append("card")
            return "[REDACTED:card]"
        return m.group(0)

    def sub(pattern: re.Pattern, label: str, s: str) -> str:
        def repl(m: re.Match) -> str:
            found.append(label)
            if label == "secret":
                return f"{m.group(1)} is [REDACTED:secret]"
            return f"[REDACTED:{label}]"

        return pattern.sub(repl, s)

    text = _CARD_RE.sub(card, text)
    text = sub(_SSN_RE, "ssn", text)
    text = sub(_SECRET_RE, "secret", text)
    text = sub(_IBAN_RE, "iban", text)
    return text, sorted(set(found))


@dataclass
class ForgetDecision:
    fact: ExtractedFact | None  # None when dropped
    reason: str | None = None


@dataclass
class ForgetPolicy:
    min_confidence: float = 0.5
    default_country_code: str = "1"
    retention_days: int = 365
    resolved_issue_ttl_days: int = 30
    version: str = field(default=RULESET_VERSION, init=False)

    def redact(self, text: str) -> tuple[str, list[str]]:
        return redact_pii(text)

    def evaluate(self, fact: ExtractedFact) -> ForgetDecision:
        """Stateless rules (2-4). Returns the fact (possibly normalised) or a drop reason."""
        # Redaction markers in `evidence` are harmless (the model only ever saw redacted text); a marker
        # in the key/value means the fact *is* the sensitive datum. Raw PII anywhere is always fatal.
        blob = f"{fact.key} {fact.value} {fact.evidence or ''}"
        if (_MARKER_RE.search(f"{fact.key} {fact.value}") or _SENSITIVE_KEY_RE.search(fact.key)
                or redact_pii(blob)[1]):
            return ForgetDecision(None, "pii")
        if fact.kind is FactKind.contact_detail:
            try:
                if fact.key == "email":
                    fact = fact.model_copy(update={"value": normalize_email(fact.value)})
                elif fact.key == "phone":
                    fact = fact.model_copy(update={"value": normalize_phone(fact.value, self.default_country_code)})
            except ValueError:
                return ForgetDecision(None, "invalid_contact")
        if fact.confidence < self.min_confidence:
            return ForgetDecision(None, "low_confidence")
        return ForgetDecision(fact)

    def resolve_supersession(
        self, new_observed_at: datetime, existing_active: list[MemoryRecord]
    ) -> tuple[bool, list[int]]:
        """Rule 5. Returns (new_is_stale, ids_to_supersede).

        A fact is stale if an active fact for the same slot was observed *after* it
        (late or replayed out-of-order data must not overwrite newer knowledge).
        """
        if any(m.observed_at > new_observed_at for m in existing_active):
            return True, []
        return False, [m.id for m in existing_active]

    def expiry_cutoffs(self, now: datetime) -> tuple[datetime, datetime]:
        """(resolved-issue cutoff, general cutoff) for the retention sweep (rule 6)."""
        return now - timedelta(days=self.resolved_issue_ttl_days), now - timedelta(days=self.retention_days)
