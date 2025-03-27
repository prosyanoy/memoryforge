"""Identity resolution: map phone / email / user-id variants onto one customer id."""

from __future__ import annotations

import re
from dataclasses import dataclass

import structlog

from .schemas import Identifiers
from .store import Store

log = structlog.get_logger()

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_EXT_RE = re.compile(r"(?i)\s*(?:ext\.?|x)\s*\d+\s*$")


class IdentityError(Exception):
    """No usable identifier on the event."""


class IdentityConflict(Exception):
    """The event's identifiers point at more than one existing customer."""

    def __init__(self, customers: set[str]):
        self.customers = customers
        super().__init__(f"identifiers map to multiple customers: {sorted(customers)}")


def normalize_email(raw: str) -> str:
    v = raw.strip().lower()
    if v.startswith("mailto:"):
        v = v[7:]
    if not _EMAIL_RE.match(v):
        raise ValueError(f"invalid email {raw!r}")
    return v


def normalize_phone(raw: str, default_country_code: str = "1") -> str:
    """Normalise to an E.164-style string. Raises ValueError when it cannot be trusted."""
    v = _EXT_RE.sub("", raw.strip())
    digits = re.sub(r"\D", "", v)
    if v.startswith("+"):
        pass
    elif digits.startswith("00"):
        digits = digits[2:]
    elif len(digits) == 10:
        digits = default_country_code + digits
    elif len(digits) == 11 and digits.startswith(default_country_code):
        pass
    else:
        raise ValueError(f"ambiguous phone number {raw!r}")
    if not 8 <= len(digits) <= 15:
        raise ValueError(f"invalid phone number {raw!r}")
    return "+" + digits


def normalize_user_id(raw: str) -> str:
    v = re.sub(r"[\s_]+", "-", raw.strip().lower())
    if not v:
        raise ValueError("empty user id")
    return v


def normalize_identifiers(ids: Identifiers, default_country_code: str = "1") -> list[tuple[str, str]]:
    """Return [(kind, normalised value)] for every identifier that parses; skip malformed ones."""
    out: list[tuple[str, str]] = []
    for kind, raw, fn in (
        ("user_id", ids.user_id, normalize_user_id),
        ("email", ids.email, normalize_email),
        ("phone", ids.phone, lambda s: normalize_phone(s, default_country_code)),
    ):
        if not raw:
            continue
        try:
            out.append((kind, fn(raw)))
        except ValueError as exc:
            log.warning("identifier_rejected", kind=kind, reason=str(exc))
    return out


@dataclass
class Resolution:
    customer_id: str
    created: bool
    linked: list[tuple[str, str]]


class IdentityResolver:
    def __init__(self, store: Store, default_country_code: str = "1"):
        self.store = store
        self.cc = default_country_code

    def parse_reference(self, ref: str) -> tuple[str, str] | None:
        """Interpret a lookup string (e.g. from a URL) as an identifier, or None."""
        ref = ref.strip()
        try:
            if "@" in ref:
                return "email", normalize_email(ref)
            if re.fullmatch(r"\+?[\d\s().-]{8,}", ref):
                return "phone", normalize_phone(ref, self.cc)
        except ValueError:
            return None
        return None

    def lookup(self, ref: str) -> str | None:
        """Read-only: customer id (or any identifier variant) -> canonical customer id."""
        direct = self.store.canonical_customer(ref.strip())
        if direct:
            return direct
        pair = self.parse_reference(ref)
        pairs = [pair] if pair else [("user_id", normalize_user_id(ref))] if ref.strip() else []
        found = set(self.store.find_customers(pairs).values())
        return found.pop() if len(found) == 1 else None

    def resolve(self, ids: Identifiers) -> Resolution:
        """Find or create the customer for these identifiers.

        Fails closed: if the identifiers touch two different customers nothing is linked
        and IdentityConflict is raised (use Store.merge_customers to merge deliberately).
        """
        pairs = normalize_identifiers(ids, self.cc)
        if not pairs:
            raise IdentityError("no usable identifiers on event")
        with self.store.transaction():
            matches = self.store.find_customers(pairs)
            customers = set(matches.values())
            if len(customers) > 1:
                raise IdentityConflict(customers)
            created = not customers
            customer_id = self.store.create_customer() if created else next(iter(customers))
            new = [p for p in pairs if p not in matches]
            if new:
                self.store.link_identifiers(customer_id, new)
        if created:
            log.info("customer_created", customer_id=customer_id)
        elif new:
            log.info("identifiers_linked", customer_id=customer_id, linked=[k for k, _ in new])
        return Resolution(customer_id, created, new)
