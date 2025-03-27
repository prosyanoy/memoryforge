from datetime import timedelta

from conftest import T0
from memoryforge.forgetting import ForgetPolicy, luhn_ok, redact_pii
from memoryforge.schemas import ExtractedFact, FactKind, MemoryRecord


def fact(**kw):
    base = dict(kind="preference", key="k", value="v", confidence=0.9)
    return ExtractedFact(**{**base, **kw})


def rec(id, observed_at):
    return MemoryRecord(id=id, customer_id="c", kind=FactKind.preference, key="k", value="v", confidence=0.9,
                        observed_at=observed_at, source_event_id="e", conversation_id="c", prompt_version="v2",
                        extractor="x", created_at=T0)


def test_luhn():
    assert luhn_ok("4242424242424242") and not luhn_ok("4242424242424241")


def test_redaction_categories():
    text, cats = redact_pii("card 4242-4242-4242-4242 ssn 123-45-6789 PIN: 9876 iban GB82 WEST 1234 5698 7654 32")
    assert cats == ["card", "iban", "secret", "ssn"]
    assert "4242" not in text and "9876" not in text and "123-45" not in text and "WEST" not in text


def test_redaction_leaves_phone_numbers_and_forgot_password_alone():
    text = "call me on 415 555 0123 or +1 (415) 555-0123, I forgot my password"
    assert redact_pii(text) == (text, [])


def test_pii_facts_dropped():
    p = ForgetPolicy()
    assert p.evaluate(fact(key="credit_card", value="visa")).reason == "pii"
    assert p.evaluate(fact(value="it was [REDACTED:card]")).reason == "pii"
    assert p.evaluate(fact(evidence="ssn 123-45-6789")).reason == "pii"
    assert p.evaluate(fact(key="date_of_birth", value="1990")).reason == "pii"


def test_low_confidence_dropped_boundary_kept():
    p = ForgetPolicy(min_confidence=0.5)
    assert p.evaluate(fact(confidence=0.49)).reason == "low_confidence"
    assert p.evaluate(fact(confidence=0.5)).fact is not None


def test_contact_details_validated_and_normalised():
    p = ForgetPolicy()
    d = p.evaluate(fact(kind="contact_detail", key="email", value="Ann@Example.COM"))
    assert d.fact.value == "ann@example.com"
    d = p.evaluate(fact(kind="contact_detail", key="phone", value="(415) 555-0100"))
    assert d.fact.value == "+14155550100"
    assert p.evaluate(fact(kind="contact_detail", key="email", value="nope")).reason == "invalid_contact"


def test_supersession_rules():
    p = ForgetPolicy()
    old = [rec(1, T0), rec(2, T0 - timedelta(days=1))]
    assert p.resolve_supersession(T0 + timedelta(seconds=1), old) == (False, [1, 2])
    assert p.resolve_supersession(T0, old) == (False, [1, 2])  # equal timestamps: newest write wins
    assert p.resolve_supersession(T0 - timedelta(seconds=1), old) == (True, [])  # late data is stale
    assert p.resolve_supersession(T0, []) == (False, [])


def test_expiry_cutoffs():
    resolved, general = ForgetPolicy(retention_days=100, resolved_issue_ttl_days=10).expiry_cutoffs(T0)
    assert T0 - resolved == timedelta(days=10) and T0 - general == timedelta(days=100)


def test_marker_in_evidence_is_fine_but_in_value_is_not():
    p = ForgetPolicy()
    assert p.evaluate(fact(evidence="text me, card is [REDACTED:card]")).fact is not None
    assert p.evaluate(fact(value="[REDACTED:card]")).reason == "pii"
