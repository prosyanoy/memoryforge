import pytest

from memoryforge.identity import (IdentityConflict, IdentityError, normalize_email, normalize_phone,
                                  normalize_user_id)
from memoryforge.schemas import Identifiers


@pytest.mark.parametrize("raw", ["(415) 555-0100", "415.555.0100", "+1 415 555 0100", "1-415-555-0100",
                                 "001 415 555 0100", "+14155550100", "415 555 0100 ext. 22", "415-555-0100 x9"])
def test_phone_variants_normalise_identically(raw):
    assert normalize_phone(raw) == "+14155550100"


@pytest.mark.parametrize("raw", ["555-0100", "abc", "+123", "12345678901234567890"])
def test_bad_phones_rejected(raw):
    with pytest.raises(ValueError):
        normalize_phone(raw)


def test_email_and_user_id_variants():
    assert normalize_email(" mailto:Ann@Example.COM ") == "ann@example.com"
    with pytest.raises(ValueError):
        normalize_email("not-an-email")
    assert normalize_user_id("USR_123") == normalize_user_id(" usr-123 ") == "usr-123"


def test_all_variants_resolve_to_one_customer(svc):
    r = svc.identity
    a = r.resolve(Identifiers(user_id="USR_1", email="Ann@x.com", phone="(415) 555-0100"))
    assert a.created
    for ids in (Identifiers(email="ann@x.com"), Identifiers(phone="+1 415 555 0100"), Identifiers(user_id="usr-1"),
                Identifiers(email="ANN@X.COM", phone="415.555.0100")):
        b = r.resolve(ids)
        assert b.customer_id == a.customer_id and not b.created


def test_new_identifier_is_linked_and_then_resolves(svc):
    r = svc.identity
    a = r.resolve(Identifiers(email="a@x.com"))
    b = r.resolve(Identifiers(email="a@x.com", phone="415-555-0100"))
    assert b.linked == [("phone", "+14155550100")]
    assert r.resolve(Identifiers(phone="4155550100")).customer_id == a.customer_id


def test_conflict_fails_closed_and_links_nothing(svc):
    r = svc.identity
    a = r.resolve(Identifiers(email="a@x.com"))
    b = r.resolve(Identifiers(email="b@x.com", user_id="u-b"))
    with pytest.raises(IdentityConflict) as e:
        r.resolve(Identifiers(email="a@x.com", user_id="u-b", phone="415-555-0100"))
    assert e.value.customers == {a.customer_id, b.customer_id}
    assert svc.store.find_customers([("phone", "+14155550100")]) == {}  # nothing linked by the failed call


def test_conflict_between_two_known_customers(svc):
    r = svc.identity
    r.resolve(Identifiers(email="a@x.com"))
    r.resolve(Identifiers(phone="415-555-0100"))
    with pytest.raises(IdentityConflict):
        r.resolve(Identifiers(email="a@x.com", phone="415-555-0100"))


def test_unusable_identifiers_rejected(svc):
    with pytest.raises(IdentityError):
        svc.identity.resolve(Identifiers(email="garbage", phone="12"))


def test_lookup_by_id_or_variant_and_merge_pointer(svc):
    r = svc.identity
    a = r.resolve(Identifiers(email="a@x.com", phone="415-555-0100")).customer_id
    b = r.resolve(Identifiers(email="b@x.com")).customer_id
    assert r.lookup(a) == a and r.lookup("A@X.com") == a and r.lookup("(415) 555-0100") == a
    assert r.lookup("nobody@x.com") is None and r.lookup("cus_doesnotexist") is None
    svc.store.merge_customers(b, a)
    assert r.lookup(b) == a and r.lookup("b@x.com") == a
