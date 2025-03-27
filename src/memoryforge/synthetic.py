"""Deterministic synthetic conversation generator (demo data, tests, benchmark)."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from .schemas import ConversationEvent, Identifiers

FIRST = ["Ava", "Liam", "Noah", "Mia", "Zoe", "Ethan", "Lena", "Omar", "Priya", "Hugo", "Iris", "Kofi"]
LAST = ["Stone", "Rivera", "Chen", "Okafor", "Novak", "Silva", "Haddad", "Larsen", "Mori", "Dubois"]
THINGS = ["dark mode", "the mobile app", "weekly summaries", "the annual plan", "keyboard shortcuts",
          "the compact layout", "voice notes", "the desktop client", "paper invoices", "short replies"]
CHANNELS = ["email", "phone", "text", "whatsapp"]
TIMES = ["after 5pm", "in the morning", "before 10am", "on weekends", "in the evening"]
LANGS = ["spanish", "french", "german", "portuguese"]
ISSUES = [
    ("login", "I can't log in to my account since the update."),
    ("billing", "I was charged twice for my subscription this month."),
    ("shipping", "My package never arrived and tracking is stuck."),
    ("app_stability", "The app keeps crashing when I open the dashboard."),
]
RESOLUTIONS = {
    "login": "Thanks, the login problem is fixed now.",
    "billing": "The billing issue is resolved, I got the refund.",
    "shipping": "The package finally arrived, the shipping thing is sorted.",
    "app_stability": "The crash is fixed after the latest release.",
}


@dataclass
class Persona:
    index: int
    name: str
    user_id: str
    email: str
    phone: str  # canonical 10-digit local part (US)
    truth: dict[str, str] = field(default_factory=dict)  # key -> expected current value (best effort)


def _phone_variants(p: str) -> list[str]:
    return [f"+1{p}", f"({p[:3]}) {p[3:6]}-{p[6:]}", f"{p[:3]}.{p[3:6]}.{p[6:]}", f"1-{p[:3]}-{p[3:6]}-{p[6:]}"]


def _email_variants(e: str) -> list[str]:
    return [e, e.upper(), f" {e} ", f"mailto:{e}"]


def _uid_variants(u: str) -> list[str]:
    return [u, u.upper(), u.replace("-", "_"), f" {u} "]


def make_persona(i: int, rng: random.Random) -> Persona:
    first, last = rng.choice(FIRST), rng.choice(LAST)
    return Persona(
        index=i, name=f"{first} {last}", user_id=f"usr-{i:06d}",
        email=f"{first.lower()}.{last.lower()}{i}@example.com", phone=f"415555{i % 10000:04d}",
    )


Fact = tuple[str, str, str, str | None]  # (kind, key, value, issue status)


def _slug(s: str) -> str:
    return "_".join("".join(c if c.isalnum() else " " for c in s.lower()).split())


def _messages(p: Persona, rng: random.Random, first: bool) -> list[tuple[str, str, set[Fact]]]:
    """(role, text, expected facts) triples for one conversation. Expected facts are the ground truth
    the extractor + forget rules should produce (post-normalisation)."""
    ch = rng.choice(CHANNELS)
    t1, thing, l1, l2 = rng.choice(TIMES), rng.choice(THINGS), rng.choice(LANGS), rng.choice(LANGS)
    chan_value = {"text": "sms"}.get(ch, ch)
    options: list[tuple[str, set[Fact]]] = [
        (f"I prefer {ch} for anything urgent, please.", {("preference", "contact_channel", chan_value, None)}),
        (f"I'd rather you contact me {t1}.", {("preference", "contact_time", t1, None)}),
        (f"I love {thing}.", {("preference", f"likes_{_slug(thing.removeprefix('the '))}", thing.removeprefix("the "), None)}),
        (f"I speak {l1}, so replies in {l2} would help.", {("preference", "language", l1, None)}),
        ("Please don't call me, text works better.", {("preference", "avoid_channel_phone", "do not contact via phone", None)}),
        ("Maybe I prefer chat support, not sure yet.", set()),  # hedged -> forgotten
        ("My card is 4242 4242 4242 4242 and my password is hunter2, can you update billing?", set()),  # PII
        ("My SSN is 123-45-6789 if you need it to verify.", set()),  # PII
    ]
    topic, issue = rng.choice(ISSUES)
    msgs: list[tuple[str, str, set[Fact]]] = []
    if first:
        msgs.append(("customer",
                     f"Hi, my name is {p.name.split()[0]}. My email is {p.email} and my phone number is {_phone_variants(p.phone)[1]}.",
                     {("contact_detail", "preferred_name", p.name.split()[0], None),
                      ("contact_detail", "email", p.email, None),
                      ("contact_detail", "phone", f"+1{p.phone}", None)}))
        msgs.append(("agent", "Thanks! How can I help today?", set()))
    msgs.append(("customer", issue, {("issue", topic, issue, "open")}))
    msgs.append(("agent", "Sorry about that, looking into it now.", set()))
    text, expected = rng.choice(options)
    msgs.append(("customer", text, expected))
    if rng.random() < 0.4:
        msgs.append(("customer", RESOLUTIONS[topic], {("issue", topic, RESOLUTIONS[topic], "resolved")}))
    return msgs


def generate_events(customers: int = 20, conversations: int = 3, seed: int = 7,
                    start: datetime | None = None, truth: dict[str, set[Fact]] | None = None) -> tuple[list[ConversationEvent], dict[str, str]]:
    """Return (events in chronological order, event_id -> persona user_id ground truth).

    The first event of each customer carries all three identifiers (so variants can be linked);
    later events carry a random non-empty subset in a random formatting variant.
    If `truth` is given it is filled with event_id -> expected facts (for extraction evaluation).
    """
    rng = random.Random(seed)
    start = start or datetime(2026, 1, 1, tzinfo=timezone.utc)
    personas = [make_persona(i, rng) for i in range(customers)]
    events: list[ConversationEvent] = []
    owner: dict[str, str] = {}
    for p in personas:
        t = start + timedelta(hours=rng.randint(0, 72))
        for c in range(conversations):
            conv_id = f"conv-{seed}-{p.index}-{c}"
            for n, (role, text, expected) in enumerate(_messages(p, rng, first=(c == 0))):
                t += timedelta(minutes=rng.randint(1, 30))
                if c == 0 and n == 0:
                    parts = {"user_id", "email", "phone"}
                else:
                    parts = set(rng.sample(["user_id", "email", "phone"], rng.randint(1, 3)))
                ids = Identifiers(
                    user_id=rng.choice(_uid_variants(p.user_id)) if "user_id" in parts else None,
                    email=rng.choice(_email_variants(p.email)) if "email" in parts else None,
                    phone=rng.choice(_phone_variants(p.phone)) if "phone" in parts else None,
                )
                ev = ConversationEvent(
                    event_id=f"evt-{seed}-{p.index}-{c}-{n}", conversation_id=conv_id, timestamp=t,
                    role=role, text=text, identifiers=ids, channel=rng.choice(["chat", "email", "voice"]),
                )
                events.append(ev)
                if truth is not None:
                    truth[ev.event_id] = expected if role == "customer" else set()
                owner[ev.event_id] = p.user_id
            t += timedelta(days=rng.randint(1, 20))
    events.sort(key=lambda e: e.timestamp)
    return events, owner
