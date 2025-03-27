"""Deterministic rule-based stand-in for an LLM.

It honours the same contract as a real model: it receives the rendered prompt text, finds the
<message> block, and returns a JSON document matching ExtractionResult. Same input, same output,
no network - which makes the whole pipeline testable and the demo runnable offline.
"""

from __future__ import annotations

import html
import json
import re

_MSG_RE = re.compile(r"<message>\n(.*?)\n</message>", re.S)
_TIME_RE = re.compile(r"<event_time>(.*?)</event_time>")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(r"(?<![\w@])(\+?\(?\d[\d\s().-]{7,}\d)(?![\w@])")
_NAME = re.compile(r"(?i:my name is)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)")
_CHANNEL = re.compile(r"(?i)\b(e-?mail|phone|call|sms|text|whatsapp)\b")
_PREFER_CUE = re.compile(r"(?i)\b(prefer|rather|please|best|better|only)\b")
_AVOID = re.compile(r"(?i)\b(?:do not|don't|dont|never|stop)\b[^.!?]{0,25}?\b(e-?mail|call|text|sms)")
_TIME = re.compile(r"(?i)\b((?:after|before)\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?|in the (?:morning|afternoon|evening)|on weekends|on weekdays)")
_CONTACT_CUE = re.compile(r"(?i)\b(contact|call|reach|text|email|message|ping)\b")
_LANG = re.compile(r"(?i)\b(?:speak|prefer|in|reply in|write in)\s+(spanish|french|german|english|portuguese|italian|japanese)\b")
_LIKE = re.compile(
    r"(?i)\bI\s+(?:really\s+|do\s+)?(?:(prefer|like|love|enjoy)|(?:would|'d)\s+(prefer))\s+(?:the\s+|to\s+use\s+)?"
    r"([a-z][a-z\s-]{2,40}?)(?=\s+(?:over|rather|to|because|but|and|when|so)\b|[.,!?;]|$)"
)
_DISLIKE = re.compile(
    r"(?i)\bI\s+(?:really\s+)?(?:hate|dislike|can't stand|cannot stand)\s+(?:the\s+)?([a-z][a-z\s-]{2,40}?)"
    r"(?=\s+(?:because|but|and|when|so)\b|[.,!?;]|$)"
)
_HEDGE = re.compile(r"(?i)\b(maybe|i think|i guess|not sure|perhaps|possibly|might)\b")

_PROBLEM = re.compile(
    r"(?i)\b(not working|doesn'?t work|does not work|isn'?t working|broken|can'?t|cannot|can not|won'?t|unable|"
    r"problem|issue|trouble|failed|failing|error|wrong|twice|late|missing|never (?:arrived|came)|"
    r"still (?:waiting|haven'?t)|crash\w*|freez\w*|stuck|overcharged|locked out)\b"
)
_NO_PROBLEM = re.compile(r"(?i)\b(no problem|not a problem|no issue|no worries)\b")
_RESOLVED = re.compile(r"(?i)\b(fixed|resolved|working now|works now|sorted|all good now|solved)\b")
_TOPICS = {
    "login": re.compile(r"(?i)log ?in|sign ?in|password|locked out|2fa|verification code"),
    "billing": re.compile(r"(?i)charge[ds]?|billing|invoice|refund|payment|overcharged|subscription fee"),
    "shipping": re.compile(r"(?i)deliver|shipping|package|parcel|tracking|arrive"),
    "app_stability": re.compile(r"(?i)crash|freez|frozen|\bbug\b|glitch|not loading|won'?t load|\bslow\b"),
}
_CHANNEL_VALUE = {"email": "email", "e-mail": "email", "phone": "phone", "call": "phone", "sms": "sms",
                  "text": "sms", "whatsapp": "whatsapp"}


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:40]


def _channel(word: str) -> str:
    return _CHANNEL_VALUE[word.lower()]


def extract_facts(message: str, event_time: str) -> list[dict]:
    facts: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    def add(kind: str, key: str, value: str, conf: float, evidence: str, status: str | None = None) -> None:
        if (kind, key, value) in seen or not key:
            return
        seen.add((kind, key, value))
        f = {"kind": kind, "key": key, "value": value, "confidence": round(conf, 2),
             "observed_at": event_time, "evidence": evidence[:280]}
        if status:
            f["status"] = status
        facts.append(f)

    sentences = [s.strip() for s in _SENT_SPLIT.split(message) if s.strip()]
    msg_topics = {t for t, rx in _TOPICS.items() if rx.search(message)}
    for sent in sentences:
        hedge = 0.5 if _HEDGE.search(sent) else 1.0

        # --- contact details -------------------------------------------------------
        for m in _EMAIL.finditer(sent):
            cue = re.search(r"(?i)my e-?mail|reach me|contact me|e-?mail is|send .* to", sent)
            add("contact_detail", "email", m.group(0), (0.95 if cue else 0.7) * hedge, sent)
        phone_scan = _EMAIL.sub(" ", sent)
        for m in _PHONE.finditer(phone_scan):
            digits = re.sub(r"\D", "", m.group(1))
            if 10 <= len(digits) <= 15:
                cue = re.search(r"(?i)phone|number|call me|reach me|cell|mobile", sent)
                add("contact_detail", "phone", m.group(1).strip(), (0.9 if cue else 0.55) * hedge, sent)
        if m := _NAME.search(sent):
            add("contact_detail", "preferred_name", m.group(1), 0.9 * hedge, sent)

        # --- preferences -----------------------------------------------------------
        handled = False
        for m in _AVOID.finditer(sent):
            ch = _channel(m.group(1))
            add("preference", f"avoid_channel_{ch}", f"do not contact via {ch}", 0.9 * hedge, sent)
            handled = True
        if not handled and _PREFER_CUE.search(sent) and (m := _CHANNEL.search(sent)) and not _EMAIL.search(sent):
            add("preference", "contact_channel", _channel(m.group(1)), 0.85 * hedge, sent)
            handled = True
        if _CONTACT_CUE.search(sent) and (m := _TIME.search(sent)):
            add("preference", "contact_time", m.group(1).lower(), 0.75 * hedge, sent)
            handled = True
        if m := _LANG.search(sent):
            add("preference", "language", m.group(1).lower(), 0.8 * hedge, sent)
            handled = True
        if not handled and (m := _LIKE.search(sent)):
            verb = (m.group(1) or m.group(2)).lower()
            add("preference", f"likes_{_slug(m.group(3))}", m.group(3).strip().lower(),
                (0.6 if verb == "like" else 0.78) * hedge, sent)
        if m := _DISLIKE.search(sent):
            add("preference", f"dislikes_{_slug(m.group(1))}", m.group(1).strip().lower(), 0.7 * hedge, sent)

        # --- issues ----------------------------------------------------------------
        if _NO_PROBLEM.search(sent):
            continue
        topics = [t for t, rx in _TOPICS.items() if rx.search(sent)]
        if _RESOLVED.search(sent):
            topic = topics[0] if topics else (next(iter(msg_topics)) if len(msg_topics) == 1 else None)
            if topic:
                add("issue", topic, sent[:200], 0.85 * hedge, sent, "resolved")
        elif _PROBLEM.search(sent):
            if topics:
                add("issue", topics[0], sent[:200], 0.85 * hedge, sent, "open")
            elif re.search(r"(?i)\b(problem|issue)\b", sent):
                add("issue", "other", sent[:200], 0.55 * hedge, sent, "open")
    return facts


class StubLLM:
    name = "stub-rules-v1"

    def generate(self, *, system: str, prompt: str, json_schema: dict | None = None) -> str:
        m = _MSG_RE.search(prompt)
        if not m:
            return json.dumps({"facts": []})
        t = _TIME_RE.search(prompt)
        message = html.unescape(m.group(1))
        return json.dumps({"facts": extract_facts(message, t.group(1) if t else "")})
