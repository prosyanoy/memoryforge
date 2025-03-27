#!/usr/bin/env python
"""Scripted ~60 s walkthrough, built for screen recording: replay a conversation, watch memory update, query it.

    python scripts/demo.py            # paced for recording
    python scripts/demo.py --pace 0   # instant
"""
import argparse
import io
import time
from datetime import datetime, timedelta, timezone

from memoryforge.config import Settings
from memoryforge.schemas import ConversationEvent, Identifiers
from memoryforge.services import build_services
from memoryforge.telemetry import configure_logging

ap = argparse.ArgumentParser()
ap.add_argument("--pace", type=float, default=4.0, help="seconds to pause between steps")
pace = ap.parse_args().pace
configure_logging("ERROR", "json", stream=io.StringIO())
svc = build_services(Settings(_env_file=None, db_path=":memory:", run_worker=False))
t0 = datetime(2026, 5, 1, 9, 0, tzinfo=timezone.utc)

CONVERSATION = [
    ("customer", {"email": "Dana.Lee@Example.com"}, "Hi, my name is Dana. My email is dana.lee@example.com. My package never arrived."),
    ("agent", {"email": "dana.lee@example.com"}, "Sorry about that, I'm checking the tracking now."),
    ("customer", {"phone": "(415) 555-0142", "email": "dana.lee@example.com"}, "I prefer email. My card is 4242 4242 4242 4242 if you need it."),
    ("customer", {"phone": "+1 415 555 0142"}, "Actually please text me instead, after 5pm. The package arrived, all good now."),
]


def show(title: str) -> None:
    print(f"\n=== {title}")
    time.sleep(pace)


def dump(ref: str, q: str = "") -> None:
    cid = svc.identity.lookup(ref)
    res = svc.memory.retrieve(cid, q, k=8)
    print(f"customer={cid} q={q!r} ({res.took_ms} ms)")
    for r in res.results:
        m = r.memory
        print(f"  {r.score:.3f}  {m.kind.value:<14} {m.key:<16} {m.value}" + (f"  [{m.status.value}]" if m.status else ""))


for i, (role, ids, text) in enumerate(CONVERSATION):
    show(f"event {i + 1}/{len(CONVERSATION)} [{role}] {ids}\n    \"{text}\"")
    svc.queue.publish(ConversationEvent(event_id=f"demo-{i}", conversation_id="demo", timestamp=t0 + timedelta(minutes=5 * i),
                                        role=role, text=text, identifiers=Identifiers(**ids)))
    print("  pipeline ->", dict(svc.worker.drain()))
    dump("dana.lee@example.com")

show("query by a different identifier variant: phone")
dump("415-555-0142", "how should we contact her")
show("replay the whole log (idempotent) and show the stage metrics")
print("  replay ->", dict(svc.worker.replay(0)))
for line in svc.telemetry.metrics.render().decode().splitlines():
    if line.startswith("memoryforge_stage_total") and 'status="ok"' in line:
        print(" ", line)
print("\nNote: the card number was redacted before extraction and never stored; superseded facts are history.")
