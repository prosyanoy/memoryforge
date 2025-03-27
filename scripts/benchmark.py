#!/usr/bin/env python
"""Retrieval latency benchmark: reports p50/p95/p99 for the service layer and the HTTP (ASGI) layer.

    python scripts/benchmark.py --customers 500 --queries 3000
    python scripts/benchmark.py --whale 5000 --json

Builds an in-memory deployment, ingests synthetic conversations through the real pipeline, then
times GET /memory/{id}?q=... Optionally adds one "whale" customer with thousands of distinct facts to
show how latency scales with the size of a single customer's memory (search is per-customer).
"""
from __future__ import annotations

import argparse
import io
import json
import random
import statistics
import time
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from memoryforge.api import create_app
from memoryforge.config import Settings
from memoryforge.schemas import ConversationEvent, Identifiers
from memoryforge.services import build_services
from memoryforge.synthetic import THINGS, generate_events
from memoryforge.telemetry import configure_logging

QUERIES = ["how should we contact this customer", "open billing problem", "login issue", "phone number",
           "email address", "preferred language", "what do they like", "shipping delay", "app crashes",
           "do not call", "best time to reach", "name"]


def pct(samples: list[float], p: float) -> float:
    s = sorted(samples)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def summarize(ms: list[float]) -> dict:
    return {"n": len(ms), "mean_ms": round(statistics.fmean(ms), 3), "p50_ms": round(pct(ms, 50), 3),
            "p95_ms": round(pct(ms, 95), 3), "p99_ms": round(pct(ms, 99), 3), "max_ms": round(max(ms), 3)}


def add_whale(svc, n: int) -> str:
    base = datetime(2026, 2, 1, tzinfo=timezone.utc)
    rng = random.Random(5)
    adjectives = ["red", "quiet", "fast", "tiny", "old", "bright", "heavy", "soft"]
    for i in range(n):
        thing = f"{rng.choice(adjectives)} {rng.choice(THINGS)} model{i}"
        svc.queue.publish(ConversationEvent(
            event_id=f"whale-{i}", conversation_id="whale", timestamp=base + timedelta(minutes=i),
            text=f"I really love {thing}.", identifiers=Identifiers(user_id="whale-customer")))
    svc.worker.drain()
    return svc.identity.lookup("whale-customer")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--customers", type=int, default=300)
    ap.add_argument("--conversations", type=int, default=4)
    ap.add_argument("--queries", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--whale", type=int, default=0, help="extra customer with N distinct facts")
    ap.add_argument("--vector-backend", default="auto", choices=["auto", "faiss", "numpy"])
    ap.add_argument("--embedder", default="hash", choices=["hash", "fastembed"])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    configure_logging("ERROR", "json", stream=io.StringIO())
    svc = build_services(Settings(_env_file=None, db_path=":memory:", run_worker=False, log_level="ERROR",
                                  vector_backend=args.vector_backend, embedder=args.embedder))
    events, owner = generate_events(args.customers, args.conversations, seed=1)
    t0 = time.perf_counter()
    for e in events:
        svc.queue.publish(e)
    outcomes = svc.worker.drain()
    ingest_s = time.perf_counter() - t0
    stats = svc.store.stats()
    cids = sorted({svc.identity.lookup(e.identifiers.user_id) for e in events if e.identifiers.user_id})
    whale = add_whale(svc, args.whale) if args.whale else None

    rng = random.Random(42)
    targets = [(rng.choice(cids), rng.choice(QUERIES)) for _ in range(args.warmup + args.queries)]
    if whale:
        targets = [(whale if i % 2 == 0 else c, q) for i, (c, q) in enumerate(targets)]

    def run(call) -> list[float]:
        for c, q in targets[: args.warmup]:
            call(c, q)
        out = []
        for c, q in targets[args.warmup:]:
            t = time.perf_counter()
            call(c, q)
            out.append((time.perf_counter() - t) * 1000)
        return out

    service_ms = run(lambda c, q: svc.memory.retrieve(c, q, k=10))
    with TestClient(create_app(services=svc)) as client:
        http_ms = run(lambda c, q: client.get(f"/memory/{c}", params={"q": q, "k": 10}).raise_for_status())

    report = {
        "config": {"customers": args.customers, "events": len(events), "whale_facts": args.whale,
                   "vector_backend": svc.index.backend, "embedder": svc.embedder.name},
        "ingest": {"events_per_s": round(len(events) / ingest_s, 1), "outcomes": dict(outcomes),
                   "active_memories": stats["active_memories"]},
        "retrieval_service": summarize(service_ms),
        "retrieval_http_asgi": summarize(http_ms),
    }
    if args.json:
        print(json.dumps(report, indent=2))
        return
    print(f"MemoryForge benchmark | backend={svc.index.backend} embedder={svc.embedder.name} "
          f"customers={args.customers} memories={stats['active_memories']} whale_facts={args.whale}")
    print(f"ingest: {report['ingest']['events_per_s']} events/s through the full pipeline")
    print(f"{'layer':<22}{'n':>6}{'mean':>9}{'p50':>9}{'p95':>9}{'p99':>9}{'max':>9}  (ms)")
    for name, key in (("service (in-process)", "retrieval_service"), ("HTTP (ASGI test client)", "retrieval_http_asgi")):
        r = report[key]
        print(f"{name:<22}{r['n']:>6}{r['mean_ms']:>9}{r['p50_ms']:>9}{r['p95_ms']:>9}{r['p99_ms']:>9}{r['max_ms']:>9}")


if __name__ == "__main__":
    main()
