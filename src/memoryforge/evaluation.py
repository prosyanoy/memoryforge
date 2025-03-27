"""Quality evaluation on the synthetic set: extraction P/R/F1 and the crossed-memory (leakage) check."""

from __future__ import annotations

import io
import random
import time
from dataclasses import dataclass, asdict

from .config import Settings
from .services import build_services
from .synthetic import Fact, generate_events
from .telemetry import configure_logging


@dataclass
class EvalReport:
    customers: int
    conversations: int
    events: int
    retrievals: int
    results_checked: int
    leaked_results: int
    customers_resolved_uniquely: int
    precision: float
    recall: float
    f1: float
    seconds: float

    def as_dict(self) -> dict:
        return asdict(self)


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def evaluate(customers: int = 300, conversations: int = 4, seed: int = 21, queries_per_customer: int = 12,
             vector_backend: str = "auto") -> EvalReport:
    t0 = time.perf_counter()
    configure_logging("ERROR", "json", stream=io.StringIO())
    svc = build_services(Settings(_env_file=None, db_path=":memory:", run_worker=False, vector_backend=vector_backend))
    truth: dict[str, set[Fact]] = {}
    events, owner = generate_events(customers, conversations, seed, truth=truth)

    # 1) extraction quality: extractor + forget rules vs ground truth, per customer message
    tp = fp = fn = 0
    for e in events:
        if e.role != "customer":
            continue
        predicted: set[Fact] = set()
        for f in svc.extractor.extract(e).facts:
            d = svc.policy.evaluate(f)
            if d.fact is not None:
                predicted.add((d.fact.kind.value, d.fact.key, d.fact.value.lower(),
                               d.fact.status.value if d.fact.status else None))
        expected = {(k, key, v.lower(), st) for k, key, v, st in truth[e.event_id]}
        # issue values are free text in the stub's output; compare issues on (kind, key, status)
        norm = lambda s: {(k, key, "" if k == "issue" else v, st) for k, key, v, st in s}  # noqa: E731
        predicted, expected = norm(predicted), norm(expected)
        tp += len(predicted & expected)
        fp += len(predicted - expected)
        fn += len(expected - predicted)
    precision, recall, f1 = _prf(tp, fp, fn)

    # 2) run the real pipeline, then probe for crossed memory
    for e in events:
        svc.queue.publish(e)
    svc.worker.drain()
    cust_of_user = {}
    for e in events:
        if e.identifiers.user_id:
            cust_of_user[owner[e.event_id]] = svc.identity.lookup(e.identifiers.user_id)
    unique = len(set(cust_of_user.values()))
    event_owner = {e.event_id: cust_of_user[owner[e.event_id]] for e in events}
    all_values = [m.value for c in list(cust_of_user.values())[:50] for m in svc.store.list_memories(c)]
    rng = random.Random(seed)
    base_queries = ["", "contact", "login problem", "billing", "email address", "phone number", "language",
                    "what do they like", "shipping"]
    retrievals = checked = leaked = 0
    for cid in cust_of_user.values():
        queries = base_queries + [rng.choice(all_values) for _ in range(max(0, queries_per_customer - len(base_queries)))]
        for q in queries:
            for hist in (False, True):
                retrievals += 1
                for r in svc.memory.retrieve(cid, q, k=100, include_history=hist).results:
                    checked += 1
                    if r.memory.customer_id != cid or event_owner[r.memory.source_event_id] != cid:
                        leaked += 1
    n_conv = len({e.conversation_id for e in events})
    svc.close()
    return EvalReport(customers, n_conv, len(events), retrievals, checked, leaked, unique,
                      round(precision, 4), round(recall, 4), round(f1, 4), round(time.perf_counter() - t0, 1))
