#!/usr/bin/env python
"""Quality report on synthetic data: extraction F1 and the crossed-memory (zero-leakage) check.

    python scripts/evaluate.py                       # 300 customers, ~1,200 conversations
    python scripts/evaluate.py --customers 800 --json
Exits non-zero if any retrieved memory belongs to a different customer.
"""
import argparse
import json
import sys

from memoryforge.evaluation import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--customers", type=int, default=300)
ap.add_argument("--conversations", type=int, default=4, help="conversations per customer")
ap.add_argument("--seed", type=int, default=21)
ap.add_argument("--json", action="store_true")
a = ap.parse_args()
r = evaluate(a.customers, a.conversations, a.seed)
if a.json:
    print(json.dumps(r.as_dict(), indent=2))
else:
    print(f"customers={r.customers} conversations={r.conversations} events={r.events} ({r.seconds}s)")
    print(f"identity: {r.customers_resolved_uniquely}/{r.customers} personas resolved to exactly one distinct customer")
    print(f"crossed memory: {r.leaked_results} leaked of {r.results_checked} results over {r.retrievals} retrievals")
    print(f"extraction: precision={r.precision} recall={r.recall} F1={r.f1}")
sys.exit(1 if r.leaked_results or r.customers_resolved_uniquely != r.customers else 0)
