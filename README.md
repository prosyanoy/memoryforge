# MemoryForge

An end-to-end mini **customer memory** service: it ingests raw conversation events, extracts structured facts with an LLM-style extractor, resolves them to the right customer, and retrieves them in milliseconds.

Everything runs locally with zero secrets and zero downloads by default (SQLite, in-process queue, deterministic stub extractor, hashing embedder). Kafka/Redpanda, Ollama and real embeddings are drop-in options.

```
  conversation events (Kafka/Redpanda | in-process queue)
            │
            ▼
  ┌──────────┐   ┌───────────┐   ┌───────────────┐   ┌──────────────────┐   ┌────────────────────┐
  │  INGEST  │──►│  RESOLVE  │──►│    EXTRACT    │──►│      INDEX       │──►│      RETRIEVE      │
  │ dedupe,  │   │ phone/    │   │ versioned     │   │ forget rules,    │   │ GET /memory/{id}?q │
  │ offsets, │   │ email/id  │   │ prompt + LLM  │   │ supersession,    │   │ per-customer FAISS │
  │ replay   │   │ → 1 cust. │   │ → Pydantic    │   │ embed → SQLite   │   │ + recency/conf.    │
  └──────────┘   └───────────┘   │ facts         │   │ + FAISS (1 tx)   │   │ ranking            │
       │               │         └───────────────┘   └──────────────────┘   └────────────────────┘
       └───────────────┴───── any failure ► dead letter (stage, raw output, correlation id, trace) ─┘
        every stage: structlog line + OTel span + Prometheus counter/histogram
```

(Resolve runs before extract so an event is always tied to a customer before any model call.)

## Results

All numbers are reproducible with one command each, on synthetic data (1 vCPU dev VM, FAISS, offline hash embedder, stub extractor).

| Metric | Result | Command |
|---|---|---|
| p99 retrieval latency, service layer | **0.9 ms** (p50 0.66, p95 0.81) | `make bench` |
| p99 retrieval latency, HTTP via ASGI test client | **6.5–7.9 ms** (< 10 ms, includes test-client overhead) | `make bench` |
| Cross-customer leakage | **0** leaked of 68,604 retrieved memories (7,200 retrievals, 300 customers, 1,200 conversations, 4,698 events; history on and off, queries seeded with other customers' facts) | `make eval` |
| Identity resolution | 300/300 personas (phone/email/user-id variants) resolved to exactly one distinct customer | `make eval` |
| Extraction precision / recall / F1 | 1.000 / 1.000 / **1.000** on the synthetic set (stub extractor) | `make eval` |

How to read these honestly: the synthetic set and the stub extractor are written by the same author from the same templates, so F1 = 1.0 is a regression guard and a harness for comparing real models (`MEMORYFORGE_LLM_BACKEND=ollama`), not a claim about real-world accuracy. Issue facts are scored on (kind, topic, status) because their value is free text. The leakage check is an empirical test over synthetic data plus the structural isolation (per-customer index, customer-scoped SQL), not a formal proof.

## Demo

`python scripts/demo.py` is a ~60-second paced walkthrough meant for screen recording: it replays a four-message conversation (identifier variants, a redacted card number, a changed preference, a resolved issue), prints the memory after every event, queries it by a different identifier, replays the log idempotently, and prints the per-stage metrics. For live dashboards, point Prometheus at `GET /metrics` (`memoryforge_stage_total`, `memoryforge_stage_duration_seconds_bucket`, `memoryforge_events_total`, `memoryforge_facts_total`) and graph in Grafana.

## Features

| Area | What you get |
|---|---|
| **Ingestion** | Consume conversation events from a Kafka/Redpanda topic or an in-process queue. Offsets are committed *after* processing; `event_id` is an idempotency key, so redelivery and **replay** never duplicate memory. Poison messages are dead-lettered instead of blocking the stream. |
| **Extraction** | Versioned prompts (`v1`, `v2`) + a Pydantic JSON schema (`ExtractionResult`) producing *preference / issue / contact_detail* facts with confidence, timestamps, status and evidence. Output is validated, retried once with a repair hint, and dead-lettered with the raw model output if it never validates. |
| **What to forget** | Rule set (`forgetting.py`): PII is redacted *before* the model sees it (cards with Luhn check, SSN, IBAN, passwords/PINs) and any fact that still carries it is dropped; low-confidence facts are dropped; newer facts supersede older ones (history is kept inactive, late/out-of-order data cannot overwrite newer facts); a retention sweep expires resolved issues and old facts; `DELETE /memory/{customer}` erases a customer entirely. |
| **Identity resolution** | Phone / email / user-id variants (`(415) 555-0100`, `+1 415…`, `MAILTO:Ann@X.com`, `USR_1`…) normalise to one customer id. Conflicts (an event touching two customers) **fail closed** into the dead-letter table; merging is an explicit admin action. |
| **Isolation guarantee** | Vector search is partitioned per customer, and every SQL read is also constrained by `customer_id`. Tested with a property test over a synthetic population, plus per-layer tests. |
| **Retrieval** | `GET /memory/{customer}?q=...` → embed query → per-customer FAISS search → hydrate → rank by `0.6·similarity + 0.25·recency (30-day half-life) + 0.15·confidence`. Returns a score breakdown per result. |
| **Observability** | structlog JSON logs with correlation IDs (HTTP header → event → every stage) and trace/span ids; Prometheus metrics per stage (`/metrics`); OpenTelemetry spans for every stage with an in-process buffer, optional console/OTLP export; failed events keep their full trace in the dead-letter record (`GET /failures/{event_id}`). |

## Architecture

```
 POST /events ─┐                                       ┌────────────── GET /memory/{id}?q= ─────────────┐
 produce CLI ──┼─► Queue (Kafka/Redpanda | in-memory)  │ resolve id → embed q → FAISS (per customer) →  │
               │      │ poll / commit-after / seek     │ hydrate (SQL, customer-scoped) → rank → top-k  │
               │      ▼                                └────────────────────────────────────────────────┘
               │   Worker ─► Pipeline (one trace per event, one span + metrics per stage)
               │      dedupe ─► identity ─► extract ─► forget ─► embed ─► store
               │        │          │           │          │               │ one SQLite transaction:
               │  processed_   customers/   prompt +     PII redaction,   │ insert facts, supersede old
               │  events       identifiers  LLM (stub/   low-confidence,  │ ones, mark event processed;
               │  (idempotency)             Ollama) →    contact          ▼ then update vector index
               │                            Pydantic     validation     SQLite (source of truth)
               │   failures at any stage ─► dead_letters (stage, error, raw output, correlation id, trace spans)
```

Code map (`src/memoryforge/`): `schemas` (Pydantic models) · `prompts`, `llm/` (stub, Ollama), `extraction` · `forgetting` · `identity` · `store` (SQLite) · `embeddings` (hash, fastembed) · `vector` (FAISS/NumPy) · `memory` (write path + ranking) · `queue` · `pipeline` (+ worker/replay) · `telemetry` · `api` · `synthetic` (data generator) · `cli`.

## Setup

Requires Python 3.11+.

```bash
make install            # or: python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
cp .env.example .env    # optional - every value has a safe default
```

Optional extras: `.[embeddings]` (fastembed), `.[kafka]` (confluent-kafka), `.[otlp]` (OTLP exporter).

## Run

```bash
make run                # API on http://127.0.0.1:8000 with 20 synthetic customers (docs at /docs)
```

Try it:

```bash
curl -s localhost:8000/healthz
CID=$(curl -s "localhost:8000/customers/resolve?user_id=usr-000001" | python3 -c "import sys,json;print(json.load(sys.stdin)['customer_id'])")
curl -s "localhost:8000/memory/$CID?q=how+should+we+contact+them&k=3"

# ingest a live event (wait=true blocks until it was processed); the same customer is found by phone variant
curl -s -X POST "localhost:8000/events?wait=true" -H 'X-Correlation-ID: demo-1' -H 'content-type: application/json' -d '{
  "event_id":"live-1","conversation_id":"c-1","timestamp":"2026-05-01T10:00:00Z",
  "text":"Actually please text me. My card is 4111 1111 1111 1111.",
  "identifiers":{"phone":"(415) 555-0001"}}'
curl -s "localhost:8000/memory/%2B14155550001?q=contact"      # any known identifier works as the path ref
curl -s localhost:8000/metrics | grep memoryforge_stage
curl -s localhost:8000/failures                                # dead letters; /failures/{event_id} adds trace spans
```

Other commands: `python -m memoryforge.cli seed|produce|consume|serve --help`; `uvicorn memoryforge.api:app_factory --factory`.
Admin routes (`/admin/replay`, `/admin/merge`, `/admin/sweep`, `/failures`, `DELETE /memory/…`) are open locally; set `MEMORYFORGE_ADMIN_TOKEN` and send `X-Admin-Token` to protect them.

### Kafka / Redpanda

```bash
docker compose up -d redpanda redpanda-init          # broker on localhost:19092, topic conversation.events
pip install -e ".[kafka]"
export MEMORYFORGE_QUEUE_BACKEND=kafka
python -m memoryforge.cli serve                      # API + consumer
python -m memoryforge.cli produce --customers 50     # in another shell: publish synthetic traffic
curl -X POST localhost:8000/admin/replay -H 'content-type: application/json' -d '{"from_offset":0,"rebuild":false}'
```

`rebuild: true` wipes derived memory (identities are kept) and re-extracts the whole log, e.g. after changing `MEMORYFORGE_PROMPT_VERSION`. The Kafka path is covered by unit tests against a fake broker; I could not run a live broker in the build environment, so treat it as lightly integration-tested.

### Real models (optional)

```bash
docker compose --profile llm up -d ollama && docker compose exec ollama ollama pull llama3.2
export MEMORYFORGE_LLM_BACKEND=ollama      # structured output via Ollama's `format` JSON schema
export MEMORYFORGE_EMBEDDER=fastembed      # BAAI/bge-small-en-v1.5, downloaded on first use
```

Changing the embedder is safe: on startup, stored vectors from a different embedder are re-embedded.

## Test

```bash
make test        # 82 tests, ~4 s, no network or services needed
```

Covers schemas, the stub extractor and retry/repair logic, prompt-injection escaping, PII and forget rules, identity normalisation/conflicts/merge, idempotency, out-of-order data, replay (idempotent and rebuild), dead letters with trace spans, correlation ids in logs, metrics per stage, both vector backends, ranking, retention, the HTTP API, admin auth, the Kafka queue logic (fake broker), and the **isolation guarantee** (property test over 25 customers × ~35 queries, with history on and off; `make eval` repeats it at 300 customers / 1,200 conversations).

## Benchmark

```bash
make bench       # python scripts/benchmark.py --whale 5000   (see --help; --json for machine output)
```

It ingests synthetic conversations through the real pipeline, then times `GET /memory/{id}?q=…` over thousands of requests. Sample run (1 vCPU dev VM, FAISS, hash embedder, 300 customers / 2.4k memories, plus one 5,000-fact customer in half the queries):

| layer | p50 | p95 | p99 |
|---|---|---|---|
| retrieval service (in-process) | 0.66 ms | 0.81 ms | 0.90 ms |
| HTTP via ASGI test client | 4.4 ms | 5.1 ms | 6.5 ms |

The HTTP row is dominated by the Starlette `TestClient` (thread-hop + httpx) rather than the service; real servers differ, so run the script on your own hardware. Query embeddings are cached, so repeated queries skip the embedder (with fastembed, a cold query costs one ONNX forward pass). Ingestion throughput through the full pipeline was ~430 events/s with the stub extractor.

## Key design decisions

- **Store is the source of truth, the index is derived.** Facts, supersession and the idempotency marker commit in *one* SQLite transaction; the FAISS index is updated afterwards and rebuilt from SQLite on startup. A crash can never leave a half-processed event.
- **At-least-once delivery + idempotent consumer = effectively exactly-once.** Offsets commit after processing; the `processed_events` marker makes redelivery a no-op. Failed events are *not* marked processed, so replay retries them (the dead-letter row counts attempts and clears on success).
- **Isolation by construction, then by check.** One vector partition per customer (no filter to forget), customer-scoped SQL on every read, identity conflicts quarantined rather than merged, and a property test that asserts every returned memory traces back to an event of the queried customer. Authentication/authorisation of *who may call the API* is out of scope for the MVP; the guarantee is about the memory layer.
- **Redact first, then ask the model.** High-risk PII never reaches the LLM or the database; the post-extraction filter is a second line of defence. Email/phone are intentionally kept: they are first-class contact details in the spec. The PII rules are regex-based and deliberately conservative; they are not a compliance product.
- **LLM output is untrusted.** Strict schema validation, one repair retry, timestamps clamped to the event time (models invent dates), message text HTML-escaped inside the prompt so it cannot close the `<message>` block, and agent-role messages are never extracted.
- **Latest observation wins per (kind, key).** Preferences like `contact_channel` are single-valued slots; issues are keyed by topic (`login`, `billing`, …) with an `open`/`resolved` status. Late-arriving older data is dropped, so the active set is independent of arrival order.
- **Offline-first embeddings.** The default `hash` embedder (signed feature hashing + a tiny concept lexicon) is lexical-ish, not truly semantic; it keeps tests and demos hermetic. `fastembed` gives real semantic vectors with one env var. FAISS is `IndexFlatIP` (exact) per customer; per-customer memory is small, so exact search is both fastest and simplest. A NumPy backend is used if FAISS is unavailable.
- **Stub extractor = same contract as an LLM.** It parses the rendered prompt and returns JSON, so the entire stack (prompting, parsing, retries, forgetting) is exercised identically for the stub and for Ollama.
- **Observability is built in, not bolted on.** One trace per event, one span + Prometheus counter/histogram per stage (`dedupe, identity, extract, forget, embed, store, retrieve_*`). Failures snapshot their trace into the dead-letter row, so diagnosis works even after the in-memory span buffer rolls over.
- **SQLite only.** A single locked connection (WAL for files) is enough for the MVP and keeps `:memory:` test databases trivial. Moving to Postgres means reimplementing `store.Store` (+ pgvector would replace FAISS); the rest of the code only talks to that interface.
- **Not handled (by design):** multi-partition Kafka ordering/rebalancing (topic is created with one partition), identifier reassignment (a phone passed to a new owner), multi-worker horizontal scaling (the in-process FAISS index is per process), and API authentication.

## Configuration

All settings are `MEMORYFORGE_*` environment variables (see `.env.example`, defined in `src/memoryforge/config.py`). No real secrets are needed or stored anywhere.
