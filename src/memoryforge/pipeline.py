"""The ingestion pipeline: dedupe -> identity -> extract -> forget -> embed -> store, plus the queue worker."""

from __future__ import annotations

import threading
from collections import Counter
from typing import Callable

import structlog
from opentelemetry.trace import Status, StatusCode
from pydantic import ValidationError

from .extraction import Extractor
from .identity import IdentityResolver
from .memory import MemoryService
from .queue import EventQueue, QueueMessage
from .schemas import ConversationEvent
from .store import Store
from .telemetry import Telemetry

log = structlog.get_logger()


class Pipeline:
    def __init__(self, store: Store, identity: IdentityResolver, extractor: Extractor, memory: MemoryService,
                 telemetry: Telemetry):
        self.store, self.identity, self.extractor = store, identity, extractor
        self.memory, self.tel = memory, telemetry

    def process(self, event: ConversationEvent, offset: int | None = None) -> str:
        """Process one event. Returns the outcome: processed | duplicate | skipped | failed."""
        cid = event.correlation_id or event.event_id
        failure: BaseException | None = None
        outcome = "processed"
        trace_id = None
        with structlog.contextvars.bound_contextvars(
            correlation_id=cid, event_id=event.event_id, conversation_id=event.conversation_id
        ):
            with self.tel.tracer.start_as_current_span(
                "pipeline.process", attributes={"event_id": event.event_id, "correlation_id": cid}
            ) as root:
                trace_id = format(root.get_span_context().trace_id, "032x")
                try:
                    outcome = self._run(event, offset)
                except Exception as exc:  # noqa: BLE001 - every failure must end up in the dead-letter table
                    failure, outcome = exc, "failed"
                    root.record_exception(exc)
                    root.set_status(Status(StatusCode.ERROR, str(exc)[:200]))
                root.set_attribute("outcome", outcome)
            self.tel.metrics.events_total.labels(outcome).inc()
            if failure is not None:
                self._dead_letter(event, offset, cid, trace_id, failure)
            else:
                log.info("event_done", outcome=outcome)
        return outcome

    def _run(self, event: ConversationEvent, offset: int | None) -> str:
        tel = self.tel
        with tel.stage("dedupe"):
            if self.store.is_processed(event.event_id):
                log.info("event_duplicate_skipped")
                return "duplicate"
        with tel.stage("identity") as span:
            res = self.identity.resolve(event.identifiers)
            span.set_attribute("customer_id", res.customer_id)
        structlog.contextvars.bind_contextvars(customer_id=res.customer_id)
        if event.role != "customer":
            self.store.mark_processed(event.event_id, "skipped", offset)
            return "skipped"
        with tel.stage("extract") as span:
            result = self.extractor.extract(event)
            span.set_attribute("prompt_version", result.prompt_version)
            span.set_attribute("facts_extracted", len(result.facts))
        with tel.stage("forget") as span:
            kept = []
            for fact in result.facts:
                d = self.extractor.policy.evaluate(fact)
                if d.fact is None:
                    tel.metrics.facts_total.labels(f"forgotten:{d.reason}").inc()
                    log.info("fact_forgotten", reason=d.reason, kind=fact.kind.value, key=fact.key)
                else:
                    kept.append(d.fact)
            span.set_attribute("facts_kept", len(kept))
        with tel.stage("embed"):
            vectors = self.memory.embed_facts(kept)
        with tel.stage("store") as span:
            stats = self.memory.commit(event, res.customer_id, kept, vectors, prompt_version=result.prompt_version,
                                       extractor=result.extractor, offset=offset)
            span.set_attribute("facts_stored", stats["stored"])
        log.info("facts_stored", **stats)
        return "processed"

    def _dead_letter(self, event: ConversationEvent, offset: int | None, cid: str, trace_id: str | None,
                     exc: BaseException) -> None:
        stage = getattr(exc, "memoryforge_stage", "unknown")
        log.error("event_failed", stage=stage, error_type=type(exc).__name__, error=str(exc)[:300])
        detail = str(exc)
        raw = getattr(exc, "raw", None)
        if raw:
            detail += f" | raw model output: {raw[:500]}"
        self.store.record_dead_letter(
            event_id=event.event_id, correlation_id=cid, trace_id=trace_id, stage=stage,
            error_type=type(exc).__name__, error=detail, payload=event.model_dump_json(),
            spans=self.tel.spans.by_trace(trace_id) if trace_id else [], offset=offset,
        )


class Worker:
    """Consumes the queue, feeds the pipeline, commits offsets after processing (at-least-once)."""

    def __init__(self, queue: EventQueue, pipeline: Pipeline, telemetry: Telemetry, batch_size: int = 100):
        self.queue, self.pipeline, self.tel, self.batch_size = queue, pipeline, telemetry, batch_size
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _handle(self, msg: QueueMessage) -> str:
        try:
            event = ConversationEvent.model_validate_json(msg.value)
        except ValidationError as exc:
            # A poison message must not block the partition: dead-letter it under a synthetic id.
            eid = f"undecodable-{msg.partition}-{msg.offset}"
            self.pipeline.store.record_dead_letter(
                event_id=eid, correlation_id=eid, trace_id=None, stage="decode", error_type="ValidationError",
                error=str(exc)[:1000], payload=msg.value.decode("utf-8", "replace")[:2000], spans=[], offset=msg.offset,
            )
            self.tel.metrics.stage_total.labels("decode", "error").inc()
            self.tel.metrics.events_total.labels("failed").inc()
            log.error("event_undecodable", offset=msg.offset, error=str(exc)[:200])
            return "failed"
        return self.pipeline.process(event, msg.offset)

    def run_once(self, timeout: float = 0.0) -> Counter:
        outcomes: Counter = Counter()
        with self._lock:
            for msg in self.queue.poll(self.batch_size, timeout):
                outcomes[self._handle(msg)] += 1
                self.queue.commit(msg)
        return outcomes

    def drain(self) -> Counter:
        """Process until the queue is empty (used by tests, the CLI and ?wait=true)."""
        total: Counter = Counter()
        while True:
            got = self.run_once(0.0)
            if not got:
                return total
            total.update(got)

    def replay(self, from_offset: int = 0, rebuild: bool = False) -> Counter:
        """Re-read the log. Idempotent by default; `rebuild` first wipes derived memory state
        (identities are kept) so the whole log is re-extracted, e.g. after a prompt change."""
        with self._lock:
            if rebuild:
                self.pipeline.store.reset_derived()
                self.pipeline.memory.index.clear()
            self.queue.seek_to(from_offset)
            log.info("replay_started", from_offset=from_offset, rebuild=rebuild)
            return self.drain()

    def start(self, poll_timeout: float = 0.5) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()

        def loop() -> None:
            while not self._stop.is_set():
                try:
                    self.run_once(poll_timeout)
                except Exception:  # noqa: BLE001
                    log.exception("worker_loop_error")
                    self._stop.wait(1.0)

        self._thread = threading.Thread(target=loop, name="memoryforge-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
