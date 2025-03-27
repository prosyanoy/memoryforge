"""Observability: structlog setup, Prometheus metrics, OpenTelemetry tracing with an in-process span buffer."""

from __future__ import annotations

import logging
import sys
import time
from collections import deque
from contextlib import contextmanager
from typing import Iterator, Sequence, TextIO

import structlog
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

from .config import Settings

LATENCY_BUCKETS = (0.0002, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)


# ---- logging ------------------------------------------------------------------------


def _add_trace_ids(_logger, _name, event_dict: dict) -> dict:
    ctx = trace.get_current_span().get_span_context()
    if ctx.is_valid:
        event_dict["trace_id"] = format(ctx.trace_id, "032x")
        event_dict["span_id"] = format(ctx.span_id, "016x")
    return event_dict


def configure_logging(level: str = "INFO", fmt: str = "console", stream: TextIO | None = None) -> None:
    renderer = structlog.processors.JSONRenderer() if fmt == "json" else structlog.dev.ConsoleRenderer(colors=False)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _add_trace_ids,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level.upper())),
        logger_factory=structlog.PrintLoggerFactory(file=stream or sys.stderr),
        cache_logger_on_first_use=False,
    )


def current_correlation_id() -> str | None:
    return structlog.contextvars.get_contextvars().get("correlation_id")


# ---- metrics --------------------------------------------------------------------------


class Metrics:
    """Prometheus metrics on a private registry (so tests/apps never collide)."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.stage_total = Counter(
            "memoryforge_stage_total", "Pipeline stage executions", ["stage", "status"], registry=self.registry
        )
        self.stage_seconds = Histogram(
            "memoryforge_stage_duration_seconds", "Pipeline stage latency", ["stage"],
            buckets=LATENCY_BUCKETS, registry=self.registry,
        )
        self.events_total = Counter(
            "memoryforge_events_total", "Events by final outcome", ["outcome"], registry=self.registry
        )
        self.facts_total = Counter(
            "memoryforge_facts_total", "Facts by disposition (stored / forgotten:<reason>)", ["disposition"],
            registry=self.registry,
        )
        self.retrievals_total = Counter(
            "memoryforge_retrievals_total", "Retrieval requests", ["status"], registry=self.registry
        )

    def render(self) -> bytes:
        return generate_latest(self.registry)


# ---- tracing --------------------------------------------------------------------------


class RingSpanExporter(SpanExporter):
    """Keeps the most recent finished spans in memory so failures can be diagnosed via the API."""

    def __init__(self, size: int = 5000):
        self._spans: deque[dict] = deque(maxlen=size)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        for s in spans:
            ctx = s.get_span_context()
            self._spans.append(
                {
                    "trace_id": format(ctx.trace_id, "032x"),
                    "span_id": format(ctx.span_id, "016x"),
                    "parent_id": format(s.parent.span_id, "016x") if s.parent else None,
                    "name": s.name,
                    "duration_ms": round((s.end_time - s.start_time) / 1e6, 3),
                    "status": s.status.status_code.name,
                    "status_description": s.status.description,
                    "attributes": {k: v for k, v in (s.attributes or {}).items()},
                    "events": [
                        {"name": e.name, "attributes": {k: str(v)[:500] for k, v in (e.attributes or {}).items()}}
                        for e in s.events
                    ],
                    "start_ns": s.start_time,
                }
            )
        return SpanExportResult.SUCCESS

    def by_trace(self, trace_id: str) -> list[dict]:
        return sorted((s for s in list(self._spans) if s["trace_id"] == trace_id), key=lambda s: s["start_ns"])


class Telemetry:
    def __init__(self, settings: Settings | None = None):
        settings = settings or Settings()
        self.metrics = Metrics()
        self.spans = RingSpanExporter(settings.span_buffer_size)
        self.provider = TracerProvider(resource=Resource.create({"service.name": "memoryforge"}))
        self.provider.add_span_processor(SimpleSpanProcessor(self.spans))
        if settings.otel_exporter == "console":
            self.provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        elif settings.otel_exporter == "otlp":
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            except ImportError as exc:
                raise RuntimeError("install the 'otlp' extra for MEMORYFORGE_OTEL_EXPORTER=otlp") from exc
            self.provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        self.tracer = self.provider.get_tracer("memoryforge")
        self._log = structlog.get_logger("memoryforge.stage")

    @contextmanager
    def stage(self, name: str, **attrs) -> Iterator[trace.Span]:
        """Span + Prometheus counter/histogram + error log around one pipeline stage."""
        cid = current_correlation_id()
        if cid:
            attrs.setdefault("correlation_id", cid)
        start = time.perf_counter()
        status = "ok"
        with self.tracer.start_as_current_span(f"stage.{name}", attributes=attrs) as span:
            try:
                yield span
            except BaseException as exc:
                status = "error"
                if not hasattr(exc, "memoryforge_stage"):
                    exc.memoryforge_stage = name  # innermost stage wins
                self._log.error("stage_failed", stage=name, error_type=type(exc).__name__, error=str(exc)[:300])
                raise
            finally:
                self.metrics.stage_total.labels(name, status).inc()
                self.metrics.stage_seconds.labels(name).observe(time.perf_counter() - start)
