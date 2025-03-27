"""Composition root: wires settings into a ready-to-use set of services."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from .config import Settings
from .embeddings import Embedder, build_embedder
from .extraction import Extractor
from .forgetting import ForgetPolicy
from .identity import IdentityResolver
from .llm import LLMClient, build_llm
from .memory import MemoryService
from .pipeline import Pipeline, Worker
from .queue import EventQueue, build_queue
from .schemas import utcnow
from .store import Store
from .telemetry import Telemetry
from .vector import VectorIndex


@dataclass
class Services:
    settings: Settings
    telemetry: Telemetry
    store: Store
    index: VectorIndex
    embedder: Embedder
    policy: ForgetPolicy
    identity: IdentityResolver
    extractor: Extractor
    memory: MemoryService
    pipeline: Pipeline
    queue: EventQueue
    worker: Worker

    def close(self) -> None:
        self.worker.stop()
        self.queue.close()
        self.store.close()


def build_services(settings: Settings | None = None, *, llm: LLMClient | None = None,
                   queue: EventQueue | None = None, clock: Callable[[], datetime] = utcnow) -> Services:
    s = settings or Settings()
    tel = Telemetry(s)
    store = Store(s.db_path)
    embedder = build_embedder(s)
    index = VectorIndex(embedder.dim, s.vector_backend)
    policy = ForgetPolicy(
        min_confidence=s.min_confidence, default_country_code=s.default_country_code,
        retention_days=s.retention_days, resolved_issue_ttl_days=s.resolved_issue_ttl_days,
    )
    identity = IdentityResolver(store, s.default_country_code)
    extractor = Extractor(llm or build_llm(s), policy, s.prompt_version, s.extraction_attempts)
    memory = MemoryService(store, index, embedder, policy, tel, s, clock)
    memory.rebuild_index()
    pipeline = Pipeline(store, identity, extractor, memory, tel)
    q = queue or build_queue(s)
    return Services(s, tel, store, index, embedder, policy, identity, extractor, memory, pipeline, q,
                    Worker(q, pipeline, tel))
