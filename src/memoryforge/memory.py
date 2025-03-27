"""MemoryService: transactional write path (supersession + index sync) and ranked retrieval."""

from __future__ import annotations

import math
import time
from datetime import datetime
from typing import Callable

import numpy as np
import structlog

from .config import Settings
from .embeddings import Embedder
from .forgetting import ForgetPolicy
from .schemas import (
    ConversationEvent,
    ExtractedFact,
    RetrievalResponse,
    ScoreBreakdown,
    ScoredMemory,
    fact_text,
    utcnow,
)
from .store import Store
from .telemetry import Telemetry
from .vector import VectorIndex

log = structlog.get_logger()


class MemoryService:
    def __init__(self, store: Store, index: VectorIndex, embedder: Embedder, policy: ForgetPolicy,
                 telemetry: Telemetry, settings: Settings, clock: Callable[[], datetime] = utcnow):
        self.store, self.index, self.embedder = store, index, embedder
        self.policy, self.tel, self.settings, self.clock = policy, telemetry, settings, clock
        self._query_cache: dict[str, np.ndarray] = {}

    # ---- write path -----------------------------------------------------------------

    def embed_facts(self, facts: list[ExtractedFact]) -> np.ndarray:
        texts = [fact_text(f.kind, f.key, f.value, f.status) for f in facts]
        return self.embedder.embed(texts) if texts else np.zeros((0, self.embedder.dim), dtype=np.float32)

    def commit(self, event: ConversationEvent, customer_id: str, facts: list[ExtractedFact],
               vectors: np.ndarray, *, prompt_version: str, extractor: str, offset: int | None,
               outcome: str = "processed") -> dict[str, int]:
        """Atomically store facts, supersede older ones and mark the event processed.

        Index updates happen after the DB commit; if the process dies in between, the index
        is rebuilt from the store on startup, so the store is always the source of truth.
        """
        stored, stale = 0, 0
        added: list[tuple[int, np.ndarray]] = []
        removed: list[int] = []
        with self.store.transaction():
            for fact, vec in zip(facts, vectors):
                existing = self.store.active_in_slot(customer_id, fact.kind, fact.key)
                is_stale, to_supersede = self.policy.resolve_supersession(fact.observed_at, existing)
                if is_stale:
                    stale += 1
                    continue
                mem_id = self.store.insert_memory(
                    {
                        "customer_id": customer_id, "kind": fact.kind, "key": fact.key, "value": fact.value,
                        "confidence": fact.confidence, "observed_at": fact.observed_at, "status": fact.status,
                        "evidence": fact.evidence, "source_event_id": event.event_id,
                        "conversation_id": event.conversation_id, "prompt_version": prompt_version,
                        "extractor": extractor,
                    },
                    self.embedder.name, vec,
                )
                if to_supersede:
                    self.store.supersede(to_supersede, mem_id)
                    removed.extend(to_supersede)
                added.append((mem_id, vec))
                stored += 1
            self.store.mark_processed(event.event_id, outcome, offset)
        if removed:
            self.index.remove(customer_id, removed)
        for mem_id, vec in added:
            self.index.add(customer_id, mem_id, vec)
        for _ in range(stale):
            self.tel.metrics.facts_total.labels("forgotten:superseded").inc()
        if stored:
            self.tel.metrics.facts_total.labels("stored").inc(stored)
        return {"stored": stored, "superseded_on_arrival": stale, "replaced": len(removed)}

    # ---- lifecycle ------------------------------------------------------------------

    def rebuild_index(self) -> int:
        """Reload every active vector from SQLite; re-embed rows made by a different embedder."""
        self.index.clear()
        by_customer: dict[str, tuple[list[int], list[np.ndarray]]] = {}
        for r in self.store.active_embeddings():
            blob = np.frombuffer(r["embedding"], dtype=np.float32)
            if r["embedder"] != self.embedder.name or blob.shape[0] != self.embedder.dim:
                status = r["status"]
                blob = self.embedder.embed([fact_text(r["kind"], r["key"], r["value"], status)])[0]
                self.store.update_embedding(r["id"], self.embedder.name, blob)
            ids, vecs = by_customer.setdefault(r["customer_id"], ([], []))
            ids.append(r["id"])
            vecs.append(blob)
        for cid, (ids, vecs) in by_customer.items():
            self.index.add_many(cid, ids, np.vstack(vecs))
        return self.index.size()

    def forget_customer(self, customer_id: str) -> int:
        ids = self.store.delete_customer_data(customer_id)
        self.index.drop_customer(customer_id)
        log.info("customer_forgotten", customer_id=customer_id, memories=len(ids))
        return len(ids)

    def sweep_expired(self) -> int:
        resolved_cut, any_cut = self.policy.expiry_cutoffs(self.clock())
        removed = self.store.delete_expired(resolved_cut, any_cut)
        per_customer: dict[str, list[int]] = {}
        for mem_id, cid in removed:
            per_customer.setdefault(cid, []).append(mem_id)
        for cid, ids in per_customer.items():
            self.index.remove(cid, ids)
        if removed:
            self.tel.metrics.facts_total.labels("forgotten:expired").inc(len(removed))
        return len(removed)

    # ---- retrieval ------------------------------------------------------------------

    def _embed_query(self, q: str) -> np.ndarray:
        vec = self._query_cache.get(q)
        if vec is None:
            vec = self.embedder.embed([q])[0]
            if len(self._query_cache) > 2048:
                self._query_cache.clear()
            self._query_cache[q] = vec
        return vec

    def retrieve(self, customer_id: str, query: str = "", k: int = 10, include_history: bool = False) -> RetrievalResponse:
        t0 = time.perf_counter()
        s = self.settings
        now = self.clock()
        with self.tel.stage("retrieve", customer_id=customer_id):
            sims: dict[int, float] = {}
            if query.strip() and not include_history:
                with self.tel.stage("retrieve_embed"):
                    qvec = self._embed_query(query.strip())
                with self.tel.stage("retrieve_search"):
                    sims = dict(self.index.search(customer_id, qvec, max(k * 4, 50)))
                with self.tel.stage("retrieve_hydrate"):
                    mems = self.store.get_memories(customer_id, list(sims))
                    records = [m for m in mems.values() if m.active]
            else:
                with self.tel.stage("retrieve_hydrate"):
                    records = self.store.list_memories(customer_id, include_history)
                if query.strip():  # history search: score by embedding on the fly
                    qvec = self._embed_query(query.strip())
                    vecs = self.embedder.embed([fact_text(m.kind, m.key, m.value, m.status) for m in records])
                    sims = {m.id: float(v @ qvec) for m, v in zip(records, vecs)}
            with self.tel.stage("retrieve_rank"):
                has_q = bool(query.strip())
                w_sim = s.weight_similarity if has_q else 0.0
                total_w = w_sim + s.weight_recency + s.weight_confidence
                scored = []
                for m in records:
                    age_days = max(0.0, (now - m.observed_at).total_seconds() / 86400)
                    recency = math.exp(-math.log(2) * age_days / s.recency_half_life_days)
                    sim = max(0.0, sims.get(m.id, 0.0)) if has_q else 0.0
                    score = (w_sim * sim + s.weight_recency * recency + s.weight_confidence * m.confidence) / total_w
                    scored.append(ScoredMemory(
                        memory=m, score=round(score, 5),
                        breakdown=ScoreBreakdown(similarity=round(sim, 5), recency=round(recency, 5),
                                                 confidence=m.confidence),
                    ))
                scored.sort(key=lambda x: (-x.score, -x.memory.id))
        self.tel.metrics.retrievals_total.labels("ok").inc()
        return RetrievalResponse(customer_id=customer_id, query=query, took_ms=round((time.perf_counter() - t0) * 1000, 3),
                                 results=scored[:k])
