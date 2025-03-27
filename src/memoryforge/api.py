"""FastAPI application."""

from __future__ import annotations

import time
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from typing import Annotated

import structlog
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from pydantic import BaseModel

from . import __version__
from .config import Settings
from .schemas import ConversationEvent, RetrievalResponse
from .services import Services, build_services
from .telemetry import configure_logging

log = structlog.get_logger()


class ReplayRequest(BaseModel):
    from_offset: int = 0
    rebuild: bool = False


class MergeRequest(BaseModel):
    source: str
    target: str


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    settings = services.settings if services else (settings or Settings())
    configure_logging(settings.log_level, settings.log_format)
    svc = services or build_services(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if settings.run_worker:
            svc.worker.start()
        log.info("memoryforge_started", version=__version__, vector_backend=svc.index.backend,
                 embedder=svc.embedder.name, llm=svc.extractor.llm.name, prompt_version=svc.extractor.prompt_version)
        yield
        svc.close()

    app = FastAPI(title="MemoryForge", version=__version__, lifespan=lifespan)
    app.state.services = svc

    @app.middleware("http")
    async def correlation(request: Request, call_next):
        cid = request.headers.get("x-correlation-id") or uuid.uuid4().hex
        with structlog.contextvars.bound_contextvars(correlation_id=cid):
            request.state.correlation_id = cid
            response = await call_next(request)
        response.headers["X-Correlation-ID"] = cid
        return response

    def admin(x_admin_token: Annotated[str | None, Header()] = None) -> None:
        if settings.admin_token and x_admin_token != settings.admin_token:
            raise HTTPException(401, "invalid or missing X-Admin-Token")

    def customer_or_404(ref: str) -> str:
        cid = svc.identity.lookup(ref)
        if cid is None:
            raise HTTPException(404, "unknown customer")
        return cid

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "queue_lag": svc.queue.lag(), "index_vectors": svc.index.size(),
                **svc.store.stats()}

    @app.post("/events", status_code=202)
    def ingest(request: Request, body: ConversationEvent | list[ConversationEvent],
               wait: bool = Query(False, description="process synchronously before returning")):
        events = body if isinstance(body, list) else [body]
        offsets = []
        for e in events:
            if not e.correlation_id:  # tie async processing back to this request
                e = e.model_copy(update={"correlation_id": request.state.correlation_id})
            offsets.append(svc.queue.publish(e))
        outcomes = None
        if wait:
            svc.worker.drain()  # the background worker may have taken some; wait for the rest below
            deadline = time.monotonic() + 10
            statuses = {}
            for e in events:
                while (st := svc.store.event_status(e.event_id)) is None and time.monotonic() < deadline:
                    time.sleep(0.01)
                statuses[e.event_id] = st or "pending"
            outcomes = dict(Counter(statuses.values()))
        return {"accepted": len(events), "offsets": offsets, "outcomes": outcomes}

    @app.get("/memory/{customer_ref}", response_model=RetrievalResponse)
    def get_memory(customer_ref: str, q: str = Query("", max_length=500), k: int = Query(10, ge=1, le=100),
                   include_history: bool = False):
        """`customer_ref` is a customer id, or any known email / phone / user-id variant."""
        cid = customer_or_404(customer_ref)
        return svc.memory.retrieve(cid, q, k, include_history)

    @app.delete("/memory/{customer_ref}", dependencies=[Depends(admin)])
    def forget(customer_ref: str):
        cid = customer_or_404(customer_ref)
        return {"customer_id": cid, "memories_deleted": svc.memory.forget_customer(cid)}

    @app.get("/customers/resolve")
    def resolve(user_id: str | None = None, email: str | None = None, phone: str | None = None):
        refs = [r for r in (user_id, email, phone) if r]
        if not refs:
            raise HTTPException(422, "provide user_id, email or phone")
        found = {svc.identity.lookup(r) for r in refs}
        if len(found) > 1 and None not in found:
            raise HTTPException(409, "identifiers belong to different customers")
        cid = next((c for c in found if c), None)
        return {"customer_id": cid, "identifiers": svc.store.identifiers_of(cid) if cid else []}

    @app.post("/admin/merge", dependencies=[Depends(admin)])
    def merge(req: MergeRequest):
        src, dst = customer_or_404(req.source), customer_or_404(req.target)
        try:
            moved = svc.store.merge_customers(src, dst)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        svc.memory.rebuild_index()
        return {"merged": src, "into": dst, "memories_moved": len(moved)}

    @app.post("/admin/replay", dependencies=[Depends(admin)])
    def replay(req: ReplayRequest):
        return {"outcomes": dict(svc.worker.replay(req.from_offset, req.rebuild))}

    @app.post("/admin/sweep", dependencies=[Depends(admin)])
    def sweep():
        return {"expired_removed": svc.memory.sweep_expired()}

    @app.get("/failures", dependencies=[Depends(admin)])
    def failures(limit: int = Query(100, ge=1, le=1000)):
        return svc.store.list_dead_letters(limit)

    @app.get("/failures/{event_id}", dependencies=[Depends(admin)])
    def failure_detail(event_id: str):
        """Dead-letter record + the trace spans captured when the pipeline failed."""
        rec = svc.store.get_dead_letter(event_id)
        if rec is None:
            raise HTTPException(404, "no failure recorded for that event")
        return rec

    @app.get("/metrics")
    def metrics():
        return Response(svc.telemetry.metrics.render(), media_type="text/plain; version=0.0.4")

    return app


def app_factory() -> FastAPI:  # `uvicorn memoryforge.api:app_factory --factory`
    return create_app()
