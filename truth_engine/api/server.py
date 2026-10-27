"""HTTP API + web UI (step 9).

    python -m truth_engine serve [--port 8000]       # then open http://localhost:8000

Endpoints (JSON):
    GET  /api/health                     models loaded, which live sources are available
    POST /api/audit         {text, live}           -> audit result
    POST /api/audit/stream  {text, live}           -> server-sent events: progress, then the result
    POST /api/ask           {question, live}       -> answer
    GET  /api/traces                               recent runs
    GET  /api/traces/{id}                          one run
    POST /api/feedback      {query_id, claim_id, label, note}
    GET  /api/watch | POST /api/watch {text} | DELETE /api/watch {text} | POST /api/watch/recheck
    GET  /api/stats                      usage, verdict mix, drift vs benchmark, feedback accuracy
    GET  /metrics                        Prometheus metrics

Deployment settings (environment):
    TE_ALLOW_LIVE=0      refuse live literature/web searches (public demo: protects API quotas)
    TE_RATE_LIMIT=20     per-client requests per minute on audit/ask (0 = off)
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..config import settings
from ..feedback.traces import TraceStore
from ..pipeline.orchestrator import Orchestrator
from ..retrieval import EvidenceEngine
from .monitoring import EXPENSIVE, Metrics, RateLimiter, client_id, log_request
from .views import trace_view

WEB = Path(__file__).resolve().parent.parent.parent / "web"
MAX_TEXT = 6000           # characters per audit: bounds latency and cost


class AuditIn(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT)
    live: bool = False


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    live: bool = False


class FeedbackIn(BaseModel):
    query_id: str
    claim_id: str
    label: str
    note: str = ""


class WatchIn(BaseModel):
    text: str = Field(min_length=3, max_length=500)


def create_app(*, engine: EvidenceEngine | None = None, store: TraceStore | None = None,
               warm: bool = True) -> FastAPI:
    app = FastAPI(title="AI Truth Engine", version="0.11")
    state = {"engine": engine, "store": store, "ready": False}
    lock = threading.Lock()
    metrics = Metrics()
    limiter = RateLimiter(int(os.getenv("TE_RATE_LIMIT", "0")))
    allow_live = os.getenv("TE_ALLOW_LIVE", "1") != "0"
    app.state.metrics = metrics

    @app.middleware("http")
    async def observe(request: Request, call_next):
        t0, client = time.time(), client_id(request.client.host if request.client else None)
        if request.url.path in EXPENSIVE and not limiter.allow(client):
            metrics.rate_limited += 1
            resp = JSONResponse({"detail": "Too many requests: please wait a minute."}, status_code=429)
        else:
            resp = await call_next(request)
        dt = time.time() - t0
        if request.url.path.startswith("/api"):
            metrics.request(request.url.path, resp.status_code, dt)
            log_request(request.url.path, resp.status_code, dt, client)
        return resp

    def check_live(live: bool) -> None:
        if live and not allow_live:
            raise HTTPException(403, "Live literature search is disabled on this server.")

    def _engine() -> EvidenceEngine:
        with lock:
            if state["engine"] is None:
                state["engine"] = EvidenceEngine()
            if state["store"] is None:
                state["store"] = TraceStore()
            return state["engine"]

    def orch(live: bool) -> Orchestrator:
        return Orchestrator(live=live, engine=_engine(), store=state["store"])

    def warm_up():
        """Load the search index and models before the first request."""
        from ..pipeline.nli import get_nli
        from ..pipeline.reranker import get_reranker
        _engine().store.search("warm up", k=1)
        get_nli()
        get_reranker()
        state["ready"] = True

    if warm:
        threading.Thread(target=warm_up, daemon=True).start()
    else:
        state["ready"] = True

    @app.get("/api/health")
    def health():
        from ..pipeline.nli import get_nli
        nli = get_nli() if state["ready"] else None
        from ..models_registry import summary
        return {"ready": state["ready"], "nli": nli.name if nli else None, "allow_live": allow_live,
                "live_sources": {"pubmed": allow_live, "openalex": allow_live, "web": allow_live and settings.has_web},
                "max_text": MAX_TEXT, "models": summary()}

    @app.post("/api/audit")
    def audit(body: AuditIn):
        check_live(body.live)
        view = trace_view(orch(body.live).audit(body.text))
        metrics.result(view, body.live)
        return view

    @app.post("/api/audit/stream")
    def audit_stream(body: AuditIn):
        check_live(body.live)
        q: queue.Queue = queue.Queue()

        def work():
            try:
                t = orch(body.live).audit(body.text, on_event=lambda ev: q.put(("progress", {
                    "stage": ev.stage, "status": ev.status, "message": ev.message})))
                view = trace_view(t)
                metrics.result(view, body.live)
                q.put(("result", view))
            except Exception as e:                      # surface failures to the page, not a hung stream
                q.put(("error", {"message": f"{type(e).__name__}: {e}"}))
            q.put(None)

        threading.Thread(target=work, daemon=True).start()

        def events():
            while (item := q.get()) is not None:
                kind, data = item
                yield f"event: {kind}\ndata: {json.dumps(data)}\n\n"
        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/ask")
    def ask(body: AskIn):
        check_live(body.live)
        view = trace_view(orch(body.live).ask(body.question))
        metrics.result(view, body.live)
        return view

    @app.get("/api/traces")
    def traces(limit: int = 30):
        _engine()
        return state["store"].list_traces(limit)

    @app.get("/api/traces/{query_id}")
    def trace(query_id: str):
        _engine()
        t = state["store"].get_trace(query_id)
        if t is None:
            raise HTTPException(404, "no such run")
        return trace_view(t)

    @app.post("/api/feedback")
    def feedback(body: FeedbackIn):
        _engine()
        try:
            return {"id": state["store"].add_feedback(body.query_id, body.claim_id, body.label, body.note)}
        except (KeyError, ValueError) as e:
            raise HTTPException(400, str(e))

    @app.get("/api/watch")
    def watched():
        _engine()
        return state["store"].watched()

    @app.post("/api/watch")
    def watch(body: WatchIn):
        _engine()
        return {"key": state["store"].watch(body.text)}

    @app.delete("/api/watch")
    def unwatch(body: WatchIn):
        _engine()
        return {"removed": state["store"].unwatch(body.text)}

    @app.post("/api/watch/recheck")
    def recheck(live: bool = False):
        check_live(live)
        return {"changes": orch(live).recheck_watched()}

    @app.get("/api/stats")
    def stats():
        _engine()
        fb = state["store"].feedback()
        from ..feedback.learning import same_verdict
        rated = [f for f in fb if f["predicted"]]
        return metrics.stats() | {"feedback": {
            "count": len(fb),
            "agreement": round(sum(same_verdict(f["predicted"], f["label"]) for f in rated) / len(rated), 3)
            if rated else None}}

    @app.get("/metrics")
    def prometheus():
        return PlainTextResponse(metrics.prometheus(), media_type="text/plain; version=0.0.4")

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")

    return app


def serve(port: int = 8000, host: str = "127.0.0.1") -> None:
    import uvicorn
    uvicorn.run(create_app(), host=host, port=port, log_level="info")
