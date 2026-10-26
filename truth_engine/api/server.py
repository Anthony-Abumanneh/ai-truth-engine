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
"""
from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..config import settings
from ..feedback.traces import TraceStore
from ..pipeline.orchestrator import Orchestrator
from ..retrieval import EvidenceEngine
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
    app = FastAPI(title="AI Truth Engine", version="0.9")
    state = {"engine": engine, "store": store, "ready": False}
    lock = threading.Lock()

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
        return {"ready": state["ready"], "nli": nli.name if nli else None,
                "live_sources": {"pubmed": True, "openalex": True, "web": settings.has_web},
                "max_text": MAX_TEXT}

    @app.post("/api/audit")
    def audit(body: AuditIn):
        return trace_view(orch(body.live).audit(body.text))

    @app.post("/api/audit/stream")
    def audit_stream(body: AuditIn):
        q: queue.Queue = queue.Queue()

        def work():
            try:
                t = orch(body.live).audit(body.text, on_event=lambda ev: q.put(("progress", {
                    "stage": ev.stage, "status": ev.status, "message": ev.message})))
                q.put(("result", trace_view(t)))
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
        return trace_view(orch(body.live).ask(body.question))

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
        return {"changes": orch(live).recheck_watched()}

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")

    return app


def serve(port: int = 8000, host: str = "127.0.0.1") -> None:
    import uvicorn
    uvicorn.run(create_app(), host=host, port=port, log_level="info")
