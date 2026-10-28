"""Step 9: HTTP API and web page (offline engine, temporary database)."""
import json

import pytest
from fastapi.testclient import TestClient

from truth_engine.api.server import create_app
from truth_engine.feedback.traces import TraceStore
from truth_engine.retrieval import EvidenceEngine


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    store = TraceStore(tmp_path_factory.mktemp("db") / "t.sqlite")
    return TestClient(create_app(engine=EvidenceEngine(), store=store, warm=False))


def test_health_and_page(client):
    h = client.get("/api/health").json()
    assert h["ready"] and "pubmed" in h["live_sources"]
    page = client.get("/")
    assert page.status_code == 200 and "AI Truth Engine" in page.text


def test_audit_returns_claims_with_spans_and_evidence(client):
    text = "Veridamine improves sleep quality by 18%. I love it!"
    r = client.post("/api/audit", json={"text": text}).json()
    c = r["claims"][0]
    assert c["status"] == "contradicted" and c["contradicting"] and c["red_flags"]
    s, e = c["span"]
    assert text[s:e].startswith("Veridamine")                    # the page highlights this range
    assert r["skipped"] and r["counts"] == {"contradicted": 1}


def test_stream_sends_progress_then_result(client):
    with client.stream("POST", "/api/audit/stream", json={"text": "Python was first released in 1991."}) as r:
        body = "".join(r.iter_text())
    events = [blk.split("\n")[0].removeprefix("event: ") for blk in body.strip().split("\n\n")]
    assert events[0] == "progress" and events[-1] == "result"
    result = json.loads(body.strip().split("\n\n")[-1].split("data: ", 1)[1])
    assert result["claims"][0]["status"] == "supported"


def test_ask_history_and_feedback(client):
    r = client.post("/api/ask", json={"question": "Who discovered penicillin?"}).json()
    assert r["answer"]["summary"] == "Alexander Fleming" and r["candidates"]
    runs = client.get("/api/traces").json()
    assert any(t["query_id"] == r["query_id"] for t in runs)
    again = client.get(f"/api/traces/{r['query_id']}").json()
    assert again["answer"]["summary"] == "Alexander Fleming"
    ok = client.post("/api/feedback", json={"query_id": r["query_id"], "claim_id": r["claims"][0]["id"],
                                             "label": "supported"})
    assert ok.status_code == 200
    bad = client.post("/api/feedback", json={"query_id": r["query_id"], "claim_id": r["claims"][0]["id"],
                                              "label": "definitely"})
    assert bad.status_code == 400
    assert client.get("/api/traces/nope").status_code == 404


def test_watch_endpoints(client):
    client.post("/api/watch", json={"text": "Python was first released in 1991."})
    assert client.get("/api/watch").json()[0]["text"].startswith("Python")
    assert client.post("/api/watch/recheck").json() == {"changes": []}
    assert client.get("/api/watch").json()[0]["last_status"] == "supported"
    assert client.request("DELETE", "/api/watch", json={"text": "Python was first released in 1991."}).json()["removed"]


def test_input_limits(client):
    assert client.post("/api/audit", json={"text": ""}).status_code == 422
    assert client.post("/api/audit", json={"text": "x" * 7000}).status_code == 422


def test_stats_metrics_and_versions(client):
    client.post("/api/audit", json={"text": "Python was first released in 1991."})
    s = client.get("/api/stats").json()
    assert s["claims_checked"] >= 1 and "drift_vs_benchmark" in s and "feedback" in s
    m = client.get("/metrics").text
    assert "truth_engine_requests_total" in m and 'verdict="supported"' in m
    assert "models" in client.get("/api/health").json()


def test_rate_limit_and_live_gate(monkeypatch, tmp_path):
    from truth_engine.api.monitoring import RateLimiter
    rl = RateLimiter(2)
    assert rl.allow("a") and rl.allow("a") and not rl.allow("a") and rl.allow("b")
    monkeypatch.setenv("TE_ALLOW_LIVE", "0")
    monkeypatch.setenv("TE_RATE_LIMIT", "1")
    c = TestClient(create_app(engine=EvidenceEngine(), store=TraceStore(tmp_path / "x.sqlite"), warm=False))
    assert c.get("/api/health").json()["allow_live"] is False
    assert c.post("/api/audit", json={"text": "Python was first released in 1991.", "live": True}).status_code == 403
    assert c.post("/api/audit", json={"text": "Python was first released in 1991."}).status_code == 429   # bucket used
