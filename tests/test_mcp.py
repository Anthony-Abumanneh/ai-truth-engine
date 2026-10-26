"""Step 9.5: MCP server tools, called through a real MCP client (in-process)."""
import json

import anyio
import pytest
from mcp.client.client import Client

from truth_engine import mcp_server
from truth_engine.feedback.traces import TraceStore
from truth_engine.pipeline.orchestrator import Orchestrator
from truth_engine.retrieval import EvidenceEngine


@pytest.fixture(scope="module", autouse=True)
def offline_orchestrator(tmp_path_factory):
    store = TraceStore(tmp_path_factory.mktemp("db") / "t.sqlite")
    engine = EvidenceEngine()
    mcp_server._orch.update({"engine": engine, False: Orchestrator(live=False, engine=engine, store=store)})
    yield
    mcp_server._orch.clear()


def call(name, args):
    async def go():
        async with Client(mcp_server.server) as c:
            tools = {t.name for t in (await c.list_tools()).tools}
            res = await c.call_tool(name, args)
            data = getattr(res, "structured_content", None) or json.loads(res.content[0].text)
            return tools, data
    return anyio.run(go)


def test_tools_listed_and_audit_works():
    tools, data = call("audit_text", {"text": "Veridamine improves sleep quality by 18%. I love it!"})
    assert tools == {"audit_text", "ask_question", "watch_claim", "recheck_watched_claims"}
    c = data["claims"][0]
    assert c["verdict"] == "contradicted" and c["evidence_against"] and c["red_flags"]
    assert data["not_checked"] == ["I love it!"]


def test_ask_and_watch():
    _, a = call("ask_question", {"question": "Who discovered penicillin?"})
    assert a["answer"] == "Alexander Fleming" and a["confidence"] > 0.5
    _, w = call("watch_claim", {"claim": "Python was first released in 1991."})
    assert w["watching"]
    _, r = call("recheck_watched_claims", {})
    assert r == {"changes": []}
