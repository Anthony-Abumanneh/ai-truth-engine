"""MCP server: lets Claude and other MCP clients fact-check text with the engine (step 9.5).

    python -m truth_engine.mcp_server                 # stdio transport

Register with Claude Code:
    claude mcp add truth-engine -- /path/to/ai-truth-engine/.venv/bin/python -m truth_engine.mcp_server

Tools return compact JSON so the calling model spends few tokens: per claim the
verdict, calibrated confidence, the strongest evidence for and against (title,
link, quote), and red flags. Models load on the first call (~20-60 s), then stay
in memory. Offline by default (local corpus); live=True searches PubMed,
OpenAlex and the web (slower).
"""
from __future__ import annotations

import threading

from mcp.server.mcpserver import MCPServer

from .api.views import trace_view

INSTRUCTIONS = """Evidence auditor for scientific and health claims.
Use audit_text on any text (including your own draft answer) to check each factual claim against
the research literature; use ask_question for a direct evidence-based answer. Report verdicts
faithfully: 'unverified' means the evidence is missing or conflicting, not that the claim is false.
Cite the returned sources. Not medical advice."""

server = MCPServer(name="truth-engine", title="AI Truth Engine", instructions=INSTRUCTIONS, version="0.9.5")
_lock = threading.Lock()
_orch: dict = {}


def _orchestrator(live: bool):
    from .pipeline.orchestrator import Orchestrator
    from .retrieval import EvidenceEngine
    with _lock:
        if "engine" not in _orch:
            _orch["engine"] = EvidenceEngine()
        if live not in _orch:
            _orch[live] = Orchestrator(live=live, engine=_orch["engine"])
        return _orch[live]


def _compact_claim(c: dict) -> dict:
    ev = lambda items: [{k: e[k] for k in ("title", "url", "study", "text") if e.get(k)} for e in items[:2]]  # noqa: E731
    out = {"claim": c["text"], "verdict": c["status"], "confidence": c["confidence"],
           "confidence_label": c["confidence_label"], "why": c["rationale"][:400],
           "evidence_for": ev(c["supporting"]), "evidence_against": ev(c["contradicting"])}
    if c["leaning"]:
        out["leans"] = c["leaning"]
    if c["red_flags"]:
        out["red_flags"] = [f["message"] for f in c["red_flags"]]
    if c["supporting_sources"]:
        out["independent_lines_of_support"] = f"{c['independent_lines']} of {c['supporting_sources']} sources"
    return out


@server.tool(description="Check every factual claim in a text against the research literature. Returns one "
                         "verdict per claim (supported / partially_supported / contradicted / unverified) with "
                         "calibrated confidence, evidence for and against, and red flags. Set live=true to "
                         "search PubMed, OpenAlex and the web (slower, ~10-60 s); otherwise a local corpus.")
def audit_text(text: str, live: bool = False) -> dict:
    t = trace_view(_orchestrator(live).audit(text[:6000]))
    return {"claims": [_compact_claim(c) for c in t["claims"]], "summary": t["counts"],
            "not_checked": [s["text"] for s in t["skipped"]], "run_id": t["query_id"],
            "note": "Not medical advice. 'unverified' = evidence missing or conflicting."}


@server.tool(description="Answer a factual or yes/no science/health question from verified evidence, or say "
                         "the evidence is not good enough. Set live=true to search the live literature.")
def ask_question(question: str, live: bool = False) -> dict:
    t = trace_view(_orchestrator(live).ask(question[:500]))
    a = t.get("answer", {})
    return {"answer": a.get("summary"), "decision": a.get("decision"), "confidence": a.get("confidence"),
            "confidence_label": a.get("confidence_label"), "caveats": a.get("caveats", []),
            "evidence": [_compact_claim(c) for c in t["claims"]][:3], "run_id": t["query_id"]}


@server.tool(description="Save a claim so the engine can re-check it later (e.g. when new studies appear).")
def watch_claim(claim: str) -> dict:
    return {"watching": _orchestrator(False).watch(claim[:500])}


@server.tool(description="Re-check all saved claims and report any whose verdict or confidence changed.")
def recheck_watched_claims(live: bool = False) -> dict:
    return {"changes": _orchestrator(live).recheck_watched()}


def main() -> None:
    import logging
    import os
    os.environ.setdefault("TQDM_DISABLE", "1")             # no progress bars on stderr
    for name in ("httpx", "sentence_transformers", "huggingface_hub", "transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)
    server.run("stdio")


if __name__ == "__main__":
    main()
