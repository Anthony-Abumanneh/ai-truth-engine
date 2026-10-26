"""Compact JSON views of a Trace for the web UI and API clients.

A full Trace carries every retrieved passage and source; the UI needs the
verdict, why, the strongest evidence on each side, red flags and where each
claim sits in the input text (for highlighting).
"""
from __future__ import annotations

from ..schemas import ClaimVerdict, EvidenceItem, Trace

MAX_EVIDENCE = 5


def _evidence(items: list[EvidenceItem]) -> list[dict]:
    out, seen = [], set()
    for i in sorted(items, key=lambda i: -i.weight):
        if i.source.id in seen:
            continue                       # one entry per source: the strongest sentence
        seen.add(i.source.id)
        q = i.source.quality
        out.append({"source_id": i.source.id, "title": i.source.title, "url": i.source.url,
                    "publisher": i.source.publisher, "published": i.source.published,
                    "type": i.source.source_type, "tier": q.tier if q else None,
                    "study": (i.source.metadata or {}).get("study_label"),
                    "text": i.passage.text, "stance": i.stance.value, "weight": round(i.weight, 3),
                    "lineage": i.lineage})
        if len(out) >= MAX_EVIDENCE:
            break
    return out


def claim_view(claim, v: ClaimVerdict, route: dict | None) -> dict:
    cd = v.confidence_detail
    passes = (route or {}).get("passes", [])
    return {
        "id": claim.id, "text": claim.text, "sentence": claim.sentence, "span": list(claim.span) if claim.span else None,
        "status": v.status.value, "decision": v.decision.value if v.decision else None,
        "leaning": v.leaning.value if v.leaning else None, "conflicting": v.conflicting,
        "confidence": round(cd.calibrated, 3) if cd else round(v.confidence, 3),
        "confidence_label": cd.label if cd else None, "rationale": v.rationale,
        "supporting_sources": v.supporting_source_count, "independent_lines": v.independent_support_count,
        "supporting": _evidence(v.supporting), "contradicting": _evidence(v.contradicting),
        "red_flags": [{"kind": f.kind, "severity": f.severity, "message": f.message} for f in v.red_flags],
        "route": [{"tier": p.get("tier"), "why_next": p.get("escalated_because")} for p in passes],
    }


def trace_view(t: Trace) -> dict:
    by_id = {v.claim_id: v for v in t.verdicts}
    claims = [claim_view(c, by_id[c.id], t.routing.get(c.id)) for c in t.claims if c.id in by_id]
    counts: dict[str, int] = {}
    for c in claims:
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    out = {"query_id": t.query_id, "mode": t.mode, "input": t.question, "created_at": t.created_at,
           "latency_s": t.latency_s, "claims": claims, "counts": counts, "skipped": t.skipped,
           "verifiers": t.verifiers, "errors": t.errors}
    if t.final is not None:
        out["answer"] = {"decision": t.final.decision.value, "summary": t.final.summary, "body": t.final.body,
                         "confidence": round(t.final.confidence.calibrated, 3),
                         "confidence_label": t.final.confidence.label, "caveats": t.final.caveats,
                         "uncertainty": t.final.uncertainty}
    if t.candidates:
        out["candidates"] = [{"model": c.model, "answer": c.final_value or c.answer} for c in t.candidates]
    return out
