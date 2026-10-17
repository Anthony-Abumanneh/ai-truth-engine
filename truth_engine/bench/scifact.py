"""SciFact end-to-end harness (CONCEPTS.md §4, §10).

SciFact: 1,409 expert-written scientific claims, each labeled SUPPORT,
CONTRADICT or NOT ENOUGH INFO against a corpus of 5,183 abstracts. Here the
whole engine runs open-domain: it searches the full corpus itself (it is not
told which abstract is relevant), then verifies, critiques and judges.

Data: data/scifact/data/ (3.1 MB; see README). Records are cached as JSONL
under data/bench/ so analysis (calibration, ablations) needs no reruns.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from ..config import DATA_DIR
from ..retrieval import DocumentStore, EvidenceEngine
from ..retrieval.knowledge_graph import KnowledgeGraph
from ..schemas import EvidenceStatus
from ..pipeline.audit import ALL_COMPONENTS, _Run, check_claim
from ..pipeline.claims import make_claim
from ..pipeline.router import Router

SCIFACT = DATA_DIR / "scifact" / "data"
BENCH_DIR = DATA_DIR / "bench"
LABEL = {EvidenceStatus.SUPPORTED: "SUPPORT", EvidenceStatus.PARTIALLY_SUPPORTED: "SUPPORT",
         EvidenceStatus.INFERRED: "SUPPORT", EvidenceStatus.CONTRADICTED: "CONTRADICT",
         EvidenceStatus.UNVERIFIED: "NEI", EvidenceStatus.SPECULATIVE: "NEI"}


def load_claims(split: str) -> list[dict]:
    out = []
    for c in map(json.loads, open(SCIFACT / f"claims_{split}.jsonl")):
        labels = Counter(s["label"] for sets in c.get("evidence", {}).values() for s in sets)
        out.append({"id": c["id"], "claim": c["claim"], "gold": labels.most_common(1)[0][0] if labels else "NEI",
                    "evidence_docs": [f"sf-{d}" for d in c.get("evidence", {})],
                    "cited_docs": [f"sf-{d}" for d in c.get("cited_doc_ids", [])]})
    return out


def build_engine() -> EvidenceEngine:
    """An evidence engine whose 'corpus' is the 5,183 SciFact abstracts (no graph, no web)."""
    store = DocumentStore()          # embedder: TE_EMBEDDINGS / TE_EMBEDDINGS_MODEL (default bge-base)
    for d in map(json.loads, open(SCIFACT / "corpus.jsonl")):
        store.add_document(" ".join(d["abstract"]), id=f"sf-{d['doc_id']}", title=d["title"], type="paper",
                           authority=0.75, primary="true")
    eng = EvidenceEngine(store=store, graph=KnowledgeGraph())
    eng.web.client = None            # never touch the web in benchmark runs
    return eng


def run(split: str, *, components=ALL_COMPONENTS, calibrator=None, limit: int | None = None, ids=None,
        engine: EvidenceEngine | None = None, tag: str = "", max_tier=None, progress: bool = True) -> list[dict]:
    """Audit each claim (as one claim, no splitting) and record the outcome."""
    claims = load_claims(split)
    if ids is not None:
        claims = [c for c in claims if c["id"] in ids]
    claims = claims[:limit]
    engine = engine or build_engine()
    run_ = _Run(engine, None, components)
    router = Router(live=False, has_nli=run_.verifier(True, False).nli is not None)
    out, t0 = [], time.time()
    for n, c in enumerate(claims, 1):
        claim = make_claim(c["claim"].rstrip("."), sentence=c["claim"])
        t = time.time()
        v, route = check_claim(claim, run_, router, components=components, calibrator=calibrator, max_tier=max_tier)
        pre = v.leaning or v.status                     # verdict before the judge's abstention
        retrieved = {i.source.id for i in v.supporting + v.contradicting}
        out.append({
            "id": c["id"], "claim": c["claim"], "gold": c["gold"],
            "status": pre.value, "pred": LABEL[pre], "final_status": v.status.value, "final": LABEL[v.status],
            "decision": v.decision.value if v.decision else None,
            "correct": LABEL[pre] == c["gold"], "final_correct": LABEL[v.status] == c["gold"],
            "signals": {k: getattr(v.confidence_detail, k) for k in (
                "evidence_support", "contradiction", "source_quality", "source_independence", "verifier_agreement",
                "model_agreement", "historical_reliability", "unresolved_uncertainty", "difficulty_penalty")},
            "raw": v.confidence_detail.raw_score, "calibrated": v.confidence_detail.calibrated,
            "support": v.support_score, "contra": v.contradiction_score, "conflicting": v.conflicting,
            "evidence_hit": bool(set(c["evidence_docs"]) & retrieved), "has_gold_evidence": bool(c["evidence_docs"]),
            "critic_changed": bool(v.critique and not v.critique.survived),
            "red_flags": [f.kind for f in v.red_flags], "tiers": [p.get("tier") for p in route["passes"]],
            "latency_s": round(time.time() - t, 3)})
        if progress and (n % 25 == 0 or n == len(claims)):
            acc = sum(r["final_correct"] for r in out) / len(out)
            print(f"  [{split}{tag}] {n}/{len(claims)}  accuracy so far {acc:.3f}  ({time.time() - t0:.0f}s)",
                  flush=True)
    return out


def save(records: list[dict], name: str) -> Path:
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    path = BENCH_DIR / f"{name}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    return path


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in open(BENCH_DIR / f"{name}.jsonl")]
