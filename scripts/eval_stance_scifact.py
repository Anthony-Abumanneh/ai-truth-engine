"""Stance check on SciFact dev (oracle evidence): does the verifier label an
expert-chosen evidence passage correctly as support / contradict / neutral?

Isolates the stance step (no retrieval, no aggregation). Compares the rules,
the local NLI model, and the fused verifier. Step 8 builds the full benchmark.

    python scripts/eval_stance_scifact.py
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from truth_engine.pipeline.claims import make_claim          # noqa: E402
from truth_engine.pipeline.nli import get_nli                # noqa: E402
from truth_engine.pipeline.verifier import NLI_DECIDE, fuse, rule_stance  # noqa: E402
from truth_engine.schemas import Stance                       # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "scifact" / "data"
LABELS = ["support", "contradict", "neutral"]


def load(split: str = "dev"):
    corpus = {d["doc_id"]: d for d in map(json.loads, open(DATA / "corpus.jsonl"))}
    rows = []
    for c in map(json.loads, open(DATA / f"claims_{split}.jsonl")):
        if c["evidence"]:
            for doc_id, sets in c["evidence"].items():
                doc = corpus[int(doc_id)]
                sents = sorted({i for s in sets for i in s["sentences"]})
                label = sets[0]["label"].lower()
                rows.append((c["claim"], " ".join(doc["abstract"][i] for i in sents), doc["title"], label))
        else:
            doc = corpus[c["cited_doc_ids"][0]]
            rows.append((c["claim"], " ".join(doc["abstract"][:2]), doc["title"], "neutral"))
    return rows


def as_label(st: Stance) -> str:
    return {Stance.ENTAILS: "support", Stance.PARTIAL: "support", Stance.CONTRADICTS: "contradict"}.get(st, "neutral")


def nli_label(p: dict) -> str:
    if p["entailment"] >= NLI_DECIDE:
        return "support"
    if p["contradiction"] >= NLI_DECIDE:
        return "contradict"
    return "neutral"


def report(name, gold, pred):
    acc = sum(g == p for g, p in zip(gold, pred)) / len(gold)
    f1s = []
    for lab in LABELS:
        tp = sum(g == p == lab for g, p in zip(gold, pred))
        prec = tp / max(1, sum(p == lab for p in pred))
        rec = tp / max(1, sum(g == lab for g in gold))
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    print(f"{name:14} accuracy {acc:.3f}   macro-F1 {sum(f1s) / 3:.3f}   "
          + "  ".join(f"F1[{lab}] {f:.2f}" for lab, f in zip(LABELS, f1s)))


def main():
    rows = load(sys.argv[1] if len(sys.argv) > 1 else "dev")
    gold = [r[3] for r in rows]
    model = get_nli()
    print(f"SciFact: {len(rows)} claim/evidence pairs {dict(Counter(gold))}; model: "
          f"{model.name if model else 'none'}")
    rules, nli, fused = [], [], []
    t0 = time.time()
    probs = model.predict([(premise, claim) for claim, premise, _, _ in rows]) if model else [None] * len(rows)
    for (claim_text, premise, title, _), p in zip(rows, probs):
        rs = rule_stance(make_claim(claim_text), premise, title)
        rules.append(as_label(rs.stance))
        if p:
            nli.append(nli_label(p))
            fused.append(as_label(fuse(rs, p)[0]))
    print(f"({time.time() - t0:.1f}s)")
    report("rules", gold, rules)
    if model:
        report("NLI model", gold, nli)
        report("fused", gold, fused)
    else:
        print("NLI model not available (python -m truth_engine.pipeline.nli download)")


if __name__ == "__main__":
    main()
