"""End-to-end A/B of verifier settings on SciFact TRAIN claims (dev stays untouched).

    python scripts/experiment_verifier.py [n_claims]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from truth_engine.bench import scifact as sf          # noqa: E402
from truth_engine.pipeline import verifier as V       # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
LABELS = ["SUPPORT", "CONTRADICT", "NEI"]


def macro_f1(gold, pred):
    f = []
    for lab in LABELS:
        tp = sum(g == p == lab for g, p in zip(gold, pred))
        pr, rc = tp / max(1, pred.count(lab)), tp / max(1, gold.count(lab))
        f.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return sum(f) / 3


engine = sf.build_engine()
for scope, ctx in (("gated", False), ("all", False), ("gated", True), ("all", True)):
    V.NLI_SCOPE, V.PREMISE_CONTEXT = scope, ctx
    t = time.time()
    recs = sf.run("train", components={"critic"}, limit=n, engine=engine, progress=False)
    gold, pred = [r["gold"] for r in recs], [r["pred"] for r in recs]
    acc = sum(g == p for g, p in zip(gold, pred)) / len(recs)
    hit = sum(r["evidence_hit"] for r in recs if r["has_gold_evidence"]) / max(1, sum(r["has_gold_evidence"] for r in recs))
    print(f"scope={scope:5} context={ctx!s:5}  accuracy {acc:.3f}  macro-F1 {macro_f1(gold, pred):.3f}  "
          f"gold evidence used {hit:.2f}  ({(time.time() - t) / len(recs):.1f}s/claim)", flush=True)
