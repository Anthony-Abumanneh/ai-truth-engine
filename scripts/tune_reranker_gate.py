"""Choose the reranker gate threshold end to end on SciFact TRAIN claims (dev untouched).

    python scripts/tune_reranker_gate.py [n_claims] [model_key]
"""
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from truth_engine.bench import scifact as sf  # noqa: E402
from truth_engine.bench.metrics import summarize  # noqa: E402
from truth_engine.pipeline import verifier as V  # noqa: E402
from truth_engine.pipeline.audit import RAW_COMPONENTS  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
if len(sys.argv) > 2:
    import os
    os.environ["TE_RERANKER_MODEL"] = sys.argv[2]
ids = [c["id"] for c in sf.load_claims("train")]
random.Random(0).shuffle(ids)
ids = set(ids[:n])
engine = sf.build_engine()
for gate in (None, 0.05, 0.2, 0.5, 0.8):
    V.RERANK_GATE = gate
    t = time.time()
    recs = sf.run("train", components=RAW_COMPONENTS, engine=engine, ids=ids, progress=False)
    s = summarize([{"gold": r["gold"], "pred": r["pred"]} for r in recs])
    print(f"gate={gate!s:5}  accuracy {s['accuracy']:.3f}  macro-F1 {s['macro_f1']:.3f}  "
          f"NEI predicted {sum(r['pred'] == 'NEI' for r in recs)}/{len(recs)}  ({time.time() - t:.0f}s)", flush=True)
