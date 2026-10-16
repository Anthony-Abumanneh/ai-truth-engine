"""Raw NLI quality of one or more verifier models (no rules, no retrieval).

    python scripts/eval_nli.py large ft-large base ft-base

Reports macro-F1 / accuracy on SciFact dev evidence pairs (our benchmark's claims)
and HealthVer test (never used for training or selection), plus speed.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from truth_engine.bench.training_data import OUT, download, healthver_pairs  # noqa: E402
from truth_engine.pipeline import nli  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_stance_scifact import load as scifact_dev  # noqa: E402

LABELS = ["entailment", "neutral", "contradiction"]
SF = {"support": "entailment", "contradict": "contradiction", "neutral": "neutral"}


def score(gold, pred):
    f1 = []
    for lab in LABELS:
        tp = sum(g == p == lab for g, p in zip(gold, pred))
        pr, rc = tp / max(1, pred.count(lab)), tp / max(1, gold.count(lab))
        f1.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return sum(g == p for g, p in zip(gold, pred)) / len(gold), sum(f1) / 3


download(OUT)
sets = {"scifact-dev": [(prem, claim, SF[g]) for claim, prem, _, g in scifact_dev("dev")],
        "healthver-test": [(p["premise"], p["hypothesis"], p["label"]) for p in healthver_pairs("test", OUT)]}
results = {}
for key in sys.argv[1:] or ["large"]:
    m = nli.load(key)
    if m is None:
        print(f"{key}: not available ({nli.model_dir(key)})")
        continue
    for name, rows in sets.items():
        t = time.time()
        probs = m.predict([(p, h) for p, h, _ in rows])
        ms = (time.time() - t) / len(rows) * 1000
        acc, f1 = score([g for *_, g in rows], [max(LABELS, key=lambda k: p[k]) for p in probs])
        results[f"{key}/{name}"] = {"accuracy": round(acc, 4), "macro_f1": round(f1, 4), "ms_per_pair": round(ms, 1)}
        print(f"{key:9} {name:15} n={len(rows):5}  accuracy {acc:.3f}  macro-F1 {f1:.3f}  ({ms:.0f} ms/pair on {m.device})",
              flush=True)
Path("data/finetune").mkdir(parents=True, exist_ok=True)
Path("data/finetune/eval_nli.json").write_text(json.dumps(results, indent=1))
