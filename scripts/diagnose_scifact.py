"""Where does the end-to-end pipeline lose SciFact claims that experts label
SUPPORT or CONTRADICT? Traces each claim through retrieval -> stance -> verdict.

    python scripts/diagnose_scifact.py [dev|train] [limit]

Buckets (first failing stage):
    no_gold_doc         no sentence from an expert-cited abstract was retrieved
    no_rationale        the abstract was retrieved, but not its rationale sentences
    rationale_gated     a rationale sentence was retrieved but the NLI model never read it
    rationale_neutral   the model read it and called it neutral
    rationale_wrong     the model read it and called the opposite stance
    aggregation         a rationale sentence was judged correctly, yet the verdict is wrong
    correct             the verdict matches the expert label
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from truth_engine.bench import scifact as sf  # noqa: E402
from truth_engine.pipeline.analyzer import analyze  # noqa: E402
from truth_engine.pipeline.audit import _routed, _Run  # noqa: E402
from truth_engine.pipeline.claims import make_claim  # noqa: E402
from truth_engine.pipeline.router import Router  # noqa: E402
from truth_engine.schemas import Stance  # noqa: E402
from truth_engine.textutils import content_tokens  # noqa: E402

split = sys.argv[1] if len(sys.argv) > 1 else "dev"
limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
corpus = {f"sf-{d['doc_id']}": d for d in map(json.loads, open(sf.SCIFACT / "corpus.jsonl"))}
raw = {c["id"]: c for c in map(json.loads, open(sf.SCIFACT / f"claims_{split}.jsonl"))}
WANT = {"SUPPORT": Stance.ENTAILS, "CONTRADICT": Stance.CONTRADICTS}


def overlaps(a: str, b: str) -> bool:
    ta, tb = set(content_tokens(a)), set(content_tokens(b))
    return bool(ta) and bool(tb) and len(ta & tb) / min(len(ta), len(tb)) >= 0.8


engine = sf.build_engine()
run = _Run(engine, None)
router = Router(live=False, has_nli=run.verifier(True, False).nli is not None)
rows, buckets = [], Counter()
claims = [c for c in sf.load_claims(split) if c["gold"] in WANT][:limit]
for n, c in enumerate(claims, 1):
    claim = make_claim(c["claim"].rstrip("."), sentence=c["claim"])
    v, retrieved, st, route = _routed(claim, analyze(claim.text), run, router, None, None)
    rationale = [corpus[f"sf-{d}"]["abstract"][i] for d, sets in raw[c["id"]]["evidence"].items()
                 for s in sets for i in s["sentences"]]
    gold_docs = set(c["evidence_docs"])
    got_doc = any(r.source.id in gold_docs for r in retrieved)
    rat = [r for r in retrieved if r.source.id in gold_docs and any(overlaps(r.passage.text, x) for x in rationale)]
    items = {i.passage.id: i for i in run.verifier(st.nli, st.llm).evidence_items(claim, rat)} if rat else {}
    want = WANT[c["gold"]]
    stances = [items[r.passage.id].stance for r in rat if r.passage.id in items]
    read = [items[r.passage.id].nli is not None for r in rat if r.passage.id in items]
    pred = sf.LABEL[v.status]
    if pred == c["gold"]:
        b = "correct"
    elif not got_doc:
        b = "no_gold_doc"
    elif not rat:
        b = "no_rationale"
    elif any(s == want or (want == Stance.ENTAILS and s == Stance.PARTIAL) for s in stances):
        b = "aggregation"
    elif any(s not in (Stance.NEUTRAL,) for s in stances):
        b = "rationale_wrong"
    elif not any(read):
        b = "rationale_gated"
    else:
        b = "rationale_neutral"
    buckets[b] += 1
    rows.append({"id": c["id"], "claim": c["claim"], "gold": c["gold"], "pred": pred, "bucket": b,
                 "support": v.support_score, "contra": v.contradiction_score,
                 "rationale_stances": [s.value for s in stances], "n_retrieved": len(retrieved),
                 "rationale_in_retrieved": len(rat)})
    if n % 25 == 0 or n == len(claims):
        print(f"  {n}/{len(claims)} {dict(buckets)}", flush=True)

out = sf.BENCH_DIR / f"diagnosis_{split}.jsonl"
out.write_text("\n".join(json.dumps(r) for r in rows))
print(f"\n{split}: {len(rows)} SUPPORT/CONTRADICT claims")
for b, k in buckets.most_common():
    print(f"  {b:18} {k:4}  ({k / len(rows):.0%})")
