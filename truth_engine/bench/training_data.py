"""Training data for fine-tuning the NLI verifier (step 8.5).

(premise, hypothesis, label) pairs, label in {entailment, neutral, contradiction},
from three expert-labelled sources:

    SciFact train   claim vs. its rationale sentences (support / contradict);
                    other sentences of the same abstracts as hard negatives (neutral);
                    cited abstracts of NOT ENOUGH INFO claims (neutral)
    HealthVer       COVID health claims vs. evidence snippets (train split)
    COVID-Fact      COVID claims vs. evidence sentences (supported / refuted)

Splits:
    train   everything above except the held-out SciFact claims
    val     10% of SciFact *train* claims (by claim id) + HealthVer dev: model selection
    never   SciFact dev (our benchmark) and HealthVer test (generalization check)

    python -m truth_engine.bench.training_data      # download + build data/finetune/*.jsonl
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path

from ..config import DATA_DIR
from .scifact import SCIFACT

OUT = DATA_DIR / "finetune"
SOURCES = {
    "healthver_train.csv": "https://raw.githubusercontent.com/sarrouti/HealthVer/master/data/healthver_train.csv",
    "healthver_dev.csv": "https://raw.githubusercontent.com/sarrouti/HealthVer/master/data/healthver_dev.csv",
    "healthver_test.csv": "https://raw.githubusercontent.com/sarrouti/HealthVer/master/data/healthver_test.csv",
    "covidfact.jsonl": "https://raw.githubusercontent.com/asaakyan/covidfact/main/COVIDFACT_dataset.jsonl",
}
SCIFACT_LABEL = {"SUPPORT": "entailment", "CONTRADICT": "contradiction"}
HEALTHVER_LABEL = {"Supports": "entailment", "Refutes": "contradiction", "Neutral": "neutral"}
COVIDFACT_LABEL = {"SUPPORTED": "entailment", "REFUTED": "contradiction"}
VAL_FRACTION = 0.1
SCIFACT_REPEAT = 2      # SciFact is the target domain but ~15% of pairs: show it twice per epoch


def download(out: Path = OUT) -> None:
    import httpx
    out.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        if not (out / name).exists():
            r = httpx.get(url, follow_redirects=True, timeout=120)
            r.raise_for_status()
            (out / name).write_bytes(r.content)


def _pair(premise: str, hypothesis: str, label: str, source: str) -> dict:
    return {"premise": premise.strip(), "hypothesis": hypothesis.strip(), "label": label, "source": source}


def scifact_pairs(split: str = "train", rng: random.Random | None = None,
                  negatives_per_doc: int = 2) -> dict[int, list[dict]]:
    """Pairs grouped by claim id (so splits never put one claim on both sides)."""
    rng = rng or random.Random(0)
    corpus = {d["doc_id"]: d for d in map(json.loads, open(SCIFACT / "corpus.jsonl"))}
    out: dict[int, list[dict]] = {}
    for c in map(json.loads, open(SCIFACT / f"claims_{split}.jsonl")):
        pairs = []
        if c["evidence"]:
            for doc_id, sets in c["evidence"].items():
                abstract = corpus[int(doc_id)]["abstract"]
                label = SCIFACT_LABEL[sets[0]["label"]]
                rationale = sorted({i for s in sets for i in s["sentences"]})
                pairs.append(_pair(" ".join(abstract[i] for i in rationale), c["claim"], label, "scifact"))
                for s in sets:                       # each evidence set alone, when it differs
                    if len(sets) > 1 or sorted(s["sentences"]) != rationale:
                        pairs.append(_pair(" ".join(abstract[i] for i in sorted(s["sentences"])), c["claim"], label,
                                           "scifact"))
                others = [i for i in range(len(abstract)) if i not in rationale]
                for i in rng.sample(others, min(negatives_per_doc, len(others))):   # hard negatives
                    pairs.append(_pair(" ".join(abstract[i:i + 2]), c["claim"], "neutral", "scifact-negative"))
        else:
            for doc_id in c["cited_doc_ids"][:2]:
                abstract = corpus[doc_id]["abstract"]
                pairs.append(_pair(" ".join(abstract[:2]), c["claim"], "neutral", "scifact-nei"))
                if len(abstract) > 3:
                    i = rng.randrange(2, len(abstract) - 1)
                    pairs.append(_pair(" ".join(abstract[i:i + 2]), c["claim"], "neutral", "scifact-nei"))
        out[c["id"]] = [dict(p) for p in {(p["premise"], p["label"]): p for p in pairs}.values()]
    return out


def healthver_pairs(split: str, src: Path = OUT) -> list[dict]:
    return [_pair(r["evidence"], r["claim"], HEALTHVER_LABEL[r["label"]], "healthver")
            for r in csv.DictReader(open(src / f"healthver_{split}.csv")) if r["label"] in HEALTHVER_LABEL]


def covidfact_pairs(src: Path = OUT) -> list[dict]:
    out = []
    for r in map(json.loads, open(src / "covidfact.jsonl")):
        ev = r["evidence"] if isinstance(r["evidence"], list) else [r["evidence"]]
        if ev and r["label"] in COVIDFACT_LABEL:
            out.append(_pair(" ".join(ev[:3]), r["claim"], COVIDFACT_LABEL[r["label"]], "covidfact"))
    return out


def build(out: Path = OUT, seed: int = 0) -> dict:
    download(out)
    rng = random.Random(seed)
    sf = scifact_pairs("train", rng)
    ids = sorted(sf)
    random.Random(seed).shuffle(ids)
    held = set(ids[: int(len(ids) * VAL_FRACTION)])
    # never train or select on a SciFact dev claim (some reappear word-for-word in HealthVer/COVID-Fact)
    dev = {json.loads(line)["claim"].strip().lower() for line in open(SCIFACT / "claims_dev.jsonl")}
    clean = lambda rows: [p for p in rows if p["hypothesis"].lower() not in dev]   # noqa: E731
    train = clean([p for cid in ids if cid not in held for p in sf[cid]] * SCIFACT_REPEAT
                  + healthver_pairs("train", out) + covidfact_pairs(out))
    val = clean([p for cid in ids if cid in held for p in sf[cid]] + healthver_pairs("dev", out))
    rng.shuffle(train)
    for name, rows in (("train", train), ("val", val)):
        (out / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    stats = {"removed_dev_overlap": True} | {name: {"n": len(rows), "labels": dict(Counter(r["label"] for r in rows)),
                    "sources": dict(Counter(r["source"] for r in rows))} for name, rows in (("train", train), ("val", val))}
    (out / "stats.json").write_text(json.dumps(stats | {"seed": seed, "held_out_scifact_claims": sorted(held)}, indent=1))
    return stats


if __name__ == "__main__":
    print(json.dumps(build(), indent=1))
