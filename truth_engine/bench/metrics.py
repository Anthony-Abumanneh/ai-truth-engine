"""Benchmark metrics (spec §15; CONCEPTS.md §8, §10).

Every system (ours, baselines, ablations) produces records with:
    gold, pred                SUPPORT | CONTRADICT | NEI
    confidence                P(conf_pred is right), as the system reports it (None if it can't)
    conf_pred                 the label the confidence refers to (ours: the verdict before abstention)
    answered                  True if the system *stated* a verdict (not NEI / abstained)
    cited                     source ids it relied on (optional)
    gold_docs                 source ids holding gold evidence (SciFact)
    latency_s, cost_usd, tokens
"""
from __future__ import annotations

import numpy as np

from ..pipeline.confidence import brier, ece, reliability

LABELS = ["SUPPORT", "CONTRADICT", "NEI"]


def prf(gold: list[str], pred: list[str], label: str) -> tuple[float, float, float]:
    tp = sum(g == p == label for g, p in zip(gold, pred))
    p = tp / max(1, pred.count(label))
    r = tp / max(1, gold.count(label))
    return round(p, 4), round(r, 4), round(2 * p * r / (p + r), 4) if p + r else 0.0


def summarize(records: list[dict], name: str = "") -> dict:
    gold = [r["gold"] for r in records]
    pred = [r["pred"] for r in records]
    n = len(records)
    per = {lab: dict(zip(("precision", "recall", "f1"), prf(gold, pred, lab))) for lab in LABELS}
    out = {
        "system": name, "n": n,
        "accuracy": round(np.mean([g == p for g, p in zip(gold, pred)]), 4) if n else None,
        "macro_f1": round(np.mean([per[lab]["f1"] for lab in LABELS]), 4),
        "per_label": per,
        # the spec's "contradiction detection": of the claims experts say are false, how many did we catch?
        "contradiction_recall": per["CONTRADICT"]["recall"],
        # abstention quality: are its "not enough info" answers right, and does it say so when it should?
        "nei_precision": per["NEI"]["precision"], "nei_recall": per["NEI"]["recall"],
    }
    answered = [r for r in records if r.get("answered", r["pred"] != "NEI")]
    out["coverage"] = round(len(answered) / n, 4) if n else None
    out["accuracy_on_answered"] = round(np.mean([r["gold"] == r["pred"] for r in answered]), 4) if answered else None
    conf = [r for r in records if r.get("confidence") is not None]
    if conf:
        probs = [float(r["confidence"]) for r in conf]
        right = [r["gold"] == r.get("conf_pred", r["pred"]) for r in conf]   # confidence is about conf_pred
        out["ece"], out["brier"] = ece(probs, right), brier(probs, right)
        out["reliability"] = reliability(probs, right)
    cites = [r for r in records if r.get("cited") is not None and r.get("gold_docs")]
    if cites:
        with_cites = [r for r in cites if r["cited"]]
        out["citation_precision"] = round(np.mean([len(set(r["cited"]) & set(r["gold_docs"])) / len(r["cited"])
                                                   for r in with_cites]), 4) if with_cites else None
        out["citation_recall"] = round(np.mean([bool(set(r["cited"]) & set(r["gold_docs"])) for r in cites]), 4)
    out["latency_s_mean"] = round(np.mean([r.get("latency_s", 0.0) for r in records]), 3) if n else None
    out["cost_usd_total"] = round(sum(r.get("cost_usd", 0.0) for r in records), 4)
    out["tokens_total"] = int(sum(r.get("tokens", 0) for r in records))
    if out["cost_usd_total"] > 0:
        out["accuracy_per_dollar"] = round(out["accuracy"] / out["cost_usd_total"], 2)
    return out


def bootstrap_diff(a: list[dict], b: list[dict], metric: str = "accuracy", n_boot: int = 2000,
                   seed: int = 0) -> dict:
    """Paired bootstrap over claims: is system a better than b, or is the gap noise?
    Records must be aligned by claim id."""
    ids = [r["id"] for r in a]
    bmap = {r["id"]: r for r in b}
    pairs = [(r, bmap[r["id"]]) for r in a if r["id"] in bmap]
    rng = np.random.default_rng(seed)

    def m(rs):
        if metric == "accuracy":
            return np.mean([r["gold"] == r["pred"] for r in rs])
        return summarize(rs)[metric]
    base = m([p for p, _ in pairs]) - m([q for _, q in pairs])
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(pairs), len(pairs))
        sample = [pairs[i] for i in idx]
        diffs.append(m([p for p, _ in sample]) - m([q for _, q in sample]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": round(float(base), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "significant": bool(lo > 0 or hi < 0), "n": len(pairs), "ids": len(ids)}


def red_flag_scores(records: list[dict]) -> dict:
    """Micro precision / recall of red flags against expected flags (built-in set)."""
    tp = fp = fn = 0
    for r in records:
        exp, got = set(r.get("expected_flags") or []), set(r.get("red_flags") or [])
        tp, fp, fn = tp + len(exp & got), fp + len(got - exp), fn + len(exp - got)
    p, rc = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return {"flag_precision": round(p, 4), "flag_recall": round(rc, 4), "expected": tp + fn, "raised": tp + fp}
