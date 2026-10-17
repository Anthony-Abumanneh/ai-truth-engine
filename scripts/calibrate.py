"""Fit confidence calibration + abstention threshold on SciFact train; report on dev.

    python scripts/calibrate.py [--limit=N] [--reuse] [--target=0.8]

1. Run the full pipeline on SciFact train (default components, with
   calibration and abstention off so raw verdicts and signals are recorded).
2. Fit the calibrator: logistic regression on the signals vs Platt scaling of
   the raw score, chosen by 5-fold cross-validated Brier score.
3. Choose the abstention threshold: the lowest one at which verdicts the
   engine *states* (supported / contradicted / partial) are right at least
   TARGET_PRECISION of the time on train. Maximizing plain accuracy picks 0
   on SciFact (the engine's main error is under-calling, which abstaining
   can't fix), so precision of stated verdicts is the product's criterion:
   a wrong stated verdict is worse than "unverified".
4. Save data/calibrator.json, then evaluate on dev: accuracy before/after
   abstention, coverage, macro-F1, ECE and Brier (raw vs calibrated), a
   reliability table, and the critic ablation.
`--reuse` re-analyses cached records in data/bench/ without rerunning.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from truth_engine.bench import scifact as sf  # noqa: E402
from truth_engine.pipeline.audit import RAW_COMPONENTS  # noqa: E402
from truth_engine.pipeline.confidence import (Calibrator, brier, cross_val_brier, ece, history_from,  # noqa: E402
                                              reliability)

LABELS = ["SUPPORT", "CONTRADICT", "NEI"]


def macro_f1(gold, pred):
    f = []
    for lab in LABELS:
        tp = sum(g == p == lab for g, p in zip(gold, pred))
        pr, rc = tp / max(1, pred.count(lab)), tp / max(1, gold.count(lab))
        f.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return round(sum(f) / 3, 4)


def with_history(recs, hist):
    out = []
    for r in recs:
        s = dict(r["signals"])
        s["historical_reliability"] = hist.get(r["status"], 0.5)
        out.append(s)
    return out


def apply(recs, cal: Calibrator, thr: float):
    sigs = with_history(recs, cal.history)
    probs = [cal(s, r["status"]) for s, r in zip(sigs, recs)]
    final = []
    for r, p in zip(recs, probs):
        decisive = r["status"] != "unverified" and not r["conflicting"]
        final.append(r["pred"] if (not decisive or p >= thr) else "NEI")
    return probs, final


def summary(name, recs, cal, thr):
    gold = [r["gold"] for r in recs]
    probs, final = apply(recs, cal, thr)
    correct = [r["correct"] for r in recs]
    decisive = [r["status"] != "unverified" and not r["conflicting"] for r in recs]
    answered = [d and p >= thr for d, p in zip(decisive, probs)]
    sel = [c for c, a in zip(correct, answered) if a]
    return {
        "set": name, "n": len(recs),
        "accuracy_raw": round(np.mean(correct), 4), "macro_f1_raw": macro_f1(gold, [r["pred"] for r in recs]),
        "accuracy_final": round(np.mean([g == f for g, f in zip(gold, final)]), 4),
        "macro_f1_final": macro_f1(gold, final),
        "decisive_share": round(np.mean(decisive), 4), "answered_share": round(np.mean(answered), 4),
        "accuracy_on_answered": round(np.mean(sel), 4) if sel else None,
        "ece_raw": ece([r["raw"] for r in recs], correct), "ece_calibrated": ece(probs, correct),
        "brier_raw": brier([r["raw"] for r in recs], correct), "brier_calibrated": brier(probs, correct),
        "evidence_hit_rate": round(np.mean([r["evidence_hit"] for r in recs if r["has_gold_evidence"]]), 4),
        "reliability": reliability(probs, correct),
    }


def main():
    limit = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--limit=")), None)
    reuse = "--reuse" in sys.argv
    heldout = "--heldout" in sys.argv     # fit only on SciFact train claims the fine-tuned verifier never saw
    tag = (f"_n{limit}" if limit else "") + ("_heldout" if heldout else "")
    engine = None
    if reuse:
        train = sf.load(f"train{tag}")
    else:
        engine = sf.build_engine()
        ids = set(json.loads((sf.DATA_DIR / "finetune" / "stats.json").read_text())["held_out_scifact_claims"]) \
            if heldout else None
        train = sf.run("train", components=RAW_COMPONENTS, limit=limit, engine=engine, ids=ids)
        sf.save(train, f"train{tag}")

    hist = history_from([r["status"] for r in train], [r["correct"] for r in train])
    sigs, statuses, correct = with_history(train, hist), [r["status"] for r in train], [r["correct"] for r in train]
    cv = {m: round(cross_val_brier(m, sigs, statuses, correct), 4) for m in ("logistic", "platt")}
    method = min(cv, key=cv.get)
    cal = Calibrator(method, history=hist).fit(sigs, statuses, correct)
    target = next((float(a.split("=")[1]) for a in sys.argv if a.startswith("--target=")), 0.8)
    gold = [r["gold"] for r in train]
    probs = [cal(s, st) for s, st in zip(sigs, statuses)]
    decisive = [r["status"] != "unverified" and not r["conflicting"] for r in train]
    grid = [round(t, 2) for t in np.arange(0.0, 0.96, 0.05)]

    def precision(t):
        sel = [c for c, d, p in zip(correct, decisive, probs) if d and p >= t]
        return np.mean(sel) if len(sel) >= 30 else 0.0
    acc_at = lambda t: np.mean([g == f for g, f in zip(gold, apply(train, cal, t)[1])])   # noqa: E731
    thr = next((t for t in grid if precision(t) >= target), None)
    if thr is None:              # target unreachable (e.g. few calibration claims): best accuracy instead
        thr = max(grid, key=lambda t: (round(acc_at(t), 4), -t))
        print(f"  precision target {target} unreachable on {len(train)} claims; threshold by accuracy instead")
    acc = {t: acc_at(t) for t in (0.0, thr)}
    cal.threshold = thr
    cal.meta = {"fit_on": f"scifact train ({len(train)} claims)", "cv_brier": cv, "method": method,
                "target_precision": target, "train_precision_at_threshold": round(float(precision(thr)), 4),
                "train_accuracy_at_threshold": round(float(acc[thr]), 4)}
    cal.save()
    print(f"calibrator: {method} (CV Brier {cv}); abstention threshold {thr}: stated verdicts "
          f"{precision(thr):.3f} right on train (target {target}); accuracy {acc[thr]:.3f} vs {acc[0.0]:.3f} at 0")

    if reuse:
        dev, dev_critic = sf.load(f"dev{tag}"), sf.load(f"dev_critic{tag}")
    else:
        dev = sf.run("dev", components=RAW_COMPONENTS, limit=limit, engine=engine)
        sf.save(dev, f"dev{tag}")
        if (sf.BENCH_DIR / f"dev_critic{tag}.jsonl").exists():      # ablation records are reused
            dev_critic = sf.load(f"dev_critic{tag}")
        elif "--no-critic" in sys.argv:                            # critic is opt-in; skip its 3x slower ablation
            dev_critic = dev
        else:
            dev_critic = sf.run("dev", components=RAW_COMPONENTS | {"critic"}, limit=limit, engine=engine,
                                tag=" (critic)")
            sf.save(dev_critic, f"dev_critic{tag}")
    report = {"calibrator": cal.meta | {"threshold": thr},
              "train": summary("train", train, cal, thr), "dev": summary("dev", dev, cal, thr),
              "dev_with_critic": summary("dev (with critic)", dev_critic, cal, thr)}
    (sf.BENCH_DIR / f"calibration_report{tag}.json").write_text(json.dumps(report, indent=1))
    for k in ("train", "dev", "dev_with_critic"):
        s = report[k]
        print(f"\n{s['set']}: n={s['n']}  accuracy raw {s['accuracy_raw']} -> with abstention {s['accuracy_final']}  "
              f"macro-F1 {s['macro_f1_raw']} -> {s['macro_f1_final']}\n"
              f"   answered {s['answered_share']:.0%} at accuracy {s['accuracy_on_answered']}  |  "
              f"ECE raw {s['ece_raw']} -> calibrated {s['ece_calibrated']}  |  "
              f"Brier {s['brier_raw']} -> {s['brier_calibrated']}  |  evidence found for {s['evidence_hit_rate']:.0%}")
    print("\nreliability (dev):")
    for row in report["dev"]["reliability"]:
        print(f"   {row['bin']}  n={row['n']:3}  says {row['mean_confidence']:.2f}  right {row['accuracy']:.2f}")


if __name__ == "__main__":
    main()
