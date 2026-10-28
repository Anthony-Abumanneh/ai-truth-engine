"""CI accuracy gate: fail if a change makes the engine measurably worse on SciFact dev.

    python scripts/benchmark.py ours --fresh && python scripts/eval_gate.py
    python scripts/eval_gate.py --update      # accept the current numbers as the new baseline

Compares data/bench/report_dev.json ("ours") with bench/eval_baseline.json. Tolerances
absorb run-to-run noise (CPU threading, library versions); anything beyond them fails CI.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "bench" / "eval_baseline.json"
REPORT = ROOT / "data" / "bench" / "report_dev.json"
# metric: (direction, tolerance). "up" = higher is better.
CHECKS = {"accuracy": ("up", 0.02), "macro_f1": ("up", 0.02), "accuracy_on_answered": ("up", 0.03),
          "ece": ("down", 0.02), "citation_recall": ("up", 0.03)}


def main() -> int:
    ours = json.loads(REPORT.read_text())["systems"]["ours"]
    current = {k: ours.get(k) for k in CHECKS} | {"n": ours["n"]}
    if "--update" in sys.argv:
        BASELINE.parent.mkdir(exist_ok=True)
        BASELINE.write_text(json.dumps(current, indent=1) + "\n")
        print("baseline updated:", current)
        return 0
    base = json.loads(BASELINE.read_text())
    if current["n"] != base["n"]:
        print(f"FAIL: evaluated {current['n']} claims, baseline has {base['n']}")
        return 1
    failed = False
    for k, (direction, tol) in CHECKS.items():
        b, c = base[k], current[k]
        worse = (b - c) if direction == "up" else (c - b)
        status = "FAIL" if worse > tol else "ok"
        failed |= status == "FAIL"
        print(f"{status:4}  {k:22} baseline {b:.3f}  now {c:.3f}  (allowed to worsen by {tol})")
    print("Gate:", "FAILED: this change makes the engine worse" if failed else "passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
