"""Step 8 benchmark: the full engine vs. LLM baselines, ablations, and the
calibration report card.

    python scripts/benchmark.py ours ablations builtin      # no API keys needed
    python scripts/benchmark.py llm --llm-n=100            # needs GEMINI/GROQ/... keys
    python scripts/benchmark.py llm --extras               # + few-shot RAG and ours + LLM judge
    python scripts/benchmark.py report                     # rebuild docs/benchmark_report.md from cache

Everything is evaluated on SciFact *dev* (held out: calibration and every
tuned setting used train only), except the built-in trap set (in-sample).
Records are cached in data/bench/sys_*.jsonl; `--fresh` reruns them.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from truth_engine.bench import builtin, metrics, systems  # noqa: E402
from truth_engine.bench import scifact as sf  # noqa: E402
from truth_engine.pipeline.audit import DEFAULT_COMPONENTS  # noqa: E402

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
OPT = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
SPLIT = OPT.get("split", "dev")
N = int(OPT["n"]) if "n" in OPT else None
LLM_N = int(OPT.get("llm-n", 100))
FRESH = "fresh" in OPT
ABLATIONS = ["nli", "falsification", "independence", "source_quality", "adaptive", "calibration", "abstention"]
_engine = None


def engine():
    global _engine
    if _engine is None:
        _engine = sf.build_engine()
    return _engine


def cases(n=None):
    return [{"id": c["id"], "claim": c["claim"], "gold": c["gold"], "gold_docs": c["evidence_docs"]}
            for c in sf.load_claims(SPLIT)][:n]


def cached(name, fn):
    path = sf.BENCH_DIR / f"sys_{name}_{SPLIT}.jsonl"
    if path.exists() and not FRESH:
        return [json.loads(x) for x in open(path)]
    recs = fn()
    failed = sum(bool(r.get("errors")) for r in recs)
    if failed > 0.1 * len(recs):          # e.g. a daily quota ran out: don't save fake "NEI" results
        print(f"  NOT SAVED: {name}: {failed}/{len(recs)} calls failed ({recs[-1]['errors'][0][:80]})", flush=True)
        return recs
    sf.save(recs, f"sys_{name}_{SPLIT}")
    return recs


def run_all():
    if "ours" in ARGS:
        cached("ours", lambda: systems.run_ours(cases(N), engine(), progress="ours"))
    if "ablations" in ARGS:
        for comp in ABLATIONS:
            cached(f"no_{comp}", lambda c=comp: systems.run_ours(cases(N), engine(), DEFAULT_COMPONENTS - {c},
                                                                 progress=f"no {c}"))
        cached("plus_critic", lambda: systems.run_ours(cases(N), engine(), DEFAULT_COMPONENTS | {"critic"},
                                                       progress="+ critic"))
    if "builtin" in ARGS:
        from truth_engine.retrieval import EvidenceEngine
        recs = systems.run_ours(builtin.load(), EvidenceEngine())
        sf.save(recs, "sys_builtin")
    if "llm" in ARGS:
        llms = systems.available_llms(float(OPT["interval"]) if "interval" in OPT else None)
        if not llms:
            sys.exit("no LLM keys found (.env: GEMINI_API_KEY / GROQ_API_KEY / ...)")
        sub = cases(LLM_N)
        for llm in llms:
            tag = llm.name.replace(":", "_").replace("/", "_")
            cached(f"A_single_{tag}", lambda m=llm: systems.run_single(sub, m))
            cached(f"B_rag_{tag}", lambda m=llm: systems.run_rag(sub, m, engine()))
        best = next((m for m in llms if "gpt-oss-120b" in m.name), llms[0])
        tag = best.name.replace(":", "_").replace("/", "_")
        if "extras" in OPT:          # the key experiment first, in case the daily quota runs out
            cached(f"ours_llm_judge_{tag}", lambda: systems.run_ours_llm_judge(sub, best, engine()))
        if "fewshot" in OPT or "extras" in OPT:
            cached(f"B_rag_fewshot_{tag}", lambda: systems.run_rag_fewshot(sub, best, engine()))
        if len(llms) >= 2:
            judge = max(llms, key=lambda m: ("large", "medium", "small").index(m.spec.tier) * -1)
            cached("C_vote", lambda: systems.run_vote(sub, llms))
            cached("D_judge", lambda: systems.run_judge(sub, llms, judge, engine()))


def report():
    rows, ours = [], None
    for path in sorted(sf.BENCH_DIR.glob(f"sys_*_{SPLIT}.jsonl")):
        name = path.stem[len("sys_"):-len(f"_{SPLIT}")]
        recs = [json.loads(x) for x in open(path)]
        rows.append((name, recs))
        if name == "ours":
            ours = recs
    out = {"split": SPLIT, "systems": {}, "vs_ours": {}}
    for name, recs in rows:
        s = metrics.summarize(recs, name)
        out["systems"][name] = s
        if ours and name != "ours":
            ids = {r["id"] for r in recs}
            same = [r for r in ours if r["id"] in ids]                  # compare on the same claims
            out["vs_ours"][name] = {"ours_on_same_claims": metrics.summarize(same, "ours")["accuracy"],
                                    **metrics.bootstrap_diff(same, recs)}
    b = sf.BENCH_DIR / "sys_builtin.jsonl"
    if b.exists():
        recs = [json.loads(x) for x in open(b)]
        out["builtin"] = metrics.summarize(recs, "builtin") | metrics.red_flag_scores(recs)
    (sf.BENCH_DIR / f"report_{SPLIT}.json").write_text(json.dumps(out, indent=1))
    write_markdown(out)
    return out


def fmt(x, pct=False):
    return "–" if x is None else (f"{x:.0%}" if pct else f"{x:.3f}" if isinstance(x, float) else str(x))


def write_markdown(out):
    S = out["systems"]
    lines = [f"# Benchmark report (SciFact {out['split']})", "",
             "Generated by `scripts/benchmark.py report`. SciFact dev is held out: calibration and every tuned "
             "setting used train only. Accuracy is 3-way (SUPPORT / CONTRADICT / NOT ENOUGH INFO).", ""]
    main = ["ours"] + sorted(k for k in S if k[:2] in ("A_", "B_", "C_", "D_") or k.startswith("ours_llm"))
    lines += ["## Engine vs baselines", "",
              "| System | n | Accuracy | Macro-F1 | Contradiction recall | Coverage | Acc. on answered | ECE | "
              "Citation recall | Latency (s) | Tokens | Cost ($) |", "|" + "---|" * 12]
    for k in main:
        if k in S:
            s = S[k]
            lines.append(f"| {k} | {s['n']} | {fmt(s['accuracy'])} | {fmt(s['macro_f1'])} | "
                         f"{fmt(s['contradiction_recall'])} | {fmt(s['coverage'], True)} | "
                         f"{fmt(s['accuracy_on_answered'])} | {fmt(s.get('ece'))} | {fmt(s.get('citation_recall'))} | "
                         f"{fmt(s['latency_s_mean'])} | {s['tokens_total']} | {fmt(s['cost_usd_total'])} |")
    if out["vs_ours"]:
        lines += ["", "Paired bootstrap on the same claims (accuracy difference, **ours minus other**, 95% CI; "
                  "positive = ours better):", ""]
        for k, d in sorted(out["vs_ours"].items()):
            lines.append(f"- {k}: {d['diff']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] "
                         f"(ours {d['ours_on_same_claims']:.3f} on n={d['n']})" + (" **significant**" if d["significant"] else ""))
    abl = sorted(k for k in S if k.startswith(("no_", "plus_")))
    if abl and "ours" in S:
        lines += ["", "## Ablations (remove one component; Δ = full minus variant, positive = the component helps)", "",
                  "| Variant | Accuracy | Δ vs full | 95% CI | Macro-F1 | ECE | Coverage | Acc. on answered | Latency (s) |",
                  "|" + "---|" * 9]
        for k in abl:
            s, d = S[k], out["vs_ours"].get(k, {})
            ci = d.get("ci95", [None, None])
            lines.append(f"| {k} | {fmt(s['accuracy'])} | {d.get('diff', 0):+.3f} | "
                         f"[{fmt(ci[0])}, {fmt(ci[1])}] | {fmt(s['macro_f1'])} | {fmt(s.get('ece'))} | "
                         f"{fmt(s['coverage'], True)} | {fmt(s['accuracy_on_answered'])} | {fmt(s['latency_s_mean'])} |")
    if "ours" in S and S["ours"].get("reliability"):
        lines += ["", "## Calibration report card (ours, dev)", "", "| Confidence bin | n | Says | Right |", "|---|---|---|---|"]
        for r in S["ours"]["reliability"]:
            lines.append(f"| {r['bin']} | {r['n']} | {r['mean_confidence']:.2f} | {r['accuracy']:.2f} |")
        lines.append(f"\nECE {fmt(S['ours'].get('ece'))}, Brier {fmt(S['ours'].get('brier'))}.")
    if "builtin" in out:
        b = out["builtin"]
        lines += ["", "## Built-in trap set (in-sample: used while building the rules)", "",
                  f"Accuracy {fmt(b['accuracy'])} on {b['n']} claims; red flags precision {fmt(b['flag_precision'])}, "
                  f"recall {fmt(b['flag_recall'])} ({b['raised']} raised, {b['expected']} expected)."]
    Path("docs").mkdir(exist_ok=True)
    Path("docs/benchmark_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    if any(a in ARGS for a in ("ours", "ablations", "builtin", "llm")):
        run_all()
    report()
