"""Audit mode: text in, every check-worthy claim verified.

    text -> extract claims -> for each claim: plan support + falsification
    queries -> gather evidence -> verify (stance, lineages, status) -> red flags

    python -m truth_engine.pipeline.audit "Veridamine improves sleep by 18% and is clinically proven."
    python -m truth_engine.pipeline.audit --live "Creatine improves memory in older adults."
    python -m truth_engine.pipeline.audit --live --max-tier=maximum "..."

Claims like "X is clinically proven" carry no effect of their own; they are
verified through the effect stated next to them in the same sentence and
are only SUPPORTED when that effect is backed by strong, uncontested evidence.
This is the step-4 pipeline; step 7's orchestrator adds routing, critique,
calibration and tracing around it.
"""
from __future__ import annotations

import re
import sys
import time

from ..retrieval import EvidenceEngine
from ..schemas import AuditReport, Claim, ClaimVerdict, EvidenceStatus
from ..science.red_flags import red_flags
from ..textutils import content_tokens, short
from .claims import CERTAINTY, CHECK_WORTHY_THRESHOLD, extract_claims
from ..schemas import BudgetTier
from .analyzer import analyze
from .router import LIVE_ONLY, Router, Strategy
from .confidence import Calibrator, breakdown
from .critic import Critic
from .judge import decide
from .verifier import Verifier, study_of



def _certainty_only(claim: Claim) -> bool:
    if not CERTAINTY.search(claim.text):
        return False
    rest = set(content_tokens(CERTAINTY.sub(" ", claim.text)))
    subject = set(content_tokens(claim.entities[0])) if claim.entities else set()
    return not (rest - subject - {"clinic", "scientif", "proven", "studi", "show", "prov", "science"})


def _certainty_verdict(claim: Claim, effect: ClaimVerdict) -> ClaimVerdict:
    strong = [i for i in effect.supporting if study_of(i.source).get("design") in ("rct", "meta_analysis",
                                                                                   "systematic_review")
              and i.source.quality and i.source.quality.tier == "A"]
    if effect.status == EvidenceStatus.SUPPORTED and strong and not effect.contradicting:
        status = EvidenceStatus.SUPPORTED
    elif effect.status == EvidenceStatus.CONTRADICTED:
        status = EvidenceStatus.CONTRADICTED
    else:
        status = EvidenceStatus.UNVERIFIED
    v = effect.model_copy(deep=True)
    v.claim_id, v.status, v.red_flags = claim.id, status, []
    v.rationale = ("Certainty claim, judged by the strength of evidence for the effect it accompanies. "
                   + effect.rationale)
    return v


class _Run:
    """Per-audit state: the evidence engine, every source seen, and one verifier
    per (NLI, LLM) combination, built on first use."""

    def __init__(self, engine: EvidenceEngine, llm_verifier):
        self.engine = engine
        self.known = dict(engine.store.sources)
        self.llm_verifier = llm_verifier
        self._verifiers: dict[tuple[bool, bool], Verifier] = {}
        self.used: set[str] = set()

    def verifier(self, nli: bool, llm: bool) -> Verifier:
        key = (nli, llm and self.llm_verifier is not None)
        if key not in self._verifiers:
            v = Verifier(self.known, llm=self.llm_verifier if key[1] else None, nli="auto" if nli else None,
                         context_fn=lambda pid: " ".join(p.text for p in self.engine.store.neighbors(pid)))
            self._verifiers[key] = v
            self.used |= {"rules"} | ({f"nli:{v.nli.name}"} if v.nli else set()) | (
                {f"llm:{self.llm_verifier[1].name}"} if key[1] else set())
        return self._verifiers[key]

    def check(self, claim: Claim, st: Strategy) -> tuple[ClaimVerdict, list]:
        eng = self.engine
        retrieved = eng.gather(claim, eng.plan(claim, list(st.sources)), k_support=st.k_support,
                               k_falsify=st.k_falsify, follow_up_studies=st.follow_up_studies)
        self.sync()
        return self.verifier(st.nli, st.llm).verify(claim, retrieved), retrieved

    def sync(self) -> None:
        """Make sources found by the last search visible to the verifiers and red-flag checks."""
        self.known.update(self.engine.last_external_sources)
        if self.engine._science is not None:
            self.known.update(self.engine.science.sources)


# Default step-6 components. The critic is opt-in: on SciFact dev it changed
# neither accuracy (0.610 vs 0.613 without) nor calibration, at 3x the time.
ALL_COMPONENTS = frozenset({"calibration", "abstention"})


def check_claim(claim: Claim, run: _Run, router: Router, *, sources=None, budget=None, max_tier=None,
                components=ALL_COMPONENTS, calibrator=None, analysis=None) -> tuple[ClaimVerdict, dict]:
    """One claim through the whole pipeline: route + verify -> critic -> red flags
    -> confidence -> judge. `components` switches step-6 parts off for ablations."""
    analysis = analysis or analyze(claim.text)
    if sources is not None:
        base = router.strategy(BudgetTier.LOW)
        st = Strategy(tuple(sources), base.nli, False, 5, 4, False, 0.0, 0.0)
        v, retrieved = run.check(claim, st)
        route = {"routing_source": "forced", "passes": [{"sources": list(sources), "status": v.status.value}]}
    else:
        v, retrieved, st, route = _routed(claim, analysis, run, router, budget, max_tier)
    route["analysis"] = {"domain": analysis.domain.value, "type": analysis.question_type.value,
                         "difficulty": analysis.difficulty, "risk": analysis.risk.value,
                         "budget": analysis.compute_budget.value}
    flags = red_flags(claim, v, run.known, run.engine.store.document_text)
    critique = None
    if "critic" in components:
        v, critique = Critic().review(claim, v, retrieved, run.engine, run.verifier(st.nli, st.llm),
                                      list(st.sources), flags)
        run.sync()
        flags = red_flags(claim, v, run.known, run.engine.store.document_text)
    v.red_flags = flags
    return _finish(claim, v, critique, analysis, components, calibrator), route


def _finish(claim, v: ClaimVerdict, critique, analysis, components, calibrator) -> ClaimVerdict:
    v.critique = critique
    cal = calibrator if "calibration" in components else None
    conf = breakdown(v, critique, analysis, cal)
    threshold = (cal.threshold if cal and cal.threshold is not None else None) if "abstention" in components else 0.0
    return decide(v, conf, threshold)


def audit(text: str, *, live: bool = False, max_tier: BudgetTier | None = None, budget: BudgetTier | None = None,
          sources: list[str] | None = None, engine: EvidenceEngine | None = None, router: Router | None = None,
          llm_extractor=None, llm_verifier=None, components=ALL_COMPONENTS, calibrator=None) -> AuditReport:
    """Audit every check-worthy claim in `text`.

    live      allow PubMed / OpenAlex / web (network); otherwise corpus + graph only
    budget    force the starting tier (default: router decides per claim)
    max_tier  highest tier escalation may reach (default: high; maximum for high-risk claims)
    sources   force one fixed pass over these sources, no routing (reproducible runs)
    components  step-6 parts to run (critic, calibration, abstention); remove some for ablations
    calibrator  confidence calibrator (default: data/calibrator.json if fitted, else raw scores)
    """
    t0 = time.time()
    engine = engine or EvidenceEngine()
    run = _Run(engine, llm_verifier)
    router = router or Router(live=live, has_web=engine.web.available, has_llm=llm_verifier is not None,
                              has_nli=run.verifier(True, False).nli is not None)
    all_claims = extract_claims(text, include_skipped=True, llm=llm_extractor)
    claims = [c for c in all_claims if c.check_worthiness >= CHECK_WORTHY_THRESHOLD]
    report = AuditReport(text=text, claims=claims,
                         skipped=[{"text": c.text, "span": c.span, "check_worthiness": c.check_worthiness}
                                  for c in all_claims if c.check_worthiness < CHECK_WORTHY_THRESHOLD])
    calibrator = Calibrator.load() if calibrator is None else calibrator
    by_sentence: dict[tuple, ClaimVerdict] = {}
    deferred: list[Claim] = []
    all_sources: set[str] = set()
    for claim in claims:
        if _certainty_only(claim):
            deferred.append(claim)
            continue
        v, route = check_claim(claim, run, router, sources=sources, budget=budget, max_tier=max_tier,
                               components=components, calibrator=calibrator)
        all_sources |= {src for p in route["passes"] for src in p["sources"]}
        report.routing[claim.id] = route
        report.verdicts.append(v)
        by_sentence.setdefault(claim.span, v)
    for claim in deferred:
        effect = by_sentence.get(claim.span)
        if effect is None:              # nothing else in the sentence: verify the bare claim
            effect = run.check(claim, router.strategy(BudgetTier.LOW))[0]
        v = _certainty_verdict(claim, effect.model_copy(update={"status": effect.leaning or effect.status}))
        v.red_flags = red_flags(claim, v, run.known, engine.store.document_text)
        v = _finish(claim, v, None, analyze(claim.text), components, calibrator)
        report.verdicts.append(v)
        report.routing[claim.id] = {"routing_source": "via effect claim", "passes": []}
    order = {c.id: i for i, c in enumerate(claims)}
    report.verdicts.sort(key=lambda v: order[v.claim_id])
    used = {i.source.id for v in report.verdicts for i in v.supporting + v.contradicting}
    report.sources = {sid: run.known[sid] for sid in used if sid in run.known}
    report.retrieval_sources = sorted(all_sources)
    report.verifiers = sorted(run.used, key=lambda x: ("rules", "nli", "llm").index(x.split(":")[0]))
    if LIVE_ONLY & all_sources:
        report.errors = list(engine.science.errors)
    report.latency_s = round(time.time() - t0, 3)
    return report


def _routed(claim: Claim, analysis, run: _Run, router: Router, budget, max_tier):
    """Start at the routed tier; climb while the verdict is weak and the cap allows."""
    cap = router.cap(analysis, max_tier)
    if budget is not None:
        tier, source, p = budget, "forced", 1.0
    else:
        tier, source, p = router.start_tier(analysis, cap)
    passes = []
    while True:
        st = router.strategy(tier)
        t = time.time()
        v, retrieved = run.check(claim, st)
        passes.append({"tier": tier.value, "sources": list(st.sources), "nli": st.nli, "llm": st.llm,
                       "status": v.status.value, "confidence": v.confidence, "latency_s": round(time.time() - t, 3)})
        reason = router.should_escalate(v, tier, cap)
        if not reason:
            break
        passes[-1]["escalated_because"] = reason
        tier = router.next_useful_tier(tier, cap)
    return v, retrieved, st, {"routing_source": source, "propensity": p, "start_tier": passes[0]["tier"],
                              "final_tier": tier.value, "cap": cap.value, "passes": passes}


# --------------------------------------------------------------------------- CLI
ICON = {"supported": "✓ SUPPORTED", "partially_supported": "◐ PARTIALLY SUPPORTED", "contradicted": "✗ CONTRADICTED",
        "unverified": "? UNVERIFIED", "inferred": "→ INFERRED", "speculative": "~ SPECULATIVE"}


def render(report: AuditReport) -> str:
    claims = {c.id: c for c in report.claims}
    lines = []
    for v in report.verdicts:
        c = claims[v.claim_id]
        lines.append(f"\n  {ICON[v.status.value]:24} \"{c.text}\"")
        lines.append(f"      {short(v.rationale, 400)}")
        if v.supporting_source_count:
            lines.append(f"      Echo check: {v.supporting_source_count} supporting source(s) -> "
                         f"{v.independent_support_count} independent.")
        for f in v.red_flags:
            lines.append(f"      ⚑ {f.kind.replace('_', ' ')}: {f.message}")
        cd = v.confidence_detail
        if cd:
            lines.append(f"      Confidence: {cd.label} ({cd.calibrated:.2f}); decision: {v.decision.value}"
                         + (f"; leans {v.leaning.value.replace('_', ' ')}" if v.leaning else ""))
        if v.critique:
            for f in v.critique.findings:
                if f.kind in ("counterevidence", "unsupported_number"):
                    lines.append(f"      Critic: {f.description}")
        route = report.routing.get(v.claim_id, {})
        if route.get("passes") and route.get("routing_source") != "forced":
            steps = " -> ".join(p["tier"] + (f" ({p['escalated_because']})" if p.get("escalated_because") else "")
                                for p in route["passes"])
            lines.append(f"      Route [{route['routing_source']}]: {steps}")
    if report.skipped:
        lines.append("\n  Not checked (opinion, question, instruction or filler): "
                     + "; ".join(f"\"{s['text']}\"" for s in report.skipped))
    if report.errors:
        lines.append("\n  Retrieval errors: " + "; ".join(report.errors))
    lines.append(f"\n  Verifiers: {', '.join(report.verifiers)}")
    lines.append(f"  {report.disclaimer}  ({report.latency_s:.1f}s)")
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    live = "--live" in argv
    cap = next((BudgetTier(a.split("=", 1)[1]) for a in argv if a.startswith("--max-tier=")), None)
    text = " ".join(a for a in argv if not a.startswith("--")) or sys.stdin.read()
    print(render(audit(text, live=live, max_tier=cap)))


if __name__ == "__main__":
    main(sys.argv[1:])
