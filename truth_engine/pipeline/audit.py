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

    def check(self, claim: Claim, st: Strategy) -> ClaimVerdict:
        eng = self.engine
        retrieved = eng.gather(claim, eng.plan(claim, list(st.sources)), k_support=st.k_support,
                               k_falsify=st.k_falsify, follow_up_studies=st.follow_up_studies)
        self.known.update(eng.last_external_sources)
        if LIVE_ONLY & set(st.sources):
            self.known.update(eng.science.sources)
        return self.verifier(st.nli, st.llm).verify(claim, retrieved)


def audit(text: str, *, live: bool = False, max_tier: BudgetTier | None = None, budget: BudgetTier | None = None,
          sources: list[str] | None = None, engine: EvidenceEngine | None = None, router: Router | None = None,
          llm_extractor=None, llm_verifier=None) -> AuditReport:
    """Audit every check-worthy claim in `text`.

    live      allow PubMed / OpenAlex / web (network); otherwise corpus + graph only
    budget    force the starting tier (default: router decides per claim)
    max_tier  highest tier escalation may reach (default: high; maximum for high-risk claims)
    sources   force one fixed pass over these sources, no routing (reproducible runs)
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
    by_sentence: dict[tuple, ClaimVerdict] = {}
    deferred: list[Claim] = []
    all_sources: set[str] = set()
    for claim in claims:
        if _certainty_only(claim):
            deferred.append(claim)
            continue
        analysis = analyze(claim.text)
        if sources is not None:
            st = router.strategy(BudgetTier.LOW)
            st = Strategy(tuple(sources), st.nli, False, 5, 4, False, 0.0, 0.0)
            v = run.check(claim, st)
            route = {"routing_source": "forced", "passes": [{"sources": list(sources), "status": v.status.value}]}
        else:
            v, route = _routed(claim, analysis, run, router, budget, max_tier)
        all_sources |= {src for p in route["passes"] for src in p["sources"]}
        route["analysis"] = {"domain": analysis.domain.value, "type": analysis.question_type.value,
                             "difficulty": analysis.difficulty, "risk": analysis.risk.value,
                             "budget": analysis.compute_budget.value}
        report.routing[claim.id] = route
        v.red_flags = red_flags(claim, v, run.known, engine.store.document_text)
        report.verdicts.append(v)
        by_sentence.setdefault(claim.span, v)
    for claim in deferred:
        effect = by_sentence.get(claim.span)
        if effect is None:              # nothing else in the sentence: verify the bare claim
            effect = run.check(claim, router.strategy(BudgetTier.LOW))
        v = _certainty_verdict(claim, effect)
        v.red_flags = red_flags(claim, v, run.known, engine.store.document_text)
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


def _routed(claim: Claim, analysis, run: _Run, router: Router, budget, max_tier) -> tuple[ClaimVerdict, dict]:
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
        v = run.check(claim, st)
        passes.append({"tier": tier.value, "sources": list(st.sources), "nli": st.nli, "llm": st.llm,
                       "status": v.status.value, "confidence": v.confidence, "latency_s": round(time.time() - t, 3)})
        reason = router.should_escalate(v, tier, cap)
        if not reason:
            break
        passes[-1]["escalated_because"] = reason
        tier = router.next_useful_tier(tier, cap)
    return v, {"routing_source": source, "propensity": p, "start_tier": passes[0]["tier"], "final_tier": tier.value,
               "cap": cap.value, "passes": passes}


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
        lines.append(f"      Provisional confidence: {v.confidence:.2f}")
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
