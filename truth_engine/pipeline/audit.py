"""Audit mode: text in, every check-worthy claim verified.

    text -> extract claims -> for each claim: plan support + falsification
    queries -> gather evidence -> verify (stance, lineages, status) -> red flags

    python -m truth_engine.pipeline.audit "Veridamine improves sleep by 18% and is clinically proven."
    python -m truth_engine.pipeline.audit --live "Creatine improves memory in older adults."

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
from .verifier import Verifier, study_of

OFFLINE_SOURCES = ["corpus", "graph"]
LIVE_SOURCES = ["corpus", "graph", "pubmed", "openalex"]


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


def audit(text: str, *, sources: list[str] | None = None, engine: EvidenceEngine | None = None,
          llm_extractor=None, llm_verifier=None) -> AuditReport:
    t0 = time.time()
    engine = engine or EvidenceEngine()
    sources = sources or OFFLINE_SOURCES
    all_claims = extract_claims(text, include_skipped=True, llm=llm_extractor)
    claims = [c for c in all_claims if c.check_worthiness >= CHECK_WORTHY_THRESHOLD]
    report = AuditReport(text=text, claims=claims, retrieval_sources=sources,
                         skipped=[{"text": c.text, "span": c.span, "check_worthiness": c.check_worthiness}
                                  for c in all_claims if c.check_worthiness < CHECK_WORTHY_THRESHOLD])
    known = dict(engine.store.sources)
    verifier = Verifier(known, llm=llm_verifier,
                        context_fn=lambda pid: " ".join(p.text for p in engine.store.neighbors(pid)))
    report.verifiers = ["rules"] + ([f"nli:{verifier.nli.name}"] if verifier.nli else []) + (
        [f"llm:{llm_verifier[1].name}"] if llm_verifier else [])
    by_sentence: dict[tuple, ClaimVerdict] = {}
    deferred: list[Claim] = []
    for claim in claims:
        if _certainty_only(claim):
            deferred.append(claim)
            continue
        retrieved = engine.gather(claim, engine.plan(claim, sources))
        known.update(engine.last_external_sources)
        if "pubmed" in sources or "openalex" in sources:
            known.update(engine.science.sources)
        v = verifier.verify(claim, retrieved)
        v.red_flags = red_flags(claim, v, known, engine.store.document_text)
        report.verdicts.append(v)
        by_sentence.setdefault(claim.span, v)
    for claim in deferred:
        effect = by_sentence.get(claim.span)
        if effect is None:              # nothing else in the sentence: verify the bare claim
            effect = verifier.verify(claim, engine.gather(claim, engine.plan(claim, sources)))
        v = _certainty_verdict(claim, effect)
        v.red_flags = red_flags(claim, v, known, engine.store.document_text)
        report.verdicts.append(v)
    order = {c.id: i for i, c in enumerate(claims)}
    report.verdicts.sort(key=lambda v: order[v.claim_id])
    used = {i.source.id for v in report.verdicts for i in v.supporting + v.contradicting}
    report.sources = {sid: known[sid] for sid in used if sid in known}
    if "pubmed" in sources or "openalex" in sources:
        report.errors = list(engine.science.errors)
    report.latency_s = round(time.time() - t0, 3)
    return report


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
    text = " ".join(a for a in argv if a != "--live") or sys.stdin.read()
    print(render(audit(text, sources=LIVE_SOURCES if live else OFFLINE_SOURCES)))


if __name__ == "__main__":
    main(sys.argv[1:])
