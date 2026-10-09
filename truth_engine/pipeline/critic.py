"""Adversarial critic (spec §4, Improvement 4; CONCEPTS.md §7: falsification,
devil's advocate).

The critic assumes every verdict is wrong and tries to prove it:

  1. targeted search: if the verdict leans *supported*, search again for
     counterevidence (null results, failed replications, reviews); if it
     leans *contradicted*, search for the strongest support. The verdict is
     recomputed with the old and new evidence together.
  2. stronger design on the losing side: a supported claim whose best
     counterevidence comes from a clearly stronger study design (or vice
     versa) is reported, and the verdict becomes UNVERIFIED (conflicting).
  3. unsupported numbers: "improves X by 18%" can't be SUPPORTED if no
     supporting source states 18%; it becomes PARTIALLY SUPPORTED.
  4. red flags and alternative explanations become findings that lower
     confidence (they don't change the verdict on their own).

Every change is recorded as a finding, so the UI can show what the critic
did and why. Ablations switch it off to measure whether it helps.
"""
from __future__ import annotations

from ..retrieval.evidence import EvidencePlan, Retrieved
from ..schemas import (Claim, ClaimVerdict, CritiqueFinding, CritiqueReport, EvidenceStatus, RedFlag, Stance)
from ..textutils import quantities
from .verifier import CAUSE_DENIAL, LIMITATION, Verifier, study_of

LEANS_SUPPORT = {EvidenceStatus.SUPPORTED, EvidenceStatus.PARTIALLY_SUPPORTED, EvidenceStatus.INFERRED,
                 EvidenceStatus.SPECULATIVE}
DESIGN_GAP = 0.2            # how much stronger (evidence-hierarchy weight) the losing side must be

FLAG_TO_FINDING = {
    "correlation_as_causation": "correlation_not_causation", "animal_to_human": "scope_overreach",
    "in_vitro_to_human": "scope_overreach", "relative_risk_only": "missing_condition",
    "overclaimed_certainty": "scope_overreach", "retracted": "counterevidence", "failed_replication": "counterevidence",
    "single_study": "hidden_assumption", "small_sample": "hidden_assumption", "industry_funded": "hidden_assumption",
    "echo_chamber": "hidden_assumption",
}


def _best_design(items) -> tuple[float, str | None]:
    best, label = 0.0, None
    for i in items:
        st = study_of(i.source)
        if st.get("weight", 0) > best and not st.get("retracted"):
            best, label = st["weight"], i.source.metadata.get("study_label")
    return best, label


class Critic:
    def __init__(self, extra_k: int = 6):
        self.extra_k = extra_k

    def attack_queries(self, claim: Claim, verdict: ClaimVerdict) -> tuple[list[str], list[str]]:
        """(support queries, falsification queries) aimed at overturning this verdict."""
        subject = claim.entities[0] if claim.entities else " ".join(claim.text.split()[:3])
        if verdict.status in LEANS_SUPPORT:
            return [], [f"{subject} no significant effect placebo", f"{subject} failed to replicate",
                        f"{subject} systematic review meta-analysis evidence", f"{claim.text} not"]
        if verdict.status == EvidenceStatus.CONTRADICTED:
            return [f"{claim.text}", f"{subject} randomized trial improved significant",
                    f"{subject} meta-analysis benefit"], []
        return [f"{subject} randomized controlled trial", f"{subject} meta-analysis"], \
            [f"{subject} no effect"]

    def review(self, claim: Claim, verdict: ClaimVerdict, retrieved: list[Retrieved], engine, verifier: Verifier,
               sources: list[str], red_flags: list[RedFlag] | None = None) -> tuple[ClaimVerdict, CritiqueReport]:
        findings: list[CritiqueFinding] = []
        before = verdict.status

        # 1. targeted search, then re-verify with everything
        support_q, falsify_q = self.attack_queries(claim, verdict)
        plan = EvidencePlan(claim.id, support_q, falsify_q, sources)
        extra = engine.gather(claim, plan, k_support=self.extra_k if support_q else 0,
                              k_falsify=self.extra_k if falsify_q else 0)
        seen = {r.passage.id for r in retrieved}
        new = [r for r in extra if r.passage.id not in seen]
        if new:
            verdict = verifier.verify(claim, retrieved + new)
            if verdict.status != before:
                side = verdict.contradicting if before in LEANS_SUPPORT else verdict.supporting
                findings.append(CritiqueFinding(
                    kind="counterevidence", target_claim_id=claim.id, severity=0.7, evidence=side[:2],
                    description=f"Targeted search changed the verdict from {before.value} to {verdict.status.value}."))

        # 2. stronger design on the losing side
        if verdict.status in (EvidenceStatus.SUPPORTED, EvidenceStatus.CONTRADICTED):
            win, lose = (verdict.supporting, verdict.contradicting) if verdict.status == EvidenceStatus.SUPPORTED \
                else (verdict.contradicting, verdict.supporting)
            w_best, w_label = _best_design(win)
            l_best, l_label = _best_design(lose)
            losing_score = verdict.contradiction_score if verdict.status == EvidenceStatus.SUPPORTED \
                else verdict.support_score
            if l_best >= w_best + DESIGN_GAP and losing_score >= 0.3:
                findings.append(CritiqueFinding(
                    kind="counterevidence", target_claim_id=claim.id, severity=0.7, evidence=lose[:2],
                    description=f"The other side has a stronger study design ({l_label}) than the evidence for "
                                f"this verdict ({w_label or 'no graded study'})."))
                verdict = verdict.model_copy(update={"status": EvidenceStatus.UNVERIFIED, "conflicting": True})

        # 3. numbers the claim states that no supporting source states
        if verdict.status == EvidenceStatus.SUPPORTED and any(q.unit not in ("year", "date")
                                                              for q in quantities(claim.text)):
            backed = any("same value" in i.rationale for i in verdict.supporting if i.stance == Stance.ENTAILS)
            if not backed:
                findings.append(CritiqueFinding(
                    kind="unsupported_number", target_claim_id=claim.id, severity=0.5,
                    description="The claim's number is not stated by any supporting source."))
                verdict = verdict.model_copy(update={"status": EvidenceStatus.PARTIALLY_SUPPORTED})

        # 4. red flags and alternative explanations
        for f in red_flags or []:
            findings.append(CritiqueFinding(kind=FLAG_TO_FINDING.get(f.kind, "hidden_assumption"),
                                            target_claim_id=claim.id, severity=f.severity, description=f.message))
        alternatives = []
        if claim.is_causal:
            for i in verdict.contradicting + verdict.supporting:
                if CAUSE_DENIAL.search(i.passage.text) or LIMITATION.search(i.passage.text):
                    alternatives.append(i.passage.text)
        if alternatives:
            findings.append(CritiqueFinding(kind="alternative_explanation", target_claim_id=claim.id, severity=0.4,
                                            description="The evidence mentions other explanations: "
                                                        + " | ".join(alternatives[:2])))
        changed = verdict.status != before
        report = CritiqueReport(
            model="critic-rules", findings=findings, survived=not changed,
            confidence_adjustment=-round(min(0.5, sum(f.severity for f in findings) * 0.1), 3),
            alternative_hypotheses=alternatives[:3])
        return verdict, report
