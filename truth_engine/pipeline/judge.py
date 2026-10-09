"""Judge: answer, answer with caveats, or abstain (spec §4, Improvements 4 and 9;
CONCEPTS.md §8: selective prediction).

Audit mode: a decisive verdict (supported / contradicted / partial / inferred)
is only *asserted* when its calibrated confidence clears a threshold. Below
it, the judge abstains: the claim is shown as UNVERIFIED with what the
evidence leaned towards. The threshold is chosen on training data to maximize
accuracy (see scripts/calibrate.py), which trades coverage for reliability.

Ask mode: candidates are ranked by how well their statements *verified*, not
by how many solvers proposed them (majority vote fails when solvers share a
blind spot), with agreement as a small tie-breaker.
"""
from __future__ import annotations

from ..schemas import (CandidateAnswer, ClaimVerdict, ConfidenceBreakdown, Decision, DiversityReport,
                       EvidenceStatus, JudgeResult)

DEFAULT_THRESHOLD = 0.5
CAVEAT_SEVERITY = 0.5


def decide(verdict: ClaimVerdict, conf: ConfidenceBreakdown, threshold: float | None = None) -> ClaimVerdict:
    """Return the verdict as it should be presented, with `decision` set."""
    thr = DEFAULT_THRESHOLD if threshold is None else threshold
    v = verdict.model_copy()
    v.confidence_detail, v.confidence = conf, conf.calibrated
    if v.status == EvidenceStatus.UNVERIFIED or v.conflicting:
        v.decision = Decision.ABSTAIN
        return v
    if conf.calibrated < thr:
        v.decision, v.leaning, v.status = Decision.ABSTAIN, v.status, EvidenceStatus.UNVERIFIED
        v.rationale = (f"Leans {v.leaning.value.replace('_', ' ')}, but the confidence ({conf.calibrated:.2f}) is "
                       f"below the bar to state it. " + v.rationale)
        return v
    serious = any(f.severity >= CAVEAT_SEVERITY for f in v.red_flags)
    v.decision = Decision.ANSWER_WITH_CAVEATS if serious or conf.label in ("LOW", "MODERATE") else Decision.ANSWER
    return v


def judge_candidates(candidates: list[CandidateAnswer], verified: dict[str, ClaimVerdict],
                     diversity: DiversityReport | None = None, threshold: float | None = None) -> JudgeResult:
    """Pick the candidate whose statement verified best; abstain if none clears the bar."""
    thr = DEFAULT_THRESHOLD if threshold is None else threshold
    share = {}
    if diversity:
        for cluster in diversity.answer_clusters:
            for cid in cluster:
                share[cid] = len(cluster) / max(1, diversity.n_candidates)
    scored, rejected = [], []
    for c in candidates:
        v = verified.get(c.id)
        if v is None:
            continue
        support = v.confidence if v.status in (EvidenceStatus.SUPPORTED, EvidenceStatus.PARTIALLY_SUPPORTED,
                                               EvidenceStatus.INFERRED) else 0.0
        score = support + 0.05 * share.get(c.id, 0)
        scored.append((score, c, v))
        if support < thr:
            rejected.append({"candidate": c.id, "model": c.model,
                             "reason": f"{v.status.value} (confidence {v.confidence:.2f})"})
    if not scored:
        return JudgeResult(model="judge-rules", selected_candidate_id=None, decision=Decision.ABSTAIN,
                           rationale="No candidate could be verified.", reliability_requirements_met=False)
    score, best, v = max(scored, key=lambda t: t[0])
    if score - 0.05 * share.get(best.id, 0) < thr:
        return JudgeResult(model="judge-rules", selected_candidate_id=None, decision=Decision.ABSTAIN,
                           rationale=f"Best candidate ({best.model}) only reached {v.status.value} at confidence "
                                     f"{v.confidence:.2f}.", rejected=rejected, reliability_requirements_met=False)
    caveats = bool(v.red_flags) or v.confidence < 0.7
    return JudgeResult(model="judge-rules", selected_candidate_id=best.id,
                       decision=Decision.ANSWER_WITH_CAVEATS if caveats else Decision.ANSWER,
                       rationale=f"Selected {best.model}: {v.status.value} at confidence {v.confidence:.2f}.",
                       rejected=[r for r in rejected if r["candidate"] != best.id], reliability_requirements_met=True)
