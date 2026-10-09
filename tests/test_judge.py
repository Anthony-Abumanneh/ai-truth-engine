"""Step 6: critic, confidence + calibration, judge / abstention."""
import random

import numpy as np
import pytest

from truth_engine.pipeline.audit import audit
from truth_engine.pipeline.confidence import (Calibrator, brier, cross_val_brier, ece, history_from, label,
                                              raw_score, reliability, signals)
from truth_engine.pipeline.judge import decide, judge_candidates
from truth_engine.retrieval import EvidenceEngine
from truth_engine.schemas import (CandidateAnswer, ClaimVerdict, ConfidenceBreakdown, Decision, EvidenceStatus as S,
                                  RedFlag)


@pytest.fixture(scope="module")
def engine():
    return EvidenceEngine()


def _conf(p):
    sig = dict(evidence_support=0.5, contradiction=0.1, source_quality=0.8, source_independence=0.5,
               verifier_agreement=0.5, model_agreement=0.5, historical_reliability=0.5, unresolved_uncertainty=0.0,
               difficulty_penalty=0.3)
    return ConfidenceBreakdown(**sig, raw_score=p, calibrated=p, label=label(p))


# ------------------------------------------------------------------ judge
def test_judge_abstains_below_threshold_and_keeps_leaning():
    v = ClaimVerdict(claim_id="c", status=S.SUPPORTED, confidence=0.0, rationale="r")
    out = decide(v, _conf(0.4), threshold=0.6)
    assert out.decision == Decision.ABSTAIN and out.status == S.UNVERIFIED and out.leaning == S.SUPPORTED
    assert out.rationale.startswith("Leans supported")
    ok = decide(v, _conf(0.9), threshold=0.6)
    assert ok.decision == Decision.ANSWER and ok.status == S.SUPPORTED and ok.confidence == 0.9


def test_judge_adds_caveats_for_serious_red_flags():
    v = ClaimVerdict(claim_id="c", status=S.SUPPORTED, confidence=0.0,
                     red_flags=[RedFlag(kind="animal_to_human", severity=0.8, message="mice")])
    assert decide(v, _conf(0.95), threshold=0.5).decision == Decision.ANSWER_WITH_CAVEATS


def test_conflicting_evidence_always_abstains():
    v = ClaimVerdict(claim_id="c", status=S.UNVERIFIED, confidence=0.0, conflicting=True)
    assert decide(v, _conf(0.99), threshold=0.1).decision == Decision.ABSTAIN


def test_ask_judge_prefers_verified_over_majority():
    graph = CandidateAnswer(model="graph", answer="Germany", final_value="Germany")
    others = [CandidateAnswer(model=m, answer="Albert Einstein", final_value="Albert Einstein")
              for m in ("offline-swift", "offline-scholar", "offline-sage")]
    verified = {graph.id: ClaimVerdict(claim_id="g", status=S.SUPPORTED, confidence=0.8)}
    verified |= {c.id: ClaimVerdict(claim_id=c.id, status=S.UNVERIFIED, confidence=0.3) for c in others}
    res = judge_candidates([graph] + others, verified, threshold=0.5)
    assert res.selected_candidate_id == graph.id and res.reliability_requirements_met
    nothing = judge_candidates(others, verified, threshold=0.5)
    assert nothing.decision == Decision.ABSTAIN and nothing.selected_candidate_id is None


# ------------------------------------------------------------------ calibration math
def test_metrics():
    assert ece([0.9] * 10, [True] * 9 + [False]) == 0.0
    assert ece([0.9] * 10, [True] * 5 + [False] * 5) == pytest.approx(0.4)
    assert brier([1.0, 0.0], [True, False]) == 0.0
    assert reliability([0.15, 0.95], [False, True])[0]["bin"] == "0.1-0.2"


def test_calibrator_learns_and_beats_raw_on_miscalibrated_signals(tmp_path):
    """Synthetic world: verdicts with strong support are right 90% of the time,
    weak ones 30%, but the raw score is overconfident on weak ones."""
    rng = random.Random(1)
    sigs, statuses, correct = [], [], []
    for _ in range(600):
        strong = rng.random() < 0.5
        s = dict(evidence_support=0.9 if strong else 0.35, contradiction=0.05, source_quality=0.8,
                 source_independence=0.5, verifier_agreement=0.8, model_agreement=0.9, historical_reliability=0.5,
                 unresolved_uncertainty=0.0, difficulty_penalty=0.3)
        sigs.append(s)
        statuses.append("supported")
        correct.append(rng.random() < (0.9 if strong else 0.3))
    raw = [raw_score(s) for s in sigs]
    cal = Calibrator("logistic", history=history_from(statuses, correct)).fit(sigs, statuses, correct)
    probs = [cal(s, "supported") for s in sigs]
    assert ece(probs, correct) < ece(raw, correct)
    assert cross_val_brier("logistic", sigs, statuses, correct) < np.mean([(r - c) ** 2 for r, c in zip(raw, correct)])
    cal.threshold = 0.6
    cal.save(tmp_path / "c.json")
    again = Calibrator.load(tmp_path / "c.json")
    assert again.threshold == 0.6 and abs(again(sigs[0], "supported") - cal(sigs[0], "supported")) < 1e-9


# ------------------------------------------------------------------ critic + full pipeline
def test_critic_runs_and_signals_are_recorded(engine):
    rep = audit("Veridamine improves sleep quality by 18%.", engine=engine,
                components={"critic", "calibration", "abstention"})      # critic is opt-in
    v = rep.verdicts[0]
    assert v.critique is not None and v.confidence_detail is not None and v.decision is not None
    assert v.status == S.CONTRADICTED and v.confidence_detail.contradiction > 0
    kinds = {f.kind for f in v.critique.findings}
    assert "counterevidence" in kinds or "hidden_assumption" in kinds        # red flags become findings


def test_critic_downgrades_unsupported_number():
    from truth_engine.pipeline.claims import make_claim
    from truth_engine.pipeline.critic import Critic
    from truth_engine.schemas import EvidenceItem, Passage, Source, Stance

    class NoNewEvidence:
        def gather(self, *a, **k):
            return []
    src = Source(id="s", title="Python")
    item = EvidenceItem(passage=Passage(id="s#0", source_id="s", text="Python was first released in 1991."),
                        source=src, stance=Stance.ENTAILS, stance_score=0.9, relevance=0.9,
                        found_by="supporting_search", rationale="coverage 0.80")
    claim = make_claim("Python was first released in 1991 with 12% market share.")
    v = ClaimVerdict(claim_id=claim.id, status=S.SUPPORTED, confidence=0.7, supporting=[item], support_score=0.7)
    out, report = Critic().review(claim, v, [], NoNewEvidence(), verifier=None, sources=["corpus"])
    assert out.status == S.PARTIALLY_SUPPORTED and not report.survived
    assert [f.kind for f in report.findings] == ["unsupported_number"]


def test_critic_is_off_by_default(engine):
    assert audit("Veridamine improves sleep quality by 18%.", engine=engine).verdicts[0].critique is None


def test_components_can_be_switched_off(engine):
    rep = audit("Veridamine improves sleep quality by 18%.", engine=engine, components=set())
    v = rep.verdicts[0]
    assert v.critique is None and v.confidence_detail is not None


def test_signals_are_bounded(engine):
    v = audit("Einstein won the Nobel Prize in Physics in 1921.", engine=engine).verdicts[0]
    sig = signals(v, v.critique, None)
    assert all(0.0 <= x <= 1.0 for x in sig.values())
