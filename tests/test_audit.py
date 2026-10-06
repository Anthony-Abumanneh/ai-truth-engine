"""Step 4: audit mode - claim extraction, stance, verdicts, red flags.
All offline, against the demo corpus (sources: corpus + graph)."""
import pytest

from truth_engine.pipeline.audit import audit
from truth_engine.pipeline.claims import extract_claims, sentence_spans
from truth_engine.pipeline.verifier import Verifier, negated_in_scope, stance
from truth_engine.retrieval import EvidenceEngine
from truth_engine.schemas import EvidenceStatus as S
from truth_engine.schemas import Stance
from truth_engine.textutils import content_tokens


@pytest.fixture(scope="module")
def engine():
    return EvidenceEngine()


def _status(text, engine):
    return audit(text, engine=engine).verdicts[0]


# ------------------------------------------------------------------ claims
def test_spec_example_decomposes_into_atomic_claims():
    claims = extract_claims("The company launched Product X in 2022, sold 4 million units, "
                            "and discontinued it in 2024 because demand declined.")
    assert [c.text for c in claims] == [
        "The company launched Product X in 2022.", "The company sold 4 million units.",
        "The company discontinued it in 2024.", "The company discontinued it in 2024 because demand declined."]
    assert claims[-1].is_causal and not claims[0].is_causal


def test_noun_lists_are_not_split_and_pronouns_resolved():
    claims = extract_claims("Creatine improves memory and focus in older adults. It also reduces fatigue.")
    assert [c.text for c in claims] == ["Creatine improves memory and focus in older adults.",
                                        "Creatine also reduces fatigue."]


def test_check_worthiness_skips_opinions_questions_instructions():
    text = "Veridamine improves sleep quality by 18%. I love it! Is it safe? Talk to your doctor first."
    kept = extract_claims(text)
    assert [c.text for c in kept] == ["Veridamine improves sleep quality by 18%."]
    assert len(extract_claims(text, include_skipped=True)) == 4


def test_spans_point_into_original_text():
    text = "First claim here.   Veridamine improves sleep\nquality by 18%."
    c = extract_claims(text)[-1]
    assert text[c.span[0]:c.span[1]] == "Veridamine improves sleep\nquality by 18%."
    assert len(sentence_spans(text)) == 2


def test_claim_tagging():
    by = {c.text: c for c in extract_claims(
        "Brivo tea drinkers are less likely to develop dementia. Zantorel cuts stroke risk in half. "
        "Calvetrin extends lifespan in mice. High doses of vitamin C are safe. Lumorex may improve memory.")}
    assert by["Brivo tea drinkers are less likely to develop dementia."].science_kind == "association"
    assert by["Zantorel cuts stroke risk in half."].science_kind == "efficacy"
    assert by["Calvetrin extends lifespan in mice."].population == "animal"
    assert by["High doses of vitamin C are safe."].science_kind == "safety"
    assert by["Lumorex may improve memory."].hedged


# ------------------------------------------------------------------ stance
def _claim(text):
    return extract_claims(text, include_skipped=True)[0]


def test_stance_signals():
    c = _claim("Veridamine improves sleep quality by 18%.")
    assert stance(c, "In a trial, veridamine improved sleep quality by 18% compared with placebo.")[0] == Stance.ENTAILS
    assert stance(c, "Veridamine did not significantly improve sleep quality (difference 2%).")[0] == Stance.CONTRADICTS
    assert stance(c, "Veridamine improved sleep quality by 2%.")[0] == Stance.CONTRADICTS
    assert stance(c, "Evidence that veridamine improves sleep quality is of low certainty.")[0] == Stance.CONTRADICTS
    assert stance(c, "The Eiffel Tower is in Paris.")[0] == Stance.NEUTRAL


def test_negation_scope():
    claim_tokens = set(content_tokens("Calvetrin extends lifespan in mice"))
    assert not negated_in_scope("The study was in mice only, and no human data on calvetrin and lifespan exist.",
                                claim_tokens, "Calvetrin")
    assert negated_in_scope("The Nobel Prize was not awarded for the theory of relativity.",
                            set(content_tokens("won the Nobel Prize for the theory of relativity")), "Einstein")


def test_years_must_match_exactly():
    c = _claim("Python was first released in 1995.")
    assert stance(c, "Python was first released in 1991.")[0] == Stance.CONTRADICTS


# ------------------------------------------------------------------ verdicts
def test_echo_chamber_claim_is_contradicted(engine):
    v = _status("Veridamine improves sleep quality by 18%.", engine)
    assert v.status == S.CONTRADICTED
    assert v.supporting_source_count >= 3 and v.independent_support_count == 1
    kinds = {f.kind for f in v.red_flags}
    assert {"echo_chamber", "failed_replication", "industry_funded", "small_sample"} <= kinds


def test_clinically_proven_judged_through_its_effect(engine):
    rep = audit("Veridamine improves sleep quality by 18% and is clinically proven.", engine=engine)
    proven = rep.verdicts[1]
    assert proven.status == S.CONTRADICTED and "overclaimed_certainty" in {f.kind for f in proven.red_flags}


def test_retracted_support_counts_for_nothing(engine):
    v = _status("Lumorex reverses memory loss in older adults by 40%.", engine)
    assert v.status == S.UNVERIFIED and v.support_score == 0
    assert v.red_flags[0].kind == "retracted"


def test_generalization_flags(engine):
    assert "animal_to_human" in {f.kind for f in _status("Calvetrin extends lifespan by 30%.", engine).red_flags}
    assert "in_vitro_to_human" in {f.kind for f in _status("Nervaplex grows new brain cells.", engine).red_flags}
    v = _status("Drinking brivo tea prevents dementia.", engine)
    assert v.status != S.SUPPORTED and "correlation_as_causation" in {f.kind for f in v.red_flags}


def test_relative_risk_flag_quotes_absolute_risk(engine):
    v = _status("Zantorel cuts stroke risk in half.", engine)
    rr = next(f for f in v.red_flags if f.kind == "relative_risk_only")
    assert "0.2 percentage points" in rr.message


@pytest.mark.parametrize("text,want", [
    ("Einstein won the Nobel Prize in Physics for the theory of relativity.", S.CONTRADICTED),
    ("Einstein won the Nobel Prize in Physics in 1921.", S.SUPPORTED),
    ("The capital of Australia is Sydney.", S.CONTRADICTED),
    ("The capital of Australia is Canberra.", S.SUPPORTED),
    ("Penicillin was discovered by Louis Pasteur.", S.CONTRADICTED),
    ("Python was first released in 1991.", S.SUPPORTED),
    ("Python was first released in 1995.", S.CONTRADICTED),
    ("The speed of light in vacuum is 150,000 kilometres per second.", S.CONTRADICTED),
    ("The Great Wall of China is visible from the Moon with the naked eye.", S.CONTRADICTED),
    ("The Halcyon X1 was discontinued in 2024 because demand declined.", S.CONTRADICTED),
    ("Calvetrin extends lifespan in mice.", S.SUPPORTED),
])
def test_verdicts_on_demo_corpus(engine, text, want):
    # last claim: for "A because B" that's the causal claim
    assert audit(text, engine=engine).verdicts[-1].status == want


def test_overclaim_against_disputed_fact_is_not_supported(engine):
    # rules: UNVERIFIED (evidence both ways); with the NLI model: CONTRADICTED ("disputed" vs "definitely")
    v = audit("The Nile is definitely the longest river in the world.", engine=engine).verdicts[0]
    assert v.status in (S.UNVERIFIED, S.CONTRADICTED)
    assert "overclaimed_certainty" in {f.kind for f in v.red_flags}


def test_audit_report_shape(engine):
    rep = audit("Veridamine improves sleep quality by 18%. I love it!", engine=engine)
    assert len(rep.claims) == 1 and rep.skipped[0]["text"] == "I love it!"
    assert "okafor-2024" in rep.sources and "Not medical advice" in rep.disclaimer


# ------------------------------------------------------------------ optional LLM verifier
class _FakeLLM:
    """Stands in for a real provider: always answers 'contradicts'."""
    def complete_json(self, spec, system, user, **kw):
        from truth_engine.providers import LLMResponse
        return {"stance": "contradicts", "reason": "test"}, LLMResponse(text="", model="fake", cost_usd=0.001)


def test_llm_verifier_second_opinion_and_agreement(engine):
    claim = _claim("Penicillin was discovered by Alexander Fleming.")
    retrieved = engine.gather(claim, engine.plan(claim, ["corpus"]))
    v = Verifier(dict(engine.store.sources), llm=(_FakeLLM(), None)).verify(claim, retrieved)
    assert v.status == S.CONTRADICTED            # LLM stance is used when present
    assert v.verifier_agreement is not None and v.verifier_agreement < 0.5
