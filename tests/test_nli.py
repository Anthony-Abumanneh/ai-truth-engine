"""Step 4.5: local NLI model + fusion with the rules.

`fuse` is tested without the model. Tests that need the model skip when it
hasn't been downloaded (python -m truth_engine.pipeline.nli download)."""
import pytest

from truth_engine.pipeline.audit import audit
from truth_engine.pipeline.nli import load
from truth_engine.pipeline.verifier import RuleStance, Verifier, fuse
from truth_engine.retrieval import EvidenceEngine
from truth_engine.schemas import EvidenceStatus as S
from truth_engine.schemas import Stance

P = lambda e, n, c: {"entailment": e, "neutral": n, "contradiction": c}  # noqa: E731


# ------------------------------------------------------------------ fusion rules (no model needed)
def test_hard_rule_signal_beats_model():
    rule = RuleStance(Stance.CONTRADICTS, 0.6, "different value", "value")
    assert fuse(rule, P(0.95, 0.03, 0.02))[0] == Stance.CONTRADICTS


def test_model_decides_soft_cases():
    rule = RuleStance(Stance.ENTAILS, 0.8, "overlap", "overlap")
    assert fuse(rule, P(0.0, 0.05, 0.95))[0] == Stance.CONTRADICTS        # implied contradiction
    assert fuse(rule, P(0.0, 0.995, 0.005))[0] == Stance.NEUTRAL          # shares words, says nothing
    st, strength, _ = fuse(rule, P(0.35, 0.33, 0.32))                     # model unsure: rules, weakened
    assert st == Stance.PARTIAL and strength < 0.8


def test_world_knowledge(nli):
    p = nli.predict([("Michael Collins remained in lunar orbit aboard the command module Columbia.",
                        "Michael Collins walked on the Moon.")])[0]
    assert p["contradiction"] > 0.9          # the 8-bit ONNX export (step 4.5) got this wrong


def test_low_overlap_sentences():
    rule = RuleStance(Stance.NEUTRAL, 0.0, "off-topic", "offtopic")
    assert fuse(rule, P(0.9, 0.05, 0.05))[0] == Stance.ENTAILS            # paraphrase rescued
    assert fuse(rule, P(0.05, 0.5, 0.45))[0] == Stance.NEUTRAL            # likely a different event
    st, strength, _ = fuse(rule, P(0.0, 0.02, 0.98))
    assert st == Stance.CONTRADICTS and strength < 0.7                    # only when very sure, reduced


# ------------------------------------------------------------------ with the real model
@pytest.fixture(scope="module")
def nli():
    model = load()
    if model is None:
        pytest.skip("NLI model not downloaded")
    return model


def test_model_basic_inference(nli):
    out = nli.predict([("Penicillin was discovered in 1928 by Alexander Fleming.",
                        "Penicillin was discovered by Louis Pasteur."),
                       ("A man is playing a guitar on stage.", "A man is playing an instrument.")])
    assert out[0]["contradiction"] > 0.9 and out[1]["entailment"] > 0.9


def test_answer_does_not_depend_on_batch(nli):
    a = ("Michael Collins remained in lunar orbit aboard the command module Columbia.",
         "Michael Collins walked on the Moon.")
    b = ("Apollo 11 was the spaceflight that first landed humans on the Moon, and the commander and the lunar "
         "module pilot landed on 20 July 1969 near the Sea of Tranquility.", "Michael Collins walked on the Moon.")
    alone = nli.predict([a])[0]
    nli._memo.clear()
    got = nli.predict([b, a])[1]                      # padded next to a longer pair
    assert all(abs(got[k] - alone[k]) < 1e-3 for k in alone)


@pytest.mark.parametrize("text,want", [
    ("Veridamine improves sleep quality by 18%.", S.CONTRADICTED),
    ("Einstein won the Nobel Prize in Physics for the theory of relativity.", S.CONTRADICTED),
    ("The Halcyon X1 was discontinued in 2024.", S.SUPPORTED),
    ("Penicillin was discovered by Louis Pasteur.", S.CONTRADICTED),
    ("Python was first released in 1991.", S.SUPPORTED),
    ("The Halcyon X1 was discontinued in 2024 because demand declined.", S.CONTRADICTED),
])
def test_audit_with_model(nli, text, want):
    eng = EvidenceEngine()
    rep = audit(text, engine=eng)
    # audit() builds its own Verifier; check the model path explicitly too
    claim = rep.claims[-1]
    v = Verifier(dict(eng.store.sources), nli=nli).verify(claim, eng.gather(claim, eng.plan(claim, ["corpus", "graph"])))
    assert v.status == want
    assert v.verifier_agreement is not None
