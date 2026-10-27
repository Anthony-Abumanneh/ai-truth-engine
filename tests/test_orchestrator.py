"""Step 7: orchestrator (audit + ask), evidence tree, trace store, watched claims, feedback learning."""
import pytest

from truth_engine.feedback.learning import learn_from_feedback, outcomes, same_verdict
from truth_engine.feedback.traces import TraceStore
from truth_engine.pipeline.orchestrator import Orchestrator, to_statement
from truth_engine.pipeline.router import BanditPolicy
from truth_engine.pipeline.solvers import answer_type_ok
from truth_engine.retrieval import EvidenceEngine
from truth_engine.schemas import Decision


@pytest.fixture(scope="module")
def engine():
    return EvidenceEngine()


@pytest.fixture
def orch(engine, tmp_path):
    return Orchestrator(engine=engine, store=TraceStore(tmp_path / "t.sqlite"))


@pytest.mark.parametrize("q,a,stmt", [
    ("Who discovered penicillin?", "Alexander Fleming", "Alexander Fleming discovered penicillin."),
    ("What is the capital of Australia?", "Canberra", "The capital of Australia is Canberra."),
    ("In which country was the person who developed general relativity born?", "Germany",
     "The person who developed general relativity was born in Germany."),
    ("Where was Einstein born?", "Ulm", "Einstein was born in Ulm."),
    ("How tall is the Eiffel Tower?", "330 metres", "The Eiffel Tower is 330 metres tall."),
])
def test_to_statement(q, a, stmt):
    assert to_statement(q, a) == stmt


def test_answer_type_check():
    q = "In which country was the person who developed general relativity born?"
    assert answer_type_ok(q, "Germany") and not answer_type_ok(q, "Albert Einstein")
    assert answer_type_ok(q, "Some Unknown Place")          # unknown entities are not rejected


def test_audit_trace_events_tree_and_storage(orch):
    seen = []
    t = orch.audit("Veridamine improves sleep quality by 18%. I love it!", on_event=seen.append)
    assert t.mode == "audit" and len(t.verdicts) == 1 and t.skipped
    assert [e.stage for e in seen][:2] == ["audit", "extract"] and seen[-1].status == "completed"
    claim_node = t.tree.children[0]
    relations = {c.relation for c in claim_node.children}
    assert {"supports", "contradicts", "rejected"} <= relations
    echo = next(c for c in claim_node.children if c.relation == "supports")
    assert len(echo.children) >= 2                           # the pilot's copies sit under one lineage
    again = orch.store.get_trace(t.query_id)
    assert again.verdicts[0].status == t.verdicts[0].status
    rec = orch.store.export_section14()[0]
    assert rec["query_id"] == t.query_id and rec["claims"] == ["Veridamine improves sleep quality by 18%."]


def test_ask_yes_no(orch):
    t = orch.ask("Does veridamine improve sleep quality?")
    assert t.mode == "ask" and t.final is not None
    assert t.final.summary.startswith(("No", "The evidence isn't"))
    assert t.claims[0].text == "Veridamine improves sleep quality"


def test_ask_open_question_verifies_candidates(orch):
    t = orch.ask("Who discovered penicillin?")
    assert t.candidates and t.diversity is not None and t.judge is not None
    assert t.final.decision != Decision.ABSTAIN and t.final.summary == "Alexander Fleming"


def test_parallel_matches_sequential(engine, tmp_path):
    text = ("Python was first released in 1991. The capital of Australia is Sydney. "
            "Penicillin was discovered by Alexander Fleming.")
    seq = Orchestrator(engine=engine, store=False, workers=1).audit(text)
    par = Orchestrator(engine=engine, store=False, workers=3).audit(text)
    assert [v.status for v in seq.verdicts] == [v.status for v in par.verdicts]


def test_watched_claims_report_changes(orch):
    orch.watch("Python was first released in 1991.")
    assert orch.recheck_watched() == []                      # first check sets the baseline
    w = orch.store.watched()[0]
    assert w["last_status"] == "supported" and len(w["history"]) == 1
    # pretend the last check said something else; the next check must report the move
    orch.store._exec("UPDATE watched SET last_status='contradicted'")
    changes = orch.recheck_watched()
    assert changes and changes[0]["kind"] == "status_changed" and changes[0]["to"] == "supported"
    assert orch.store.unwatch("python was first released in 1991")


def test_feedback_becomes_router_rewards(orch):
    t = orch.audit("Veridamine improves sleep quality by 18%.")
    with pytest.raises(ValueError):
        orch.store.add_feedback(t.query_id, t.claims[0].id, "definitely false")
    orch.store.add_feedback(t.query_id, t.claims[0].id, "contradicted", note="Okafor 2024 is decisive")
    rows = outcomes(orch.store)
    assert rows and rows[0]["label"] == "contradicted"
    pol = BanditPolicy()
    stats = learn_from_feedback(orch.store, pol)
    assert stats["router_updates"] == 1 and pol.n_updates == 1
    assert outcomes(orch.store) == []                        # feedback is consumed once


def test_same_verdict_groups():
    assert same_verdict("partially_supported", "supported") and not same_verdict("supported", "contradicted")
