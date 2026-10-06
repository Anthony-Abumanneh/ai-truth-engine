"""Step 5: analyzer, router (tiers, escalation, contextual bandit), Ask-mode solvers."""
import random

import numpy as np
import pytest

from truth_engine.pipeline.analyzer import analyze, is_question
from truth_engine.pipeline.audit import audit
from truth_engine.pipeline.router import BanditPolicy, Router, featurize, ips_value, reward
from truth_engine.pipeline.solvers import diversity, question_to_claim, solve
from truth_engine.retrieval import EvidenceEngine
from truth_engine.schemas import BudgetTier as T
from truth_engine.schemas import Domain, QuestionType, Risk


# ------------------------------------------------------------------ analyzer
def test_analyzer_types_and_risk():
    a = analyze("Taking 4,000 mg of vitamin C daily is safe for children.")
    assert a.domain == Domain.MEDICINE and a.risk == Risk.HIGH
    a = analyze("Veridamine improves sleep quality by 18%.")
    assert a.question_type == QuestionType.CAUSAL and a.risk == Risk.MEDIUM and "literature" in a.tools
    a = analyze("What is the capital of Australia?")
    assert a.risk == Risk.LOW and a.compute_budget == T.MINIMAL
    assert analyze("In which country was the person who developed general relativity born?").question_type \
        == QuestionType.MULTI_HOP
    assert analyze("Should I take melatonin?").question_type == QuestionType.OPINION


def test_harder_and_riskier_claims_get_bigger_budgets():
    easy = analyze("Einstein won the Nobel Prize in Physics in 1921.")
    hard = analyze("High-dose vitamin D supplements cause fewer strokes than placebo in older adults because "
                   "they lower blood pressure.")
    assert hard.difficulty > easy.difficulty
    assert list(T).index(hard.compute_budget) > list(T).index(easy.compute_budget)


def test_is_question():
    assert is_question("Does creatine improve memory?") and not is_question("Creatine improves memory.")


# ------------------------------------------------------------------ router
def test_offline_strategies_drop_network_sources():
    off, on = Router(live=False), Router(live=True, has_web=False)
    assert off.strategy(T.HIGH).sources == ("corpus", "graph")
    assert on.strategy(T.HIGH).sources == ("corpus", "graph", "pubmed", "openalex")
    assert not on.strategy(T.MINIMAL).nli and on.strategy(T.LOW).nli


def test_escalation_skips_tiers_that_add_nothing():
    off = Router(live=False)
    assert off.next_useful_tier(T.LOW, T.MAXIMUM) == T.HIGH        # offline, medium == low
    assert Router(live=True).next_useful_tier(T.LOW, T.MAXIMUM) == T.MEDIUM
    assert off.next_useful_tier(T.HIGH, T.HIGH) is None            # at the cap


def test_router_uses_rules_until_policy_has_data():
    a = analyze("Veridamine improves sleep quality by 18%.")
    r = Router(policy=BanditPolicy(), min_samples=10)
    tier, source, _ = r.start_tier(a, T.HIGH)
    assert source == "rules" and tier == a.compute_budget


# ------------------------------------------------------------------ bandit
def _synthetic_world(n=1500, seed=0):
    """Easy claims are solved at 'minimal'; hard ones need 'high'. Reward = correct - cost."""
    rng = random.Random(seed)
    easy = analyze("Python was first released in 1991.")
    hard = analyze("Vitamin D supplements prevent depression in older adults because they reduce inflammation.")
    pol = BanditPolicy(epsilon=0.2)
    log = []
    for _ in range(n):
        a = rng.choice([easy, hard])
        x = featurize(a)
        arm, p = pol.choose(x, rng=rng)
        need = T.MINIMAL if a is easy else T.HIGH
        correct = list(T).index(arm) >= list(T).index(need) and rng.random() < 0.95
        r = reward(correct, arm)
        pol.update(x, arm, r)
        log.append({"x": x, "arm": arm, "p": p, "reward": r, "allowed": None})
    return pol, log, featurize(easy), featurize(hard)


def test_bandit_learns_cheapest_sufficient_tier():
    pol, _, x_easy, x_hard = _synthetic_world()
    assert pol.greedy(x_easy) == T.MINIMAL
    assert pol.greedy(x_hard) == T.HIGH


def test_offline_policy_evaluation_ranks_policies():
    pol, log, *_ = _synthetic_world()
    learned = ips_value(log, lambda x, allowed: pol.greedy(x))
    always_max = ips_value(log, lambda x, allowed: T.MAXIMUM)
    always_min = ips_value(log, lambda x, allowed: T.MINIMAL)
    assert learned > always_max and learned > always_min


def test_policy_save_load(tmp_path):
    pol, _, x_easy, _ = _synthetic_world(n=300)
    pol.save(tmp_path / "p.json")
    again = BanditPolicy.load(tmp_path / "p.json")
    assert again.n_updates == pol.n_updates and again.greedy(x_easy) == pol.greedy(x_easy)
    assert np.allclose(again.A["minimal"], pol.A["minimal"])


# ------------------------------------------------------------------ routed audit
@pytest.fixture(scope="module")
def engine():
    return EvidenceEngine()


def test_audit_records_route_and_escalates_when_empty(engine):
    rep = audit("Glorbium cures baldness in 3 days.", engine=engine)      # nothing in the corpus
    route = rep.routing[rep.claims[0].id]
    assert route["routing_source"] == "rules" and route["analysis"]["risk"] in ("medium", "high")
    assert len(route["passes"]) >= 2 and route["passes"][0]["escalated_because"] == "no evidence found"


def test_easy_claim_stays_cheap(engine):
    rep = audit("Python was first released in 1991.", engine=engine)
    route = rep.routing[rep.claims[0].id]
    assert route["start_tier"] == "minimal" and len(route["passes"]) == 1
    assert rep.verdicts[0].status.value == "supported"


def test_forced_sources_skip_routing(engine):
    rep = audit("Python was first released in 1991.", engine=engine, sources=["corpus"])
    assert rep.routing[rep.claims[0].id]["routing_source"] == "forced" and rep.retrieval_sources == ["corpus"]


# ------------------------------------------------------------------ solvers
@pytest.mark.parametrize("q,claim", [
    ("Does creatine improve memory in older adults?", "Creatine improves memory in older adults."),
    ("Did Einstein win the Nobel Prize for relativity?", "Einstein won the Nobel Prize for relativity."),
    ("Is vitamin D safe for children?", "Vitamin D is safe for children."),
    ("Can melatonin cause nightmares?", "Melatonin can cause nightmares."),
    ("Who discovered penicillin?", None),
])
def test_question_to_claim(q, claim):
    assert question_to_claim(q) == claim


def test_solvers_and_diversity():
    cands = solve("Who discovered penicillin?", use_llms=False)
    assert {c.model for c in cands} >= {"graph", "offline-swift", "offline-scholar", "offline-sage"}
    d = diversity(cands)
    assert d.majority_value == "Alexander Fleming" and d.majority_share == 1.0
    assert d.correlated_error_risk > 0 and d.notes          # same-corpus agreement is flagged


def test_majority_can_be_wrong_on_multi_hop():
    cands = solve("In which country was the person who developed general relativity born?", use_llms=False)
    by = {c.model: c.final_value for c in cands}
    assert by["graph"] == "Germany"                          # the path solver gets the full chain
    assert diversity(cands).majority_value != "Germany"      # majority vote would be wrong: step 6's judge must see this
