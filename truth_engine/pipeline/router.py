"""Compute routing (spec §4, Improvement 2; CONCEPTS.md §9: routing, cascades,
contextual bandits).

The budget ladder maps each tier to a concrete verification strategy:

    tier      sources                            verifier          depth
    minimal   corpus + graph                     rules             3 + 2
    low       corpus + graph                     rules + NLI       5 + 4
    medium    + PubMed / OpenAlex   (live)       rules + NLI       5 + 4
    high      + web (key), follow-up studies     + LLM (if any)    8 + 6
    maximum   same as high, deeper               + LLM             10 + 8

A claim starts at a tier and *escalates* one step at a time while its
verdict is weak (no evidence, conflicting evidence, low confidence) and the
cap allows. Cheap claims stay cheap; hard ones get more.

Who picks the starting tier:
  * rules   - the analyzer's budget (difficulty x risk). Default.
  * learned - a contextual bandit (LinUCB on the analyzer's features) that
              learns which starting tier maximizes  correct - lambda * cost.
              It needs outcome data (benchmark labels, user feedback), so it
              only takes over after `min_samples` updates. Choices are made
              epsilon-greedily and logged with their probability, so a new
              policy can be evaluated offline on old logs (inverse
              propensity scoring) before it is trusted.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import DATA_DIR
from ..schemas import BUDGET_ORDER, BudgetTier, ComputePlan, QuestionAnalysis, Risk


@dataclass(frozen=True)
class Strategy:
    sources: tuple[str, ...]
    nli: bool
    llm: bool
    k_support: int
    k_falsify: int
    follow_up_studies: bool          # fetch papers citing the found ones (replications, rebuttals)
    est_latency_s: float             # rough planning numbers; real latency is measured
    est_cost: float                  # relative compute units (not dollars)


STRATEGIES = {
    BudgetTier.MINIMAL: Strategy(("corpus", "graph"), False, False, 3, 2, False, 0.05, 0.1),
    BudgetTier.LOW: Strategy(("corpus", "graph"), True, False, 5, 4, False, 0.6, 0.3),
    BudgetTier.MEDIUM: Strategy(("corpus", "graph", "pubmed", "openalex"), True, False, 5, 4, False, 4.0, 1.0),
    BudgetTier.HIGH: Strategy(("corpus", "graph", "pubmed", "openalex", "web"), True, True, 8, 6, True, 9.0, 3.0),
    BudgetTier.MAXIMUM: Strategy(("corpus", "graph", "pubmed", "openalex", "web"), True, True, 10, 8, True, 15.0, 6.0),
}
LIVE_ONLY = {"pubmed", "openalex", "web"}


def tier_index(t: BudgetTier) -> int:
    return BUDGET_ORDER.index(t)


def next_tier(t: BudgetTier) -> BudgetTier | None:
    i = tier_index(t)
    return BUDGET_ORDER[i + 1] if i + 1 < len(BUDGET_ORDER) else None


# --------------------------------------------------------------------------- learned policy
FEATURES = ["is_question", "causal", "numeric", "n_entities", "length", "hedged", "certainty_words", "health",
            "efficacy", "safety", "association", "mechanism", "high_risk_terms", "current", "comparison",
            "multi_hop"]


def featurize(a: QuestionAnalysis) -> np.ndarray:
    s = a.signals
    x = [1.0, a.difficulty, float(a.risk == Risk.MEDIUM), float(a.risk == Risk.HIGH)]
    x += [min(s.get(f, 0.0), 3.0) / (3.0 if f in ("n_entities",) else 20.0 if f == "length" else 1.0)
          for f in FEATURES]
    return np.array(x, dtype=float)


@dataclass
class BanditPolicy:
    """Disjoint LinUCB with epsilon-greedy logging. One ridge regression per arm
    (= starting tier) predicts reward from claim features."""
    n_features: int = 4 + len(FEATURES)
    alpha: float = 0.5               # exploration bonus (upper confidence bound width)
    epsilon: float = 0.1             # random-arm probability when logging
    ridge: float = 1.0
    arms: list[BudgetTier] = field(default_factory=lambda: list(BUDGET_ORDER))
    A: dict = field(default_factory=dict)
    b: dict = field(default_factory=dict)
    n_updates: int = 0

    def __post_init__(self):
        for a in self.arms:
            self.A.setdefault(a.value, np.eye(self.n_features) * self.ridge)
            self.b.setdefault(a.value, np.zeros(self.n_features))

    def scores(self, x: np.ndarray, allowed: list[BudgetTier] | None = None, explore: bool = True) -> dict:
        out = {}
        for a in allowed or self.arms:
            A_inv = np.linalg.inv(self.A[a.value])
            theta = A_inv @ self.b[a.value]
            out[a] = float(theta @ x + (self.alpha * np.sqrt(x @ A_inv @ x) if explore else 0.0))
        return out

    def choose(self, x: np.ndarray, allowed: list[BudgetTier] | None = None,
               rng: random.Random | None = None) -> tuple[BudgetTier, float]:
        """(arm, probability it was chosen). Probability is logged for offline evaluation."""
        allowed = allowed or self.arms
        sc = self.scores(x, allowed)
        best = max(sc, key=sc.get)
        rng = rng or random
        arm = rng.choice(allowed) if rng.random() < self.epsilon else best
        p = (1 - self.epsilon) * (arm == best) + self.epsilon / len(allowed)
        return arm, p

    def greedy(self, x: np.ndarray, allowed: list[BudgetTier] | None = None) -> BudgetTier:
        sc = self.scores(x, allowed, explore=False)
        return max(sc, key=sc.get)

    def update(self, x: np.ndarray, arm: BudgetTier, reward: float) -> None:
        self.A[arm.value] += np.outer(x, x)
        self.b[arm.value] += reward * x
        self.n_updates += 1

    # -------------------------------------------------- persistence
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"alpha": self.alpha, "epsilon": self.epsilon, "n_updates": self.n_updates,
                                    "A": {k: v.tolist() for k, v in self.A.items()},
                                    "b": {k: v.tolist() for k, v in self.b.items()}}))

    @classmethod
    def load(cls, path: Path) -> "BanditPolicy":
        d = json.loads(path.read_text())
        p = cls(alpha=d["alpha"], epsilon=d["epsilon"])
        p.A = {k: np.array(v) for k, v in d["A"].items()}
        p.b = {k: np.array(v) for k, v in d["b"].items()}
        p.n_updates = d["n_updates"]
        return p


def reward(correct: bool, tier: BudgetTier, cost_weight: float = 0.05) -> float:
    """Being right is worth 1; each relative compute unit costs `cost_weight`."""
    return float(correct) - cost_weight * STRATEGIES[tier].est_cost


def ips_value(log: list[dict], policy) -> float:
    """Offline estimate of a policy's average reward from logged decisions
    (inverse propensity scoring). Each record: {"x", "arm", "p", "reward", "allowed"}."""
    if not log:
        return 0.0
    total = 0.0
    for r in log:
        if policy(np.asarray(r["x"]), r.get("allowed")) == r["arm"]:
            total += r["reward"] / r["p"]
    return total / len(log)


# --------------------------------------------------------------------------- router
class Router:
    def __init__(self, *, live: bool = False, has_web: bool = False, has_llm: bool = False, has_nli: bool = True,
                 policy: BanditPolicy | None = None, min_samples: int = 200,
                 policy_path: Path | None = None):
        self.live, self.has_web, self.has_llm, self.has_nli = live, has_web, has_llm, has_nli
        self.policy_path = policy_path or DATA_DIR / "router_policy.json"
        if policy is None and self.policy_path.exists():
            policy = BanditPolicy.load(self.policy_path)
        self.policy = policy
        self.min_samples = min_samples
        self.log: list[dict] = []

    def cap(self, analysis: QuestionAnalysis, max_tier: BudgetTier | None) -> BudgetTier:
        if max_tier is not None:
            return max_tier
        return BudgetTier.MAXIMUM if analysis.risk == Risk.HIGH else BudgetTier.HIGH

    def start_tier(self, analysis: QuestionAnalysis, cap: BudgetTier) -> tuple[BudgetTier, str, float]:
        allowed = [t for t in BUDGET_ORDER if tier_index(t) <= tier_index(cap)]
        if self.policy is not None and self.policy.n_updates >= self.min_samples:
            x = featurize(analysis)
            arm, p = self.policy.choose(x, allowed)
            self.log.append({"x": x.tolist(), "arm": arm, "p": p, "allowed": allowed})
            return arm, "learned", p
        rule = analysis.compute_budget
        return (rule if tier_index(rule) <= tier_index(cap) else cap), "rules", 1.0

    def strategy(self, tier: BudgetTier) -> Strategy:
        """The tier's strategy, minus anything unavailable here (no network, no keys, no model)."""
        s = STRATEGIES[tier]
        sources = tuple(x for x in s.sources if (self.live or x not in LIVE_ONLY) and (x != "web" or self.has_web))
        return Strategy(sources, s.nli and self.has_nli, s.llm and self.has_llm, s.k_support, s.k_falsify,
                        s.follow_up_studies and self.live, s.est_latency_s, s.est_cost)

    def plan(self, analysis: QuestionAnalysis, tier: BudgetTier, cap: BudgetTier, source: str) -> ComputePlan:
        st = self.strategy(tier)
        return ComputePlan(
            tier=tier, solver_models=[], verifier_model="rules" + ("+nli" if st.nli else "") + ("+llm" if st.llm else ""),
            critic_model=None, judge_model="rules", retrieval=analysis.requires_retrieval,
            retrieval_sources=list(st.sources), contradiction_search=True,
            adversarial_review=tier_index(tier) >= tier_index(BudgetTier.HIGH), use_graph="graph" in st.sources,
            tools=list(analysis.tools), max_cost_usd=1.0 if st.llm else 0.0, max_latency_s=st.est_latency_s * 3,
            escalation_allowed=tier_index(tier) < tier_index(cap), routing_source=source,
            rationale=f"{source} start at {tier.value} (analyzer: {analysis.rationale}); cap {cap.value}")

    @staticmethod
    def _what_it_does(st: Strategy) -> tuple:
        return st.sources, st.nli, st.llm, st.k_support, st.k_falsify, st.follow_up_studies

    def next_useful_tier(self, tier: BudgetTier, cap: BudgetTier) -> BudgetTier | None:
        """The next tier up to `cap` that would actually do something different here
        (offline, 'medium' adds nothing over 'low', so it is skipped)."""
        cur = self._what_it_does(self.strategy(tier))
        t = next_tier(tier)
        while t is not None and tier_index(t) <= tier_index(cap):
            if self._what_it_does(self.strategy(t)) != cur:
                return t
            t = next_tier(t)
        return None

    def should_escalate(self, verdict, tier: BudgetTier, cap: BudgetTier, threshold: float = 0.35) -> str | None:
        """Reason to climb to the next useful tier, or None."""
        if self.next_useful_tier(tier, cap) is None:
            return None
        no_evidence = not verdict.supporting and not verdict.contradicting
        if no_evidence:
            return "no evidence found"
        if verdict.conflicting:
            return "evidence conflicts"
        if verdict.status.value == "unverified":
            return "unverified"
        if verdict.confidence < threshold:
            return f"low confidence ({verdict.confidence:.2f})"
        # A decisive verdict built from one side only (or one line of evidence) gets a
        # second look when a higher tier would search more *sources* (live literature,
        # follow-up studies): one meta-analysis can be contradicted by another (seen
        # live: melatonin). Offline there is nothing broader to search, so no change.
        if verdict.status.value != "unverified":
            against = verdict.contradicting if verdict.status.value == "contradicted" else verdict.supporting
            other = verdict.supporting if verdict.status.value == "contradicted" else verdict.contradicting
            lineages = {i.lineage for i in against if i.lineage is not None}
            cur, nxt = self.strategy(tier), self.strategy(self.next_useful_tier(tier, cap))
            broader = set(nxt.sources) - set(cur.sources) or (nxt.follow_up_studies and not cur.follow_up_studies)
            if broader and (len(lineages) <= 1 or not other):
                return "only one side of the evidence found" if not other else "single line of evidence"
        return None
