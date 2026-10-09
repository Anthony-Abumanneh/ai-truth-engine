"""Learning from outcomes (spec §4, Improvements 2 and 9; CONCEPTS.md §9:
contextual bandits): stored traces + a known right answer become

  * router rewards - which starting tier would have been worth it for this
    kind of claim? reward = correct - cost_weight * cost of the tiers used
  * calibration rows - (signals, status, correct) to refit confidence

Outcomes come from user feedback (feedback table) or labeled benchmark runs
(step 8). The bandit only takes over routing after enough updates
(Router.min_samples), and every update is logged with the claim's features
so a new policy can be checked offline (ips_value) before it is trusted.
"""
from __future__ import annotations

from ..pipeline.analyzer import analyze
from ..pipeline.router import STRATEGIES, BanditPolicy, featurize
from ..schemas import BudgetTier, EvidenceStatus
from .traces import TraceStore

COST_WEIGHT = 0.05


def route_cost(route: dict) -> float:
    return sum(STRATEGIES[BudgetTier(p["tier"])].est_cost for p in route.get("passes", []) if p.get("tier"))


def same_verdict(predicted: str, label: str) -> bool:
    """Feedback labels may be coarse: 'supported' covers partial / inferred."""
    group = {"supported": "support", "partially_supported": "support", "inferred": "support",
             "contradicted": "contradict", "unverified": "unclear", "speculative": "unclear"}
    return group.get(predicted, predicted) == group.get(label, label)


def outcomes(store: TraceStore, unused_only: bool = True) -> list[dict]:
    """One row per feedback item, joined with what the engine did."""
    rows = []
    for fb in store.feedback(unused_only=unused_only):
        trace = store.get_trace(fb["query_id"])
        if trace is None:
            continue
        v = next((v for v in trace.verdicts if v.claim_id == fb["claim_id"]), None)
        route = trace.routing.get(fb["claim_id"], {})
        if v is None:
            continue
        pre = (v.leaning or v.status).value
        rows.append({"feedback_id": fb["id"], "claim": fb["claim_text"], "label": fb["label"], "predicted": pre,
                     "correct": same_verdict(pre, fb["label"]), "route": route,
                     "signals": v.confidence_detail.model_dump() if v.confidence_detail else None})
    return rows


def update_router(policy: BanditPolicy, rows: list[dict]) -> int:
    """Apply outcome rows as bandit rewards; returns how many were used."""
    n = 0
    for r in rows:
        start = r["route"].get("start_tier")
        if not start:
            continue                    # forced or certainty-claim routes carry no routing decision
        x = featurize(analyze(r["claim"]))
        policy.update(x, BudgetTier(start), float(r["correct"]) - COST_WEIGHT * route_cost(r["route"]))
        n += 1
    return n


def calibration_rows(rows: list[dict]) -> list[tuple[dict, str, bool]]:
    keys = ("evidence_support", "contradiction", "source_quality", "source_independence", "verifier_agreement",
            "model_agreement", "historical_reliability", "unresolved_uncertainty", "difficulty_penalty")
    return [({k: r["signals"][k] for k in keys}, r["predicted"], r["correct"]) for r in rows if r["signals"]]


def learn_from_feedback(store: TraceStore, policy: BanditPolicy | None = None, save_path=None) -> dict:
    """Consume unused feedback: update (and save) the router policy, mark feedback used."""
    rows = outcomes(store, unused_only=True)
    policy = policy or BanditPolicy()
    used = update_router(policy, rows)
    if save_path:
        policy.save(save_path)
    store.mark_used([r["feedback_id"] for r in rows])
    acc = sum(r["correct"] for r in rows) / len(rows) if rows else None
    return {"feedback_rows": len(rows), "router_updates": used, "policy_updates_total": policy.n_updates,
            "accuracy_on_feedback": acc, "calibration_rows": len(calibration_rows(rows))}


VALID_LABELS = {s.value for s in EvidenceStatus}
