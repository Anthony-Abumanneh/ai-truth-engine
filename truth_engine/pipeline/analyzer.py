"""Claim / question analyzer (spec §4, Improvement 1; CONCEPTS.md §9).

Reads one claim or question and outputs the spec's analysis record:
domain, type, difficulty, risk, whether retrieval / current data is needed,
and the compute budget to start with. Deterministic and explainable: every
number comes from named features in `signals`, which the router (and its
learned policy) also uses.
"""
from __future__ import annotations

import re

from ..schemas import BudgetTier, Domain, QuestionAnalysis, QuestionType, Risk
from ..textutils import content_tokens, entities, quantities
from .claims import ASSOCIATION, CERTAINTY, CLAIM_HEDGE, EFFICACY, MECHANISM, SAFETY, science_kind

DOMAIN_WORDS = {
    Domain.MEDICINE: r"patients?|disease|treatment|therap\w*|drug|dose|dosage|clinical|symptoms?|cancer|diabet\w*|"
                     r"depress\w*|dementia|stroke|heart|blood|vaccin\w*|infection|supplement\w*|vitamin|placebo|"
                     r"sleep|memory|insomnia|pain|surgery|mortality|risk of|health|cures?|cured|heal\w*|"
                     r"baldness|hair loss|weight loss|obesity|anxiety|immun\w*|inflammation|cholesterol|"
                     r"pills?|medication|nutrition|diet",
    Domain.SCIENCE: r"study|studies|trial|research|cells?|mice|genes?|protein|species|physics|chemical|molecul\w*|"
                    r"evolution|climate|neurons?|brain|lifespan|dna|light|relativity",
    Domain.BUSINESS: r"revenue|sales|company|market|customers?|profit|product|launched|discontinued|supplier",
    Domain.FINANCE: r"stocks?|invest\w*|interest rate|inflation|bonds?|portfolio|crypto\w*",
    Domain.LAW: r"\blaw\b|legal|court|statute|lawsuit|regulation|illegal",
    Domain.TECHNOLOGY: r"software|programming|python|computer|algorithm|internet|gps|satellite|ai\b",
    Domain.HISTORY: r"\bborn\b|\bdied\b|war|empire|century|ancient|histor\w*|founded|discovered",
    Domain.GEOGRAPHY: r"capital|river|mountain|country|city|continent|ocean|tallest|longest",
    Domain.MATHEMATICS: r"\bprove\b|theorem|integral|equation|derivative|\bsolve\b",
}
HIGH_RISK = re.compile(r"\b(?:dose|dosage|mg|overdose|cure|cures|treat(?:s|ment)?|stop taking|instead of|"
                       r"pregnan\w*|children|infants?|cancer|chemotherapy|vaccin\w*|safe|safety|side effects?|"
                       r"toxic\w*|suicid\w*)\b", re.I)
CURRENT = re.compile(r"\b(?:latest|current(?:ly)?|now|today|this (?:year|month|week)|recent(?:ly)?|new(?:est)?|"
                     r"20(?:2[5-9]|3\d))\b", re.I)
OPINION = re.compile(r"\b(?:should I|best|worst|better than|favorite|do you think|is it worth|recommend)\b", re.I)
QUESTION_START = re.compile(r"^(?:who|what|when|where|which|why|how|does|do|did|is|are|was|were|can|could|will|"
                            r"would|should|has|have)\b", re.I)

# Budget = f(difficulty, risk): harder or riskier -> start higher on the ladder.
_BUDGET_GRID = [  # (min difficulty, risk) -> tier, first match wins
    (0.70, Risk.HIGH, BudgetTier.MAXIMUM), (0.45, Risk.HIGH, BudgetTier.HIGH), (0.0, Risk.HIGH, BudgetTier.MEDIUM),
    (0.70, Risk.MEDIUM, BudgetTier.HIGH), (0.40, Risk.MEDIUM, BudgetTier.MEDIUM), (0.0, Risk.MEDIUM, BudgetTier.LOW),
    (0.60, Risk.LOW, BudgetTier.MEDIUM), (0.30, Risk.LOW, BudgetTier.LOW), (0.0, Risk.LOW, BudgetTier.MINIMAL),
]


def is_question(text: str) -> bool:
    return text.strip().endswith("?") or bool(QUESTION_START.match(text.strip()))


def _domain(text: str) -> tuple[Domain, dict[str, int]]:
    hits = {d.value: len(re.findall(rf"\b(?:{rx})", text, re.I)) for d, rx in DOMAIN_WORDS.items()}
    best = max(hits, key=hits.get)
    return (Domain(best) if hits[best] else Domain.GENERAL), hits


def _qtype(text: str, is_q: bool) -> QuestionType:
    t = text.strip().lower()
    if OPINION.search(text):
        return QuestionType.OPINION
    if re.search(r"\b(?:why|because|cause[sd]?|due to|leads? to|prevents?|reason)\b", t):
        return QuestionType.CAUSAL
    if re.search(r"\b(?:vs\.?|versus|compared (?:to|with)|better than|more than|less than)\b", t):
        return QuestionType.COMPARISON
    if re.search(r"\b(?:how (?:many|much)|calculate|percent(?:age)?)\b", t) and is_q:
        return QuestionType.CALCULATION
    if re.match(r"^(?:what is|what are|define)\b", t):
        return QuestionType.DEFINITION
    if re.search(r"\b(?:evidence|studies|research|meta-analys\w*|trials?)\b", t):
        return QuestionType.RESEARCH_SYNTHESIS
    if is_q and re.search(r"\b(?:who|which|where)\b.*\b(?:of|whose|that|who)\b.*\b(?:born|located|developed|"
                          r"capital)\b", t):
        return QuestionType.MULTI_HOP
    return QuestionType.FACTUAL


def analyze(text: str) -> QuestionAnalysis:
    """Analysis of one claim or question; `signals` are the features behind each number."""
    is_q = is_question(text)
    domain, domain_hits = _domain(text)
    qtype = _qtype(text, is_q)
    kind = science_kind(text)
    if qtype == QuestionType.FACTUAL and kind == "efficacy":
        qtype = QuestionType.CAUSAL           # "X improves Y" is a causal claim
    ents = entities(text)
    nums = [q for q in quantities(text) if q.unit not in ("year", "date")]
    s = {
        "is_question": float(is_q),
        "causal": float(qtype == QuestionType.CAUSAL or kind in ("causal", "efficacy")),
        "numeric": float(bool(nums)),
        "n_entities": float(len(ents)),
        "length": float(len(content_tokens(text))),
        "hedged": float(bool(CLAIM_HEDGE.search(text))),
        "certainty_words": float(bool(CERTAINTY.search(text))),
        "health": float(domain in (Domain.MEDICINE, Domain.SCIENCE)),
        "efficacy": float(bool(EFFICACY.search(text))),
        "safety": float(bool(SAFETY.search(text))),
        "association": float(bool(ASSOCIATION.search(text))),
        "mechanism": float(bool(MECHANISM.search(text))),
        "high_risk_terms": float(bool(HIGH_RISK.search(text))),
        "current": float(bool(CURRENT.search(text))),
        "comparison": float(qtype == QuestionType.COMPARISON),
        "multi_hop": float(qtype == QuestionType.MULTI_HOP),
    }
    # Difficulty: causal/mechanistic/comparative science is hard; a dated fact is easy.
    d = 0.15 + 0.2 * s["causal"] + 0.1 * s["numeric"] + 0.15 * s["comparison"] + 0.15 * s["multi_hop"] \
        + 0.1 * s["mechanism"] + 0.1 * s["health"] + 0.05 * min(s["n_entities"], 3) / 3 \
        + 0.1 * min(s["length"], 20) / 20 + 0.05 * s["current"]
    difficulty = round(min(1.0, d), 3)
    if s["high_risk_terms"] and domain == Domain.MEDICINE or s["safety"]:
        risk = Risk.HIGH
    elif domain in (Domain.MEDICINE, Domain.FINANCE, Domain.LAW) or s["health"] and s["efficacy"]:
        risk = Risk.MEDIUM
    else:
        risk = Risk.LOW
    budget = next(t for dmin, r, t in _BUDGET_GRID if r == risk and difficulty >= dmin)
    needs_retrieval = qtype not in (QuestionType.OPINION, QuestionType.PROOF, QuestionType.CALCULATION) \
        or bool(ents)
    tools = (["literature"] if s["health"] else []) + (["web"] if s["current"] else []) + \
            (["graph"] if qtype == QuestionType.MULTI_HOP or s["n_entities"] >= 2 else [])
    return QuestionAnalysis(
        domain=domain, question_type=qtype, difficulty=difficulty, risk=risk, requires_retrieval=needs_retrieval,
        requires_current_data=bool(s["current"]), tools=tools, compute_budget=budget, entities=ents, signals=s,
        rationale=f"{domain.value}/{qtype.value} ({kind}); domain hits {max(domain_hits.values())}; "
                  f"difficulty {difficulty:.2f}; risk {risk.value}")
