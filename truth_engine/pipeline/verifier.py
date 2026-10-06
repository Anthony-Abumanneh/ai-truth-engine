"""Claim verification (CONCEPTS.md §4: natural language inference; §5:
evidence hierarchy, generalization; §6: source independence).

Two levels:

1. Stance - for one claim and one evidence sentence: does the sentence
   ENTAIL, CONTRADICT, PARTIALLY support, or say nothing (NEUTRAL) about it?
   Signals, all visible in `rationale`:
     coverage       how much of the claim the sentence talks about
     polarity       null results ("did not significantly improve", "not
                    replicated", "no evidence") vs. a positive claim
     antonyms       "increased" vs "decreased"
     numbers        same unit, different value (18% vs 2%)
     weak evidence  "low certainty", "preliminary" vs. a confident claim
   With an LLM available it gives a second opinion per item; the LLM's stance
   is used and the rule/LLM agreement rate is kept as a signal.

2. Verdict - one of the six statuses. Each item gets a weight =
   source quality x applicability x stance strength, where applicability
   asks whether the study can speak to the claim (a mouse study for a human
   claim: 0.3; an observational study for a causal claim: 0.5). Items are
   grouped into independent lineages with the citation graph; a lineage
   counts once, at its strongest item (echo detector), and a lineage whose
   root is retracted counts zero. Lineages combine by noisy-OR:
       support = 1 - prod(1 - w_lineage)
   so two independent strong studies beat one, and twelve copies of one
   pilot study count as one pilot study.
"""
from __future__ import annotations

import re
from typing import NamedTuple

from ..retrieval.citation_graph import CitationGraph
from ..retrieval.evidence import Retrieved, mentions
from ..schemas import Claim, ClaimVerdict, EvidenceItem, EvidenceStatus, Source, Stance
from ..textutils import NEGATIONS, Quantity, antonym_conflict, content_tokens, coverage, entities, quantities, words
from .claims import CERTAINTY

NULL_RESULT = re.compile(
    r"\b(?:no (?:significant|meaningful|clear|detectable) (?:effect|difference|change|improvement|benefit|association|"
    r"reduction)|not significantly|did not (?:significantly )?(?:improve|reduce|increase|change|affect|promote|"
    r"prevent|differ|lower|extend)|failed to|(?:was|were) not replicated|not replicated|no (?:evidence|effect|"
    r"association|benefit)|does not support|did not support|not supported|ineffective|not effective|"
    r"no better than placebo|had no significant|showed no)\b", re.I)
DISPUTE = re.compile(r"\b(?:myth|misconception|incorrect|debunked|retracted|fabricated|not awarded|rather than|"
                     r"not by|contrary to|false|disputed|contested|no single agreed|no consensus)\b", re.I)
WEAK = re.compile(r"\b(?:(?:very )?low certainty|preliminary|insufficient evidence|inconclusive|limited evidence|"
                  r"more research is needed|(?:further|more) (?:research|studies) (?:is|are) needed|"
                  r"should be considered preliminary)\b", re.I)
LIMITATION = re.compile(r"\b(?:cannot be ruled out|confound\w*|limitations?|caution|should be interpreted)\b", re.I)
CAUSE_SPLIT = re.compile(r"\s+(?:because(?: of)?|due to|as a result of|caused by|driven by|owing to)\s+", re.I)
CAUSE_DENIAL = re.compile(r"\b(?:not by|rather than|not because(?: of)?|not due to|unrelated to|not driven by|"
                          r"not caused by)\b", re.I)
# "won the prize for X", "is used for X": the trailing purpose/reason slot
FOR_SLOT = re.compile(r"\bfor\s+(?:the |his |her |its |their |a |an )?([a-z][\w\s-]{3,60}?)\s*[.\"]?$", re.I)
HEDGED_EVIDENCE = re.compile(r"\b(?:may|might|could|suggests?|possibly|potentially|modest(?:ly)?)\b", re.I)
OBSERVATIONAL = {"cohort", "case_control", "cross_sectional"}
STRONG_DESIGNS = {"rct", "meta_analysis", "systematic_review"}


# --------------------------------------------------------------------------- stance
TO_BASE = {"km": ("m", 1000.0)}


def _base(q: Quantity) -> Quantity:
    unit, factor = TO_BASE.get(q.unit, (q.unit, 1.0))
    return Quantity(q.value * factor, unit, q.raw)


def _close(a: Quantity, b: Quantity) -> bool:
    """Years and dates must match exactly (1995 is not 'within 2%' of 1991)."""
    if a.unit in ("year", "date") or b.unit in ("year", "date"):
        return a.unit == b.unit and a.value == b.value
    return a.close_to(b)


def negated_in_scope(text: str, claim_tokens: set[str], subject: str | None = None) -> bool:
    """A bare negation only counts if it governs what the claim asserts.
    Scope = the next three content words (minus the claim's subject); "no X"
    is a quantifier over X alone. So "no human data exist" doesn't negate
    "extends lifespan in mice", "not backward compatible with Python 2"
    doesn't negate "Python was released in 1991", but "was not awarded for the
    theory of relativity" does negate "won ... for the theory of relativity"."""
    ws = words(text)
    skip = set(content_tokens(subject)) if subject else set()
    for i, w in enumerate(ws):
        if w in NEGATIONS or w.endswith("n't"):
            after = [t for t in content_tokens(" ".join(ws[i + 1:i + 8])) if t not in skip]
            window = after[:2] if w == "no" else after[:3]
            if set(window) & claim_tokens:
                return True
    return False


def _negative(text: str, claim_tokens: set[str], subject: str | None = None) -> bool:
    return bool(NULL_RESULT.search(text) or DISPUTE.search(text)) or negated_in_scope(text, claim_tokens, subject)


class RuleStance(NamedTuple):
    stance: Stance
    strength: float
    why: str
    signal: str     # which rule decided: see HARD_SIGNALS


# Rule signals precise enough to overrule the neural model: NLI models are weak
# with numbers and "for X" slots; weak-evidence and caveat phrases are explicit.
# Thresholds below were chosen on SciFact *train* with the PyTorch backend
# (scripts/eval_stance_scifact.py reports on dev):
# "different name" and null-result phrases were no better than the model there.
HARD_SIGNALS = {"value", "value_match", "slot", "weak", "limitation", "causal_denial"}


def stance(claim: Claim, passage: str, title: str = "", context: str = "") -> tuple[Stance, float, str]:
    """(stance, strength 0..1, rationale) from the rules alone."""
    r = rule_stance(claim, passage, title, context)
    return r.stance, r.strength, r.why


def rule_stance(claim: Claim, passage: str, title: str = "", context: str = "") -> RuleStance:
    """Rule-based stance for one claim/sentence pair, plus which signal decided. `context`
    (neighbouring sentences) only helps decide whether the sentence is about the
    claim's subject ("However, the title is disputed" right after "The Nile...")."""
    ct = content_tokens(CERTAINTY.sub(" ", claim.text))
    pt = content_tokens(passage)
    cov_p = coverage(ct, pt)                                      # the sentence itself
    cov_t = coverage(ct, pt + content_tokens(title))              # with the document's title as context
    subject_ok = not claim.entities or cov_p >= 0.6 or mentions(claim.entities[0], f"{title}. {context} {passage}")
    if cov_t < 0.4 or cov_p < 0.3 or not subject_ok:
        return RuleStance(Stance.NEUTRAL, 0.0, f"off-topic (coverage {cov_p:.2f})", "offtopic")

    why = [f"coverage {cov_p:.2f}"]
    cset = set(ct)
    subj = claim.entities[0] if claim.entities else None
    claim_neg = bool(NULL_RESULT.search(claim.text)) or negated_in_scope(claim.text, cset, subj)
    pass_neg = _negative(passage, cset, subj)
    ants = antonym_conflict(claim.text, passage)
    cq = [_base(q) for q in quantities(claim.text) if q.unit]
    pq = [_base(q) for q in quantities(passage) if q.unit]
    match = any(_close(a, b) for a in cq for b in pq)
    mismatch = any(any(b.unit == a.unit for b in pq) and not any(_close(a, b) for b in pq if b.unit == a.unit)
                   for a in cq)

    # "A because B" vs "..., not by B" / "rather than B": the claimed cause is explicitly denied
    cause = CAUSE_SPLIT.split(claim.text, maxsplit=1)
    if len(cause) == 2:
        denial = CAUSE_DENIAL.search(passage)
        if denial and set(content_tokens(passage[denial.end():])) & set(content_tokens(cause[1])):
            why.append(f"claimed cause denied ('{denial.group()}')")
            return RuleStance(Stance.CONTRADICTS, 0.85, "; ".join(why), "causal_denial")
    if LIMITATION.search(passage) and not NULL_RESULT.search(passage):
        return RuleStance(Stance.NEUTRAL, 0.0, "; ".join(why + ["study limitation / caveat"]), "limitation")
    slot = FOR_SLOT.search(claim.text.rstrip("."))
    if slot and re.search(r"\bfor\b", passage, re.I) and not claim_neg:
        need = set(content_tokens(slot.group(1)))
        if need and not need & set(pt) and cov_p >= 0.45:
            why.append(f"different reason/purpose (claim: 'for {slot.group(1)}')")
            return RuleStance(Stance.CONTRADICTS, round(0.4 + 0.3 * cov_p, 3), "; ".join(why), "slot")
    if pass_neg != claim_neg and cov_p >= 0.4:
        cue = (NULL_RESULT.search(passage) or DISPUTE.search(passage))
        why.append(f"polarity conflict ('{cue.group() if cue else 'negation'}')")
        return RuleStance(Stance.CONTRADICTS, round(min(1.0, 0.55 + 0.45 * cov_p), 3), "; ".join(why),
                          "null_phrase" if cue else "negation")
    if ants and cov_p >= 0.4 and not claim_neg:
        why.append(f"opposite wording ({ants[0][0]} vs {ants[0][1]})")
        return RuleStance(Stance.CONTRADICTS, round(0.5 + 0.3 * cov_p, 3), "; ".join(why), "antonym")
    if mismatch and not match and cov_p >= 0.5:
        why.append("different value: " + ", ".join(q.raw for q in pq if any(q.unit == a.unit for a in cq)))
        return RuleStance(Stance.CONTRADICTS, round(0.35 + 0.3 * cov_p, 3), "; ".join(why), "value")
    # same statement with a different name in it ("discovered by Pasteur" vs "by Fleming")
    objects = claim.entities[1:]
    if objects and not claim_neg:
        obj_tokens = {t for e in objects for t in content_tokens(e)}
        rest = [t for t in ct if t not in obj_tokens]
        others = [e for e in entities(passage) if not set(content_tokens(e)) & (obj_tokens | cset)]
        if obj_tokens and not obj_tokens & set(pt) and others and coverage(rest, pt) >= 0.75:
            why.append(f"different name ('{others[0]}' instead of '{objects[0]}')")
            return RuleStance(Stance.CONTRADICTS, 0.5, "; ".join(why), "name")
    if WEAK.search(passage) and not claim.hedged:
        why.append(f"evidence described as weak ('{WEAK.search(passage).group()}')")
        strength = 0.7 if CERTAINTY.search(claim.text) else 0.35
        return RuleStance(Stance.CONTRADICTS, strength, "; ".join(why), "weak")
    if match:
        why.append("same value")
    if match and cov_p >= 0.45 and not HEDGED_EVIDENCE.search(passage):
        # every number checks out: NLI models are unreliable with numbers, so this is a hard signal
        st = Stance.ENTAILS if cov_p >= 0.65 else Stance.PARTIAL
        strength = min(1.0, 0.4 + 0.6 * cov_p) if st == Stance.ENTAILS else 0.6 * cov_p
        return RuleStance(st, round(strength, 3), "; ".join(why), "value_match")
    if cov_p >= 0.65 and (match or not cq):
        if HEDGED_EVIDENCE.search(passage) and not claim.hedged:
            why.append(f"hedged ('{HEDGED_EVIDENCE.search(passage).group()}')")
            return RuleStance(Stance.PARTIAL, round(0.6 * cov_p, 3), "; ".join(why), "hedged")
        return RuleStance(Stance.ENTAILS, round(min(1.0, 0.4 + 0.6 * cov_p), 3), "; ".join(why), "overlap")
    if cov_p >= 0.45:
        if cq and not match:
            why.append("value not stated")
        return RuleStance(Stance.PARTIAL, round(0.6 * cov_p, 3), "; ".join(why), "partial_overlap")
    return RuleStance(Stance.NEUTRAL, 0.0, "; ".join(why + ["too little overlap"]), "low_overlap")


# --------------------------------------------------------------------------- applicability
def study_of(src: Source) -> dict:
    return (src.metadata or {}).get("study") or {}


def applicability(claim: Claim, src: Source) -> tuple[float, list[str]]:
    st = study_of(src)
    a, why = 1.0, []
    design, species = st.get("design"), st.get("species")
    if claim.population == "human":
        if species == "animal" or design == "animal":
            a *= 0.3
            why.append("animal study, human claim")
        elif species == "in_vitro" or design == "in_vitro":
            a *= 0.2
            why.append("cell study, human claim")
    if claim.is_causal and claim.science_kind != "association" and design in OBSERVATIONAL:
        a *= 0.5
        why.append("observational study, causal claim")
    return a, why


# --------------------------------------------------------------------------- rules + NLI fusion
NLI_DECIDE = 0.5          # probability at which the model's label is taken
OFFTOPIC_CONTRADICT = 0.7  # a low-overlap sentence needs this much to count against a claim...
OFFTOPIC_WEIGHT = 0.6      # ...and then counts at reduced strength (often a different event)
NLI_DISMISS = 0.95        # P(neutral) at which a soft rule signal is discarded


def fuse(rule: RuleStance, probs: dict[str, float]) -> tuple[Stance, float, str]:
    """Combine the rules with the local NLI model (CONCEPTS.md §4).

    * Hard rule signals (numbers, names, "for X" slots, explicit null-result
      phrases, weak-evidence phrases) win: the model is unreliable there.
    * Otherwise the model decides when confident: it handles paraphrase,
      implied negation and sentences that merely share words with the claim.
    * When the model is unsure, the rules stand, at reduced strength.
    """
    pe, pn, pc = probs["entailment"], probs["neutral"], probs["contradiction"]
    tag = f"NLI e={pe:.2f} n={pn:.2f} c={pc:.2f}"
    if rule.signal in HARD_SIGNALS:
        return rule.stance, rule.strength, f"{rule.why}; {tag} (rule signal kept)"
    if rule.signal == "offtopic":
        # Few shared words: the model may rescue a paraphrase that supports the claim,
        # but a "contradiction" here is usually just a different event
        # ("launched the X1" vs "discontinued the X1").
        if pe >= NLI_DECIDE:
            return Stance.ENTAILS, round(pe * 0.8, 3), f"{rule.why}; {tag} -> entails (paraphrase)"
        if pc >= OFFTOPIC_CONTRADICT:
            return Stance.CONTRADICTS, round(pc * OFFTOPIC_WEIGHT, 3), f"{rule.why}; {tag} -> contradicts (low overlap)"
        return Stance.NEUTRAL, 0.0, f"{rule.why}; {tag}"
    agrees = {Stance.ENTAILS: rule.stance in (Stance.ENTAILS, Stance.PARTIAL),
              Stance.CONTRADICTS: rule.stance == Stance.CONTRADICTS}
    if pc >= NLI_DECIDE:
        return Stance.CONTRADICTS, round(pc * (1.0 if agrees[Stance.CONTRADICTS] else 0.85), 3), \
            f"{rule.why}; {tag} -> contradicts"
    if pe >= NLI_DECIDE:
        if rule.stance == Stance.PARTIAL and rule.signal == "hedged":
            return Stance.PARTIAL, rule.strength, f"{rule.why}; {tag}"
        return Stance.ENTAILS, round(pe * (1.0 if agrees[Stance.ENTAILS] else 0.8), 3), f"{rule.why}; {tag} -> entails"
    if pn >= NLI_DISMISS:
        return Stance.NEUTRAL, 0.0, f"{rule.why}; {tag} -> not about the claim"
    st = Stance.PARTIAL if rule.stance == Stance.ENTAILS else rule.stance
    return st, round(rule.strength * 0.7, 3), f"{rule.why}; {tag} (model unsure)"


# --------------------------------------------------------------------------- verifier
LLM_SYSTEM = """You are a careful scientific fact-checker doing natural language inference.
Given a CLAIM and one EVIDENCE sentence (with its source title), decide the evidence's stance:
"entails" (evidence states the claim), "contradicts" (evidence conflicts with it, including null
results or failed replications), "partial" (supports only part of it), or "neutral" (not about it).
Return JSON: {"stance": "...", "reason": "<one short sentence>"}"""


class Verifier:
    def __init__(self, sources: dict[str, Source] | None = None, llm=None, max_llm_items: int = 8,
                 context_fn=None, nli="auto"):
        self.sources = sources if sources is not None else {}
        self.context_fn = context_fn    # passage_id -> neighbouring text (DocumentStore.neighbors)
        if nli == "auto":
            from .nli import get_nli
            nli = get_nli()
        self.nli = nli                  # local NLI model, or None (rules only)
        self.llm = llm                  # optional (provider, ModelSpec)
        self.max_llm_items = max_llm_items
        self.cost_usd = 0.0

    def _llm_stance(self, claim: Claim, r: Retrieved) -> Stance | None:
        provider, spec = self.llm
        try:
            data, resp = provider.complete_json(
                spec, LLM_SYSTEM, f"CLAIM: {claim.text}\nSOURCE: {r.source.title}\nEVIDENCE: {r.passage.text}")
            self.cost_usd += resp.cost_usd
            return Stance(str((data or {}).get("stance", "")).lower())
        except Exception:
            return None

    def evidence_items(self, claim: Claim, retrieved: list[Retrieved]) -> list[EvidenceItem]:
        ordered = sorted(retrieved, key=lambda r: -r.relevance)
        rules = []
        for r in ordered:
            ctx = self.context_fn(r.passage.id) if self.context_fn else ""
            rules.append(rule_stance(claim, r.passage.text, r.source.title, ctx))
        # The model reads every on-topic sentence; off-topic ones (wrong subject,
        # almost no shared words) are skipped, as are graph paths (not prose).
        probs: dict[int, dict] = {}
        if self.nli is not None:
            idx = [i for i, (r, rs) in enumerate(zip(ordered, rules)) if r.found_by != "graph" and (
                rs.signal != "offtopic" or self._paraphrase_candidate(claim, r))]
            pairs = [(self._premise(claim, ordered[i]), claim.text) for i in idx]
            probs = dict(zip(idx, self.nli.predict(pairs))) if pairs else {}
        items = []
        llm_budget = self.max_llm_items if self.llm else 0
        for i, (r, rs) in enumerate(zip(ordered, rules)):
            st, strength, why = fuse(rs, probs[i]) if i in probs else (rs.stance, rs.strength, rs.why)
            llm_st = None
            if llm_budget > 0 and r.found_by != "graph":
                llm_budget -= 1
                llm_st = self._llm_stance(claim, r)
                if llm_st is not None and llm_st != st:
                    why += f"; LLM verifier: {llm_st.value}"
                    st = llm_st
                    strength = max(strength, 0.6) if st != Stance.NEUTRAL else 0.0
            if r.found_by == "graph" and st == Stance.ENTAILS:
                # a path stitches several facts together; overlap with all of them
                # is not proof of the claim, so a path supports at most partially
                st, strength, why = Stance.PARTIAL, min(strength, 0.5), why + "; knowledge-graph path"
            app, app_why = applicability(claim, r.source)
            q = r.source.quality.overall if r.source.quality else 0.5
            items.append(EvidenceItem(
                passage=r.passage, source=r.source, stance=st, stance_score=strength, relevance=r.relevance,
                found_by=r.found_by, rationale="; ".join([why] + app_why), applicability=round(app, 3),
                weight=round(q * app * strength, 3), llm_stance=llm_st, rule_stance=rs.stance, nli=probs.get(i)))
        return items

    @staticmethod
    def _paraphrase_candidate(claim: Claim, r: Retrieved) -> bool:
        """Few shared words, but the right subject: let the model judge paraphrases."""
        subject = claim.entities[0] if claim.entities else None
        return bool(subject) and mentions(subject, r.passage.text) and \
            coverage(content_tokens(claim.text), content_tokens(r.passage.text)) >= 0.25

    @staticmethod
    def _premise(claim: Claim, r: Retrieved) -> str:
        """The sentence, prefixed by its title when the sentence never names the subject
        ("The trial was publicly funded." -> which trial?)."""
        subject = claim.entities[0] if claim.entities else None
        if subject and not mentions(subject, r.passage.text):
            return f"{r.source.title}. {r.passage.text}"
        return r.passage.text

    def verify(self, claim: Claim, retrieved: list[Retrieved]) -> ClaimVerdict:
        items = self.evidence_items(claim, retrieved)
        for it in items:
            self.sources.setdefault(it.source.id, it.source)
        graph = CitationGraph(self.sources)
        self._cap_derived(claim, items, graph)
        # A source that explicitly contradicts the claim doesn't also support it
        # (e.g. an encyclopedia mentioning the prize *and* saying it wasn't for relativity).
        # ("disputed" is not a denial: a source can say "usually X, but disputed")
        against = {i.source.id for i in items if i.stance == Stance.CONTRADICTS and i.stance_score >= 0.5
                   and "different value" not in i.rationale
                   and not re.search(r"disputed|contested|no single agreed|no consensus", i.rationale)}
        # a document that states the claim's value somewhere is not contradicted
        # by a different number elsewhere in it (Python 1991 vs Python 3.0 in 2008)
        matched = {i.source.id for i in items if "same value" in i.rationale}
        for i in items:
            if i.source.id in matched and "different value" in i.rationale:
                i.rationale += "; same source states the claimed value elsewhere"
                i.stance, i.weight = Stance.NEUTRAL, 0.0
        for i in items:
            if i.source.id in against and i.stance in (Stance.ENTAILS, Stance.PARTIAL):
                i.rationale += "; outweighed by an explicit contradiction in the same source"
                i.stance, i.weight = Stance.NEUTRAL, 0.0
        sup = [i for i in items if i.stance in (Stance.ENTAILS, Stance.PARTIAL) and i.weight > 0]
        con = [i for i in items if i.stance == Stance.CONTRADICTS and i.weight > 0]
        s_score, s_lineages = self._combine(sup, graph)
        c_score, c_lineages = self._combine(con, graph, offset=len(s_lineages))
        status, conflicting = self._status(claim, sup, con, s_score, c_score)
        n_sup_sources = len({i.source.id for i in sup})
        agreement = self._agreement(items)
        verdict = ClaimVerdict(
            claim_id=claim.id, status=status, confidence=self._confidence(status, s_score, c_score),
            supporting=sorted(sup, key=lambda i: -i.weight), contradicting=sorted(con, key=lambda i: -i.weight),
            independent_support_count=len(s_lineages), supporting_source_count=n_sup_sources,
            source_independence=round(len(s_lineages) / n_sup_sources, 3) if n_sup_sources else 0.0,
            support_score=s_score, contradiction_score=c_score, conflicting=conflicting,
            lineages=[sorted(g) for g in s_lineages + c_lineages], verifier_agreement=agreement)
        verdict.rationale = self._rationale(verdict)
        return verdict

    # ------------------------------------------------------------ internals
    @staticmethod
    def _agreement(items: list[EvidenceItem]) -> float | None:
        """How often the rules agree with the second verifier (LLM if it ran, else
        NLI), counting partial as support. Low agreement = shakier verdict."""
        def norm(s):
            return Stance.ENTAILS if s == Stance.PARTIAL else s

        def nli_label(p):
            return Stance.ENTAILS if p["entailment"] >= NLI_DECIDE else \
                Stance.CONTRADICTS if p["contradiction"] >= NLI_DECIDE else Stance.NEUTRAL
        pairs = [(i.rule_stance, i.llm_stance) for i in items if i.llm_stance is not None] or \
                [(i.rule_stance, nli_label(i.nli)) for i in items if i.nli is not None]
        if not pairs:
            return None
        return round(sum(norm(a) == norm(b) for a, b in pairs) / len(pairs), 3)

    def _cap_derived(self, claim: Claim, items: list[EvidenceItem], graph: CitationGraph) -> None:
        """A source restating a study can't be stronger than that study, and is
        worth nothing if the study was retracted."""
        for it in items:
            if not it.source.derived_from or it.stance == Stance.NEUTRAL:
                continue
            for root_id in graph.roots(it.source.id):
                root = self.sources.get(root_id)
                if not root or root_id == it.source.id:
                    continue
                if study_of(root).get("retracted"):
                    it.weight = 0.0
                    it.rationale += f"; restates retracted {root_id}"
                    break
                cap = (root.quality.overall if root.quality else 0.5) * applicability(claim, root)[0] * it.stance_score
                if it.weight > cap:
                    it.weight = round(cap, 3)
                    it.rationale += f"; capped at strength of {root_id}"

    def _combine(self, items: list[EvidenceItem], graph: CitationGraph, offset: int = 0):
        if not items:
            return 0.0, []
        groups = graph.lineages([i.source.id for i in items])
        prod = 1.0
        for gi, g in enumerate(groups):
            members = [i for i in items if i.source.id in g]
            for i in members:
                i.lineage = offset + gi
            if any(study_of(self.sources[sid]).get("retracted") for sid in g if sid in self.sources):
                continue                     # a retracted lineage supports nothing
            prod *= 1 - max(i.weight for i in members)
        return round(1 - prod, 3), groups

    @staticmethod
    def _status(claim: Claim, sup, con, s: float, c: float) -> tuple[EvidenceStatus, bool]:
        conflicting = s >= 0.4 and c >= 0.4 and abs(s - c) < 0.2
        # "definitely X" can't be supported while real evidence disputes X
        if CERTAINTY.search(claim.text) and c >= 0.35 and c < s + 0.2:
            conflicting = True
        if conflicting:
            return EvidenceStatus.UNVERIFIED, True
        if c >= 0.5 and c >= s + 0.2 or c >= 0.35 and s < 0.2:
            return EvidenceStatus.CONTRADICTED, False
        entails = [i for i in sup if i.stance == Stance.ENTAILS]
        if s >= 0.5 and s >= c + 0.2:
            if entails:
                return EvidenceStatus.SUPPORTED, False
            covered = set()
            for i in sup:
                covered |= set(content_tokens(i.passage.text))
            if coverage(content_tokens(claim.text), list(covered)) >= 0.85:
                return EvidenceStatus.INFERRED, False
            return EvidenceStatus.PARTIALLY_SUPPORTED, False
        if s >= 0.25 and c < 0.25:
            status = EvidenceStatus.PARTIALLY_SUPPORTED
            return (EvidenceStatus.SPECULATIVE if claim.hedged else status), False
        if claim.hedged and s > 0 and c < 0.25:
            return EvidenceStatus.SPECULATIVE, False
        return EvidenceStatus.UNVERIFIED, False

    @staticmethod
    def _confidence(status: EvidenceStatus, s: float, c: float) -> float:
        """Provisional; step 6 replaces this with calibrated confidence."""
        if status == EvidenceStatus.SUPPORTED:
            return round(s * (1 - c), 3)
        if status == EvidenceStatus.CONTRADICTED:
            return round(c * (1 - s), 3)
        return round(max(0.2, 1 - max(s, c)) * 0.6, 3)

    def _rationale(self, v: ClaimVerdict) -> str:
        parts = []
        if v.supporting:
            top = v.supporting[0]
            label = study_of(top.source).get("design")
            parts.append(f"Support: {v.supporting_source_count} source(s) -> {v.independent_support_count} independent "
                         f"line(s) of evidence; strongest: {top.source.title}"
                         + (f" ({top.source.metadata.get('study_label')})" if label else "") + ".")
        if v.contradicting:
            top = v.contradicting[0]
            parts.append(f"Against: {top.source.title}"
                         + (f" ({top.source.metadata.get('study_label')})" if study_of(top.source) else "")
                         + f": \"{top.passage.text}\"")
        if v.supporting and v.support_score == 0 and any(
                study_of(self.sources.get(r, i.source)).get("retracted")
                for i in v.supporting for r in CitationGraph(self.sources).roots(i.source.id)):
            parts.append("The only support comes from a retracted paper, so it counts for nothing.")
        if v.conflicting:
            parts.append("Strong evidence points both ways.")
        if not parts:
            parts.append("No relevant evidence found in the searched sources.")
        return " ".join(parts)
