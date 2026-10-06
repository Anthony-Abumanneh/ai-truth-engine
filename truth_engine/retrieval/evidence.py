"""Adaptive, claim-driven evidence retrieval (spec Improvements 6 and 8).

For every claim the engine:
  1. derives evidence requirements (what would support it, what would refute it)
  2. issues *supporting* queries and, separately, *falsification* queries
  3. pulls from the selected sources (corpus, web, knowledge graph, and the
     scholarly literature via PubMed / OpenAlex)
  4. ranks by relevance, drops noise (Challenge 3), and tags each passage with
     how it was found so the verifier and UI can tell the two searches apart

Falsification queries are the anti-confirmation-bias mechanism: they strip
the specific value out of a claim (so passages stating a *different* value
surface), swap polarity words, and for causal claims search for the effect
without the proposed cause (so alternative explanations surface).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..schemas import Claim, Passage, Source
from ..textutils import (antonym_conflict, content_tokens, coverage, entities, has_negation,
                         is_causal, quantities, words, CAUSAL_MARKERS, _ANT)
from .knowledge_graph import KnowledgeGraph, get_graph
from .store import DocumentStore, get_store
from .web import WebSearch, get_web

DISPUTE_WORDS = "myth misconception false not disputed incorrect however contrary not replicated no evidence"


@dataclass
class Retrieved:
    passage: Passage
    source: Source
    found_by: str            # supporting_search | contradiction_search | graph
    query: str
    relevance: float
    contradiction_potential: float = 0.0


@dataclass
class EvidencePlan:
    claim_id: str
    support_queries: list[str]
    falsify_queries: list[str]
    sources: list[str]
    requirements: list[str] = field(default_factory=list)


def claim_subject(claim: Claim) -> str | None:
    """What the claim is about: its first named entity ("Vitamin D"), else its
    first topical keyword ("creatine")."""
    if claim.entities:
        return claim.entities[0]
    from ..science.search import query_terms
    terms = query_terms(claim.text, max_terms=1)
    return terms[0] if terms else None


def mentions(subject: str, text: str) -> bool:
    """Phrase match on lowercase words, falling back to stems (creatine/creatine's)."""
    if re.search(r"(?<![\w-])" + re.escape(subject.lower()) + r"(?![\w-])", " ".join(words(text))):
        return True
    need = set(content_tokens(subject))
    return bool(need) and need <= set(content_tokens(text))


def contradiction_potential(claim: str, passage: str) -> float:
    """How likely a passage is to *contradict* a claim, before full verification.
    High when the passage is about the same thing (entity/topic overlap) but
    differs in a value, a polarity word, or negation."""
    ct, pt = content_tokens(claim), content_tokens(passage)
    topical = coverage(ct, pt)
    if topical < 0.2:
        return 0.0
    score = 0.0
    cq = [q for q in quantities(claim) if q.unit not in ("",)]
    pq = [q for q in quantities(passage) if q.unit not in ("",)]
    for a in cq:
        same_unit = [b for b in pq if b.unit == a.unit]
        if same_unit and not any(a.close_to(b) for b in same_unit):
            score += 0.45
            break
    if antonym_conflict(claim, passage):
        score += 0.35
    if has_negation(passage) != has_negation(claim):
        score += 0.3
    if re.search(r"\b(myth|misconception|disputed|not replicated|no significant|did not|not awarded|rather than|not by)\b",
                 passage, re.I):
        score += 0.25
    return round(min(1.0, score) * (0.5 + 0.5 * topical), 3)


class EvidenceEngine:
    def __init__(self, store: DocumentStore | None = None, graph: KnowledgeGraph | None = None,
                 web: WebSearch | None = None, science=None):
        self.store = store or get_store()
        self.graph = graph or get_graph()
        self.web = web or get_web()
        self._science = science
        self.last_external_sources: dict[str, Source] = {}

    @property
    def science(self):
        if self._science is None:
            from ..science.search import get_science
            self._science = get_science()
        return self._science

    # ------------------------------------------------------------ planning
    def plan(self, claim: Claim, sources: list[str], contradiction: bool = True) -> EvidencePlan:
        text = claim.text
        ents = claim.entities or entities(text)
        reqs = []
        support = [text]
        if ents:
            key_terms = [w for w in words(text) if w not in {e.lower() for e in ents}][:8]
            support.append(" ".join(ents[:3] + key_terms))
        falsify: list[str] = []
        if contradiction:
            # (a) value-free: same topic, any value -> surfaces conflicting numbers/dates
            stripped = text
            for q in quantities(text):
                stripped = stripped.replace(q.raw, " ")
            stripped = re.sub(r"\s+", " ", stripped).strip()
            if stripped != text and len(content_tokens(stripped)) >= 2:
                falsify.append(stripped)
                reqs.append("independent source stating the same value")
            # (b) polarity flip
            flipped = " ".join(sorted(_ANT.get(w, {w}), key=len)[0] if w in _ANT else w for w in words(text))
            if flipped != " ".join(words(text)):
                falsify.append(flipped)
            # (c) dispute framing
            falsify.append(" ".join((ents or content_tokens(text))[:3]) + " " + DISPUTE_WORDS)
            # (d) causal claims: look for the effect *without* the proposed cause
            if claim.is_causal or is_causal(text):
                parts = CAUSAL_MARKERS.split(text, maxsplit=1)
                effect = parts[0] if parts else text
                cause = parts[-1] if len(parts) > 1 else ""
                falsify.append(effect + " reason decision driven by")
                if cause:
                    falsify.append(cause + " unchanged stable not")
                reqs.append("evidence linking cause to effect, and evidence ruling out alternatives")
        return EvidencePlan(claim.id, support, falsify, sources, reqs)

    # ------------------------------------------------------------ retrieval
    def gather(self, claim: Claim, plan: EvidencePlan, k_support: int = 5, k_falsify: int = 4,
               min_relevance: float = 0.2, follow_up_studies: bool = False) -> list[Retrieved]:
        """`follow_up_studies`: also read papers that cite the papers found (later
        replications, failed replications, rebuttals), via OpenAlex."""
        out: dict[str, Retrieved] = {}
        external: dict[str, Source] = {}
        subject = claim_subject(claim)

        def consider(p: Passage, src: Source, found_by: str, q: str):
            # Subject gate for outside sources (papers, web): a passage must be
            # about the claim's subject, in its own text or its title. Drops
            # e.g. an omega-3 trial's sentence retrieved for a vitamin-D claim.
            if subject and src.id not in self.store.sources and not mentions(subject, f"{src.title}. {p.text}"):
                return
            title_tokens = content_tokens(src.title)
            rel = coverage(content_tokens(claim.text), content_tokens(p.text) + title_tokens)
            ent_overlap = sum(1 for e in (claim.entities or []) if e.lower() in (p.text + " " + src.title).lower())
            rel = min(1.0, rel + 0.1 * ent_overlap)
            if rel < min_relevance:
                return
            cp = contradiction_potential(claim.text, p.text)
            prev = out.get(p.id)
            if prev is None:
                out[p.id] = Retrieved(p, src, found_by, q, round(rel, 3), cp)
            elif found_by == "contradiction_search" and cp > 0 and prev.found_by != "contradiction_search":
                # the falsification search independently surfaced a passage with conflict signals
                prev.found_by, prev.query = "contradiction_search", q

        use_corpus = "corpus" in plan.sources or "internal" in plan.sources
        use_web = "web" in plan.sources and self.web.available
        lit = tuple(x for x in ("pubmed", "openalex") if x in plan.sources)
        for q in plan.support_queries:
            if use_corpus:
                for p in self.store.search(q, k=k_support):
                    consider(p, self.store.sources[p.source_id], "supporting_search", q)
            if use_web:
                for src, ps in self.web.search(q, k=4):
                    external[src.id] = src
                    for p in ps:
                        consider(p, src, "supporting_search", q)
        if lit:
            # one literature query per claim: the APIs do their own term matching
            for src, ps in self.science.search(claim.text, k=k_support, use=lit):
                external[src.id] = src
                for p in ps:
                    consider(p, src, "supporting_search", claim.text)
        for q in plan.falsify_queries:
            hits = []
            if use_corpus:
                hits += [(p, self.store.sources[p.source_id]) for p in self.store.search(q, k=k_falsify * 2)]
            if use_web:
                for src, ps in self.web.search(q, k=3):
                    external[src.id] = src
                    hits += [(p, src) for p in ps]
            # rerank falsification hits by contradiction potential, keep the top few
            hits.sort(key=lambda h: -contradiction_potential(claim.text, h[0].text))
            for p, src in hits[:k_falsify]:
                consider(p, src, "contradiction_search", q)
        if lit and plan.falsify_queries:
            # Falsification in the literature: the strongest designs (trials, reviews)
            # on the same topic, whatever they concluded, reranked by conflict signals.
            hits = [(p, src) for src, ps in self.science.search(claim.text, k=k_support, use=lit, high_evidence=True)
                    for p in ps]
            for src in {s.id: s for _, s in hits}.values():
                external[src.id] = src
            hits.sort(key=lambda h: -contradiction_potential(claim.text, h[0].text))
            for p, src in hits[:k_falsify]:
                consider(p, src, "contradiction_search", "high-evidence literature search")
        if follow_up_studies and lit:
            found = [r.source.id for r in out.values() if r.source.id in external][:3]
            hits = [(p, src) for sid in found for src, ps in self.science.cited_by(sid, k=5) for p in ps]
            for src in {s.id: s for _, s in hits}.values():
                external[src.id] = src
            hits.sort(key=lambda h: -contradiction_potential(claim.text, h[0].text))
            for p, src in hits[:k_falsify]:
                consider(p, src, "contradiction_search", "follow-up studies citing the evidence")
        if "graph" in plan.sources:
            for gp in self.graph.multi_hop(claim.text, k=2):
                if gp.score < 0.3:
                    continue
                pid = "graph:" + "|".join(f"{e.s}>{e.rel}>{e.o}" for e in gp.edges)
                src_id = gp.sources[0] if gp.sources else "knowledge-graph"
                src = self.store.sources.get(src_id) or Source(id="knowledge-graph", title="Knowledge graph",
                                                               source_type="database", authority=0.8)
                p = Passage(id=pid, source_id=src.id, text=gp.text, score=gp.score, retrieval_method="graph")
                out[pid] = Retrieved(p, src, "graph", claim.text, min(1.0, 0.5 + gp.score / 2), 0.0)
        res = sorted(out.values(), key=lambda r: (-(r.relevance + r.contradiction_potential), r.passage.id))
        self.last_external_sources = external
        return res[: k_support + k_falsify + 2]
