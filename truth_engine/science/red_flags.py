"""Science red flags (CONCEPTS.md §5): the classic ways a real finding gets
distorted on its way into a claim. Each check looks at the claim's wording
and at the *root* studies behind its supporting evidence (following
derived_from, so a blog post is judged by the study it restates).

    retracted               support rests on a retracted paper
    echo_chamber            many sources, few independent studies
    animal_to_human         mouse result stated as a human result
    in_vitro_to_human       cell-culture result stated as a human result
    correlation_as_causation  "prevents"/"causes" backed only by observational data
    relative_risk_only      "cuts risk in half" with no absolute risk
    single_study / failed_replication
    small_sample, industry_funded
    overclaimed_certainty   "clinically proven" without strong, uncontested evidence
"""
from __future__ import annotations

import re

from ..retrieval.citation_graph import CitationGraph
from ..schemas import Claim, ClaimVerdict, EvidenceStatus, RedFlag, Source
from ..pipeline.claims import CERTAINTY

OBSERVATIONAL = {"cohort", "case_control", "cross_sectional"}
STRONG = {"rct", "meta_analysis", "systematic_review"}
# A *relative* change in risk needs a magnitude ("in half", "50%", "3-fold"); "lower risk" alone is not one.
_MAGNITUDE = r"(?:in half|halve[sd]?|by \d+(?:\.\d+)?\s*%|\d+(?:\.\d+)?\s*%|\d+-fold|two-fold|twice|double[sd]?|triple[sd]?)"
RELATIVE_RISK = re.compile(
    rf"\b(?:cuts?|reduces?|lowers?|halves?|slashes?|increases?|raises?|doubles?|triples?)\b[^.]{{0,40}}\brisk\b[^.]{{0,20}}{_MAGNITUDE}"
    rf"|\b(?:halves?|doubles?|triples?)\b[^.]{{0,30}}\brisk\b"
    rf"|\brisk\b[^.]{{0,30}}{_MAGNITUDE}"
    r"|\d+(?:\.\d+)?\s*%\s*(?:lower|higher|reduced|increased|less|greater)\s+(?:risk|odds|chance)", re.I)
ABSOLUTE = re.compile(r"\babsolute risk\b|percentage points?|\bNNT\b|number needed to treat|\bin \d[\d,]* people\b|"
                      r"\b\d+ (?:in|out of) \d[\d,]*\b|from \d+(?:\.\d+)?\s*% to \d+(?:\.\d+)?\s*%", re.I)


def _study(src: Source | None) -> dict:
    return ((src.metadata or {}).get("study") or {}) if src else {}


def _roots(src: Source, sources: dict[str, Source], graph: CitationGraph) -> list[Source]:
    out = [sources[r] for r in graph.roots(src.id) if r in sources]
    return out or [src]


def red_flags(claim: Claim, verdict: ClaimVerdict, sources: dict[str, Source],
              document_text=None) -> list[RedFlag]:
    """`document_text(source_id) -> str` lets the relative-risk check read the
    whole study (to quote its absolute numbers), not only retrieved sentences."""
    graph = CitationGraph(sources)
    flags: list[RedFlag] = []
    sup_sources = list({i.source.id: i.source for i in verdict.supporting}.values())
    roots = list({r.id: r for s in sup_sources for r in _roots(s, sources, graph)}.values())
    root_studies = [(r, _study(r)) for r in roots]
    papers = [(r, st) for r, st in root_studies if st.get("design") not in (None, "news", "marketing", "opinion")]

    # retracted
    bad = [r for r, st in root_studies if st.get("retracted")]
    if bad:
        flags.append(RedFlag(kind="retracted", severity=0.9, source_ids=[r.id for r in bad],
                             message=f"Rests on a retracted paper: {bad[0].title}."))

    # echo chamber
    n, k = verdict.supporting_source_count, verdict.independent_support_count
    if n >= 3 and k <= n / 2:
        flags.append(RedFlag(kind="echo_chamber", severity=0.5, source_ids=[s.id for s in sup_sources],
                             message=f"{n} supporting sources trace back to {k} independent "
                                     f"stud{'y' if k == 1 else 'ies'}."))

    # generalization: animal / cell results stated for humans
    if claim.population == "human" and papers:
        for kind, species in (("animal_to_human", "animal"), ("in_vitro_to_human", "in_vitro")):
            hits = [r for r, st in papers if st.get("species") == species or st.get("design") == species]
            if hits:
                human = [r for r, st in papers if st.get("species") in ("human", None) and r not in hits]
                what = "mice or other animals" if species == "animal" else "cells in a dish"
                flags.append(RedFlag(kind=kind, severity=0.4 if human else 0.8, source_ids=[r.id for r in hits],
                                     message=f"The supporting result comes from {what}"
                                             + ("; human evidence is limited." if human else
                                                ", not people. It may not apply to humans.")))

    # correlation stated as causation
    if claim.is_causal and claim.science_kind != "association" and papers:
        if all(st.get("design") in OBSERVATIONAL for _, st in papers):
            flags.append(RedFlag(kind="correlation_as_causation", severity=0.7, source_ids=[r.id for r, _ in papers],
                                 message="The claim says one thing causes another, but the supporting studies are "
                                         "observational: they show an association, which can be explained by "
                                         "other factors (confounding)."))

    # relative risk without absolute risk
    if RELATIVE_RISK.search(claim.text) and not ABSOLUTE.search(claim.text):
        absolute = None
        for s in sup_sources + roots:
            text = " ".join(i.passage.text for i in verdict.supporting if i.source.id == s.id)
            if document_text:
                text += " " + (document_text(s.id) or "")
            for sent in re.split(r"(?<=[.!?])\s+", text):
                if ABSOLUTE.search(sent):
                    absolute = sent.strip()
                    break
            if absolute:
                break
        flags.append(RedFlag(kind="relative_risk_only", severity=0.6, source_ids=[s.id for s in sup_sources],
                             message="States a relative change in risk without the absolute risk."
                                     + (f" The study says: \"{absolute}\"" if absolute else "")))

    # single study, failed replication
    primary = [(r, st) for r, st in papers if st.get("design") not in ("meta_analysis", "systematic_review",
                                                                        "narrative_review")]
    if k == 1 and primary:
        root, st = primary[0]
        stronger = [i.source for i in verdict.contradicting
                    if _study(i.source).get("design") in STRONG
                    and (_study(i.source).get("sample_size") or 0) > (st.get("sample_size") or 0)]
        if stronger:
            flags.append(RedFlag(kind="failed_replication", severity=0.8, source_ids=[root.id, stronger[0].id],
                                 message=f"A larger study ({stronger[0].metadata.get('study_label')}) did not "
                                         f"replicate the original finding."))
        else:
            flags.append(RedFlag(kind="single_study", severity=0.4, source_ids=[root.id],
                                 message="All support traces back to a single study that has not been replicated."))

    # small samples, industry funding
    sizes = [st.get("sample_size") for _, st in primary if st.get("sample_size")]
    pooled = any(st.get("design") in ("meta_analysis", "systematic_review") for _, st in papers)
    if sizes and max(sizes) < 100 and not pooled:
        flags.append(RedFlag(kind="small_sample", severity=0.4, source_ids=[r.id for r, _ in primary],
                             message=f"The supporting stud{'y has' if len(sizes) == 1 else 'ies have'} small samples "
                                     f"(largest n={max(sizes)})."))
    funded = [r for r, st in papers if st.get("funding") == "industry"]
    if funded:
        flags.append(RedFlag(kind="industry_funded", severity=0.5 if len(funded) == len(papers) else 0.3,
                             source_ids=[r.id for r in funded],
                             message="Supporting research was funded by a company with a financial interest."))

    # "clinically proven"
    if CERTAINTY.search(claim.text):
        strong_uncontested = verdict.status == EvidenceStatus.SUPPORTED and not verdict.contradicting and any(
            st.get("design") in STRONG and (r.quality and r.quality.tier == "A") for r, st in papers)
        if not strong_uncontested:
            flags.append(RedFlag(kind="overclaimed_certainty", severity=0.7, source_ids=[],
                                 message=f"Claims certainty (\"{CERTAINTY.search(claim.text).group()}\") that the "
                                         "evidence does not provide."))
    return sorted(flags, key=lambda f: -f.severity)
