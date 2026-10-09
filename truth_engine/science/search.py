"""One entry point for the scholarly sources: claim -> papers as Sources + Passages.

Responsibilities beyond calling the two APIs:
  * turn a claim into a keyword query PubMed can handle
  * merge the same paper seen through PubMed and OpenAlex (by PMID / DOI)
  * rewrite OpenAlex reference ids to canonical ids so citation edges connect
  * mark systematic reviews / meta-analyses as *derived from* the retrieved
    primary studies they cite: a review that pools a trial is not an
    independent line of evidence from that trial (echo detector)
"""
from __future__ import annotations

import re
import threading

from ..schemas import Passage, Source
from ..textutils import STOPWORDS, quantities, words
from .openalex import OpenAlexClient
from .papers import REVIEW_DESIGNS, PaperRecord, to_source
from .pubmed import HIGH_EVIDENCE_FILTER, PubMedClient
from .retractions import CrossrefClient

# Words that state the *direction* or *strength* of a finding. Dropped from
# queries so the search finds studies on the topic whatever they concluded.
FINDING_WORDS = frozenset("""
improve improves improved improving boost boosts boosted increase increases increased reduce reduces reduced
lower lowers lowered raise raises raised cause causes caused prevent prevents prevented cure cures cured
treat treats treated help helps helped enhance enhances enhanced decrease decreases decreased halve halves
halved cut cuts proven clinically shown shows show study studies science scientific research evidence
significantly significant effective effect effects benefit benefits link linked associated reverses
reverse reversed extend extends extended slash slashes risk percent works work really new better best
""".split())


def query_terms(text: str, max_terms: int = 6) -> list[str]:
    nums = {q.raw.lower() for q in quantities(text)}
    out = []
    for w in words(text):
        if w in STOPWORDS or w in FINDING_WORDS or len(w) < 3 or w in nums or re.fullmatch(r"[\d.,%]+", w):
            continue
        if w not in out:
            out.append(w)
    return out[:max_terms]


class ScienceSearch:
    def __init__(self, pubmed: PubMedClient | None = None, openalex: OpenAlexClient | None = None,
                 crossref: CrossrefClient | None = None, enrich: bool = True):
        self.pubmed = pubmed or PubMedClient()
        self.openalex = openalex or OpenAlexClient()
        self.crossref = crossref or CrossrefClient(offline=self.openalex.http.offline)
        self.enrich = enrich        # journal signals + Retraction Watch lookups (2 extra batched calls)
        self.sources: dict[str, Source] = {}          # everything seen, for the citation graph
        self._aliases: dict[str, str] = {}             # "openalex:W…" -> canonical source id
        self._memo: dict[tuple, list[tuple[Source, list[Passage]]]] = {}
        self._lock = threading.RLock()      # claims run in parallel: ingest mutates shared maps

    @property
    def errors(self) -> list[str]:
        return self.pubmed.errors + self.openalex.errors + self.crossref.errors

    # --------------------------------------------------------------- search
    def search(self, text: str, *, k: int = 5, use: tuple[str, ...] = ("pubmed", "openalex"),
               high_evidence: bool = False) -> list[tuple[Source, list[Passage]]]:
        """`high_evidence` restricts to trials / systematic reviews / meta-analyses:
        the studies best placed to *refute* a claim (used by falsification search)."""
        memo_key = (text, k, tuple(use), high_evidence)
        if memo_key in self._memo:
            return self._memo[memo_key]
        terms = query_terms(text)
        records: list[PaperRecord] = []
        if terms and "pubmed" in use:
            records += self._pubmed(terms, k, high_evidence)
        if terms and "openalex" in use:
            filt = "type:review" if high_evidence else "type:review|article"
            records += self.openalex.search(" ".join(terms), k=k, filter=filt)
        out = self._ingest(records)
        self._memo[memo_key] = out
        return out

    def _pubmed(self, terms: list[str], k: int, high_evidence: bool) -> list[PaperRecord]:
        # Back off to fewer terms when the full AND-query finds nothing.
        for n in sorted({len(terms), min(4, len(terms)), min(3, len(terms)), min(2, len(terms))}, reverse=True):
            term = " ".join(terms[:n])
            if high_evidence:
                term = f"({term}) AND {HIGH_EVIDENCE_FILTER}"
            pmids = self.pubmed.search(term, retmax=k)
            if pmids:
                return self.pubmed.fetch(pmids)
        return []

    def lookup(self, ident: str) -> tuple[Source, list[Passage]] | None:
        """Fetch one paper by 'pmid:…', 'doi:…' or OpenAlex 'W…' id."""
        recs: list[PaperRecord] = []
        if ident.startswith("pmid:"):
            recs = self.pubmed.fetch([ident.split(":", 1)[1]])
        oa = self.openalex.work(ident)
        if oa:
            recs.append(oa)
        res = self._ingest(recs)
        return res[0] if res else None

    def cited_by(self, source_id: str, k: int = 10) -> list[tuple[Source, list[Passage]]]:
        """Papers citing this one (OpenAlex): later replications, rebuttals, reviews."""
        src = self.sources.get(source_id)
        oa_id = (src.metadata.get("openalex_id") if src else None) or (
            source_id.split(":", 1)[1] if source_id.startswith("openalex:") else None)
        if not oa_id:
            return []
        return self._ingest(self.openalex.cited_by(oa_id, k))

    # --------------------------------------------------------------- merging
    def snapshot(self) -> dict[str, Source]:
        with self._lock:
            return dict(self.sources)

    def _ingest(self, records: list[PaperRecord]) -> list[tuple[Source, list[Passage]]]:
        with self._lock:
            return self._ingest_locked(records)

    def _ingest_locked(self, records: list[PaperRecord]) -> list[tuple[Source, list[Passage]]]:
        merged: dict[str, PaperRecord] = {}
        by_key: dict[str, str] = {}
        for r in records:
            keys = [k for k in (r.pmid and f"pmid:{r.pmid}", r.doi and f"doi:{r.doi.lower()}",
                                r.openalex_id and f"oa:{r.openalex_id}") if k]
            hit = next((by_key[k] for k in keys if k in by_key), None)
            if hit:
                merged[hit].merge(r)
                rec = merged[hit]
            else:
                rec = r
                merged[rec.source_id] = rec
                hit = rec.source_id
            for k in keys:
                by_key[k] = hit
        if self.enrich:
            self._enrich(list(merged.values()))
        out = []
        for rec in merged.values():
            if rec.openalex_id:
                self._aliases[f"openalex:{rec.openalex_id}"] = rec.source_id
            src, passages = to_source(rec)
            prev = self.sources.get(src.id)
            if prev:   # keep citation edges learned earlier
                src.cites = list(dict.fromkeys(prev.cites + src.cites))
                src.derived_from = list(dict.fromkeys(prev.derived_from + src.derived_from))
            self.sources[src.id] = src
            out.append((src, passages))
        self._resolve_edges()
        return out

    def _enrich(self, recs: list[PaperRecord]) -> None:
        """Attach journal records (OpenAlex) and editorial notices (Crossref /
        Retraction Watch). Both are batched and cached; failures leave gaps."""
        need = [r for r in recs if r.venue is None]
        venues = self.openalex.venues(ids=[r.venue_id for r in need if r.venue_id],
                                      issns=[r.issn for r in need if r.issn and not r.venue_id])
        by_id = {v.openalex_id: v for v in venues}
        by_issn = {i: v for v in venues for i in v.issns}
        for r in need:
            r.venue = by_id.get(r.venue_id) or by_issn.get(r.issn)
        notices = self.crossref.notices([r.doi for r in recs if r.doi and not r.notices])
        for r in recs:
            if r.doi and r.doi.lower() in notices:
                r.notices = notices[r.doi.lower()]

    def _resolve_edges(self) -> None:
        for src in self.sources.values():
            src.cites = list(dict.fromkeys(self._aliases.get(c, c) for c in src.cites))
            design = (src.metadata.get("study") or {}).get("design")
            if design in REVIEW_DESIGNS - {"narrative_review"}:
                pooled = [c for c in src.cites if c in self.sources and self.sources[c].source_type == "paper"]
                src.derived_from = list(dict.fromkeys(src.derived_from + pooled))


_science: ScienceSearch | None = None


def get_science() -> ScienceSearch:
    global _science
    if _science is None:
        _science = ScienceSearch()
    return _science
