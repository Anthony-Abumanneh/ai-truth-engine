"""OpenAlex: an open index of ~250M scholarly works (CONCEPTS.md §6).

What it adds on top of PubMed:
  * `is_retracted` for papers outside biomedicine too
  * `referenced_works` (what a paper cites) and `cites:` filters (who cites
    it) -> real edges for the citation graph / echo detector
  * study-design tags and funders

No key needed. Set OPENALEX_EMAIL to join the "polite pool" (faster, more reliable).
"""
from __future__ import annotations

import json

from ..config import settings
from .cache import CachedFetcher, JsonCache
from .papers import PaperRecord
from .venues import Venue, parse_source

BASE = "https://api.openalex.org"
FIELDS = ("id,doi,title,publication_year,publication_date,ids,type,is_retracted,cited_by_count,primary_location,"
          "referenced_works,abstract_inverted_index,mesh,funders,study_designs")

VENUE_FIELDS = "id,display_name,issn_l,issn,summary_stats,is_core,is_in_doaj,listed_in"

# OpenAlex study-design tags -> PubMed-style publication types (so one grader handles both)
DESIGN_TO_PUBTYPE = {
    "meta-analysis": "Meta-Analysis", "systematic review": "Systematic Review",
    "randomized controlled trial": "Randomized Controlled Trial", "observational study": "Observational Study",
    "case report": "Case Reports", "editorial": "Editorial",
}


def short_id(oa_url: str | None) -> str | None:
    return oa_url.rsplit("/", 1)[-1] if oa_url else None


def abstract_from_index(inv: dict | None) -> str:
    """OpenAlex ships abstracts as {word: [positions]}; rebuild the text."""
    if not inv:
        return ""
    slots: dict[int, str] = {}
    for word, positions in inv.items():
        for p in positions:
            slots[p] = word
    return " ".join(slots[i] for i in sorted(slots))


def parse_work(w: dict) -> PaperRecord:
    ids = w.get("ids") or {}
    pmid = short_id(ids.get("pmid"))
    doi = (w.get("doi") or ids.get("doi") or "").replace("https://doi.org/", "") or None
    loc = w.get("primary_location") or {}
    venue = loc.get("source") or {}
    journal = venue.get("display_name")
    pub_types = [DESIGN_TO_PUBTYPE[d["display_name"].lower()] for d in w.get("study_designs") or []
                 if d.get("display_name", "").lower() in DESIGN_TO_PUBTYPE]
    if w.get("type") == "review" and not pub_types:
        pub_types.append("Review")
    if w.get("is_retracted"):
        pub_types.append("Retracted Publication")
    title = (w.get("title") or w.get("display_name") or "").removeprefix("RETRACTED: ").removeprefix("RETRACTED ARTICLE: ")
    return PaperRecord(
        title=title, abstract=abstract_from_index(w.get("abstract_inverted_index")), pmid=pmid, doi=doi,
        openalex_id=short_id(w.get("id")), journal=journal, published=w.get("publication_date"),
        pub_types=pub_types, mesh=list(dict.fromkeys(m["descriptor_name"] for m in w.get("mesh") or [])),
        references=[f"openalex:{short_id(r)}" for r in w.get("referenced_works") or []],
        funders=[f.get("display_name") for f in w.get("funders") or [] if f.get("display_name")],
        retracted=bool(w.get("is_retracted")), cited_by_count=w.get("cited_by_count"),
        issn=venue.get("issn_l"), venue_id=short_id(venue.get("id")), origin="openalex")


class OpenAlexClient:
    def __init__(self, *, cache: JsonCache | None = None, client=None, offline: bool | None = None,
                 email: str | None = None):
        self.email = settings.openalex_email if email is None else email
        self.http = CachedFetcher("openalex", per_second=8, cache=cache, client=client, offline=offline)

    @property
    def errors(self) -> list[str]:
        return self.http.errors

    def _get(self, path: str, **params) -> dict | None:
        if self.email:
            params["mailto"] = self.email
        body = self.http.get(f"{BASE}{path}", params)
        if not body:
            return None
        try:
            return json.loads(body)
        except ValueError:
            self.http.errors.append(f"openalex: unparseable response for {path}")
            return None

    def _list(self, **params) -> list[PaperRecord]:
        data = self._get("/works", select=FIELDS, **params)
        return [parse_work(w) for w in (data or {}).get("results", [])]

    # ------------------------------------------------------------------ API
    def search(self, query: str, k: int = 8, filter: str | None = None) -> list[PaperRecord]:
        params = {"search": query, "per-page": k}
        if filter:
            params["filter"] = filter
        return self._list(**params)

    def work(self, ident: str) -> PaperRecord | None:
        """`ident` may be 'W…', 'pmid:…' or 'doi:…'."""
        data = self._get(f"/works/{ident}", select=FIELDS)
        return parse_work(data) if data and data.get("id") else None

    def venues(self, *, ids: list[str] = (), issns: list[str] = ()) -> list[Venue]:
        """Journal records (citation stats, curated-list membership), batched."""
        out = []
        for key, vals in (("openalex_id", sorted(set(ids))), ("issn", sorted(set(issns)))):
            for i in range(0, len(vals), 50):
                data = self._get("/sources", filter=f"{key}:" + "|".join(vals[i:i + 50]), select=VENUE_FIELDS,
                                 **{"per-page": 50})
                out += [parse_source(s) for s in (data or {}).get("results", [])]
        return out

    def cited_by(self, work_id: str, k: int = 10) -> list[PaperRecord]:
        return self._list(filter=f"cites:{work_id}", **{"per-page": k, "sort": "cited_by_count:desc"})

    def references(self, rec: PaperRecord, k: int = 25) -> list[PaperRecord]:
        ids = [r.split(":", 1)[1] for r in rec.references if r.startswith("openalex:")][:min(k, 50)]
        if not ids:
            return []
        return self._list(filter="openalex_id:" + "|".join(ids), **{"per-page": len(ids)})
