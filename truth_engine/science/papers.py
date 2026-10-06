"""A scholarly paper from any source (PubMed, OpenAlex), and its conversion
into the engine's Source + sentence Passages so it flows through retrieval,
source-quality scoring and the citation graph like any corpus document."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..retrieval.source_quality import score_source
from ..schemas import Passage, Source
from ..textutils import sentences
from .study_design import StudyProfile, classify
from .venues import Venue

PEER_REVIEWED_AUTHORITY = 0.75
REVIEW_DESIGNS = {"meta_analysis", "systematic_review", "narrative_review"}


@dataclass
class PaperRecord:
    title: str
    abstract: str = ""
    coi: str = ""                           # conflict-of-interest statement (graded, not quoted as evidence)
    pmid: str | None = None
    doi: str | None = None
    openalex_id: str | None = None          # "W2117847125"
    journal: str | None = None
    published: str | None = None            # "YYYY-MM-DD" (month/day default to 01)
    pub_types: list[str] = field(default_factory=list)
    mesh: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)   # source ids ("pmid:…" / "openalex:W…")
    funders: list[str] = field(default_factory=list)
    retracted: bool = False
    cited_by_count: int | None = None
    issn: str | None = None
    venue_id: str | None = None             # OpenAlex source id "S…"
    venue: Venue | None = None              # filled by ScienceSearch enrichment
    notices: list[dict] = field(default_factory=list)   # Crossref / Retraction Watch notices
    origin: str = "pubmed"                   # which API produced it

    @property
    def source_id(self) -> str:
        """Canonical id: PMID when known (both APIs share it), else OpenAlex, else DOI."""
        if self.pmid:
            return f"pmid:{self.pmid}"
        if self.openalex_id:
            return f"openalex:{self.openalex_id}"
        return f"doi:{self.doi}"

    @property
    def url(self) -> str | None:
        if self.pmid:
            return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"
        if self.doi:
            return f"https://doi.org/{self.doi}"
        if self.openalex_id:
            return f"https://openalex.org/{self.openalex_id}"
        return None

    @property
    def withdrawn(self) -> bool:
        return self.retracted or any(n["type"] in ("retraction", "withdrawal", "removal") for n in self.notices)

    def profile(self) -> StudyProfile:
        p = classify(self.title, self.abstract, coi=self.coi, pub_types=self.pub_types, mesh=self.mesh, retracted=self.withdrawn)
        if p.funding is None and self.funders:
            p.funding = "public" if any(_looks_public(f) for f in self.funders) else None
            if p.funding:
                p.cues.append(f"funder: {self.funders[0]}")
        return p

    def merge(self, other: "PaperRecord") -> "PaperRecord":
        """Fill gaps from the same paper seen through another API (e.g. OpenAlex
        adds retraction status and references to a PubMed record)."""
        self.abstract = self.abstract or other.abstract
        self.coi = self.coi or other.coi
        self.doi = self.doi or other.doi
        self.pmid = self.pmid or other.pmid
        self.openalex_id = self.openalex_id or other.openalex_id
        self.journal = self.journal or other.journal
        self.published = self.published or other.published
        self.pub_types = list(dict.fromkeys(self.pub_types + other.pub_types))
        self.mesh = list(dict.fromkeys(self.mesh + other.mesh))
        self.references = list(dict.fromkeys(self.references + other.references))
        self.funders = list(dict.fromkeys(self.funders + other.funders))
        self.retracted = self.retracted or other.retracted
        self.cited_by_count = self.cited_by_count if self.cited_by_count is not None else other.cited_by_count
        self.issn = self.issn or other.issn
        self.venue_id = self.venue_id or other.venue_id
        self.venue = self.venue or other.venue
        self.notices = self.notices or other.notices
        return self


def _looks_public(funder: str) -> bool:
    f = funder.lower()
    return any(k in f for k in ("national", "institute", "council", "foundation", "university", "ministry",
                                "nhs", "government", "agency", "european", "commission", "trust"))


def to_source(rec: PaperRecord) -> tuple[Source, list[Passage]]:
    prof = rec.profile()
    stype = "review" if prof.design in REVIEW_DESIGNS else "paper"
    # Venue quality is unknown for most papers, so authority is a peer-reviewed
    # baseline nudged by design: a meta-analysis outranks an unclassified paper.
    authority = 0.1 if prof.retracted else round(PEER_REVIEWED_AUTHORITY + 0.25 * (prof.weight - 0.4), 3)
    src = Source(
        id=rec.source_id, title=rec.title, url=rec.url, source_type=stype, publisher=rec.journal,
        published=rec.published, authority=authority, is_primary=stype == "paper",
        cites=[r for r in rec.references if r != rec.source_id],
        metadata={"origin": rec.origin, "pmid": rec.pmid, "doi": rec.doi, "openalex_id": rec.openalex_id,
                  "pub_types": rec.pub_types, "cited_by_count": rec.cited_by_count,
                  "study": prof.model_dump(), "study_label": prof.label,
                  "venue": rec.venue.summary() if rec.venue else None, "notices": rec.notices},
    )
    src.quality = score_source(src)
    # strip inline section labels ("RESULTS: ...") that some abstracts carry
    texts = [rec.title] + [re.sub(r"^[A-Z][A-Z /&]{2,30}:\s+", "", t) for t in sentences(rec.abstract)]
    passages = [Passage(id=f"{src.id}#s{i}", source_id=src.id, text=t, retrieval_method=rec.origin)
                for i, t in enumerate(texts) if len(t) > 20]
    return src, passages
