"""PubMed via NCBI E-utilities (CONCEPTS.md §6: scholarly APIs).

  esearch  term  -> list of PMIDs
  efetch   PMIDs -> XML with title, abstract, journal, date, publication
                    types, MeSH terms, reference list, retraction flags

No key needed (3 requests/s). Set NCBI_API_KEY for 10/s.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET

from ..config import settings
from .cache import CachedFetcher, JsonCache
from .papers import PaperRecord

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
MONTHS = {m: i + 1 for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split())}

# Publication-type filter for the strongest designs (used by falsification searches)
HIGH_EVIDENCE_FILTER = ("(randomized controlled trial[pt] OR meta-analysis[pt] OR systematic review[pt] "
                        "OR clinical trial[pt])")


class PubMedClient:
    def __init__(self, *, cache: JsonCache | None = None, client=None, offline: bool | None = None,
                 api_key: str | None = None):
        self.api_key = settings.ncbi_key if api_key is None else api_key
        self.http = CachedFetcher("pubmed", per_second=9 if self.api_key else 2.8, cache=cache,
                                  client=client, offline=offline)

    def _params(self, **p) -> dict:
        if self.api_key:
            p["api_key"] = self.api_key
        return p

    @property
    def errors(self) -> list[str]:
        return self.http.errors

    # ------------------------------------------------------------------ API
    def search(self, term: str, retmax: int = 8, sort: str = "relevance") -> list[str]:
        body = self.http.get(f"{BASE}/esearch.fcgi", self._params(db="pubmed", term=term, retmax=retmax,
                                                                   retmode="json", sort=sort))
        if not body:
            return []
        try:
            return json.loads(body)["esearchresult"].get("idlist", [])
        except (ValueError, KeyError):
            self.http.errors.append("pubmed: unparseable esearch response")
            return []

    def fetch(self, pmids: list[str]) -> list[PaperRecord]:
        if not pmids:
            return []
        body = self.http.get(f"{BASE}/efetch.fcgi", self._params(db="pubmed", id=",".join(pmids), retmode="xml"))
        if not body:
            return []
        return parse_efetch(body)

    def search_records(self, term: str, retmax: int = 8) -> list[PaperRecord]:
        return self.fetch(self.search(term, retmax))


# ---------------------------------------------------------------------- parsing
def _text(el: ET.Element | None) -> str:
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def _date(art: ET.Element) -> str | None:
    pd = art.find(".//Journal/JournalIssue/PubDate")
    if pd is None:
        return None
    year = pd.findtext("Year") or (pd.findtext("MedlineDate") or "")[:4]
    if not year.isdigit():
        return None
    mon = (pd.findtext("Month") or "1")
    mon = MONTHS.get(mon[:3].lower(), int(mon) if mon.isdigit() else 1)
    day = pd.findtext("Day") or "1"
    return f"{year}-{mon:02d}-{int(day) if day.isdigit() else 1:02d}"


def parse_efetch(xml_text: str) -> list[PaperRecord]:
    root = ET.fromstring(xml_text)
    out = []
    for pa in root.iter("PubmedArticle"):
        mc = pa.find("MedlineCitation")
        art = mc.find("Article") if mc is not None else None
        if art is None:
            continue
        pmid = mc.findtext("PMID")
        parts = []
        for ab in art.findall("Abstract/AbstractText"):
            t = _text(ab)
            if t:
                parts.append(t if t.endswith((".", "?", "!")) else t + ".")
        doi = next((_text(a) for a in pa.findall(".//PubmedData/ArticleIdList/ArticleId")
                    if a.get("IdType") == "doi"), None)
        pub_types = [_text(p) for p in art.findall("PublicationTypeList/PublicationType")]
        mesh = [_text(d) for d in mc.findall("MeshHeadingList/MeshHeading/DescriptorName")]
        refs = []
        for ref in pa.findall(".//PubmedData/ReferenceList/Reference"):
            rid = next((_text(a) for a in ref.findall("ArticleIdList/ArticleId") if a.get("IdType") == "pubmed"), None)
            if rid:
                refs.append(f"pmid:{rid}")
        corrections = {c.get("RefType") for c in mc.findall("CommentsCorrectionsList/CommentsCorrections")}
        agencies = [_text(g.find("Agency")) for g in art.findall("GrantList/Grant") if g.find("Agency") is not None]
        coi = _text(mc.find("CoiStatement"))
        out.append(PaperRecord(
            title=_text(art.find("ArticleTitle")), abstract=" ".join(parts), coi=coi, pmid=pmid, doi=doi,
            journal=_text(art.find("Journal/Title")) or None, published=_date(art), pub_types=pub_types,
            issn=mc.findtext("MedlineJournalInfo/ISSNLinking") or art.findtext("Journal/ISSN"),
            mesh=mesh, references=refs, funders=list(dict.fromkeys(agencies)),
            retracted="Retracted Publication" in pub_types or "RetractionIn" in corrections, origin="pubmed"))
    return out
