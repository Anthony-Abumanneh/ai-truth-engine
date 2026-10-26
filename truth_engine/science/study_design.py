"""Study-design grading (CONCEPTS.md §5: evidence hierarchy, study designs,
sample size, preregistration, conflicts of interest, retractions, generalization).

Given whatever we know about a paper — PubMed publication types and MeSH
terms when available, otherwise just its title and abstract — work out:

  * design      meta-analysis, systematic review, RCT, cohort, ... opinion
  * weight      where that design sits in the evidence hierarchy (0..1)
  * sample size, preregistration, funding source, species, retraction

Structured metadata (publication types, MeSH) beats text cues, so it is
checked first; regex cues are the fallback. Every decision is recorded in
`cues` so the UI can show *why* a paper was graded the way it was.
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field

# Evidence hierarchy: higher = stronger design for answering "does X cause Y in people?"
DESIGN_WEIGHT = {
    "meta_analysis": 1.0,
    "systematic_review": 0.95,
    "rct": 0.85,
    "cohort": 0.65,
    "case_control": 0.55,
    "cross_sectional": 0.45,
    "narrative_review": 0.4,
    "case_report": 0.3,
    "animal": 0.25,
    "in_vitro": 0.2,
    "opinion": 0.15,
    "news": 0.15,
    "marketing": 0.05,
    "unknown": 0.4,
}

DESIGN_LABEL = {
    "meta_analysis": "meta-analysis", "systematic_review": "systematic review", "rct": "randomized controlled trial",
    "cohort": "cohort study", "case_control": "case-control study", "cross_sectional": "cross-sectional study",
    "narrative_review": "narrative review", "case_report": "case report", "animal": "animal study",
    "in_vitro": "in-vitro (cell) study", "opinion": "opinion / editorial", "news": "news report",
    "marketing": "marketing material", "unknown": "unknown design",
}

# PubMed publication types -> design (checked in this order: strongest first)
PUBTYPE_DESIGN = [
    ("meta-analysis", "meta_analysis"),
    ("systematic review", "systematic_review"),
    ("randomized controlled trial", "rct"),
    ("pragmatic clinical trial", "rct"),
    ("observational study", "cohort"),
    ("case reports", "case_report"),
    ("editorial", "opinion"),
    ("comment", "opinion"),
    ("letter", "opinion"),
    ("news", "news"),
    ("review", "narrative_review"),
]

# Text cues, strongest first. Animal / in-vitro are handled by species detection.
TEXT_DESIGN = [
    (r"\bmeta-?analys[ie]s\b", "meta_analysis"),
    (r"\bsystematic(?:ally)? review", "systematic_review"),
    (r"\brandomi[sz]ed\b.{0,40}\b(?:trial|controlled|placebo|double-blind)|\bRCTs?\b|\bplacebo-controlled\b",
     "rct"),
    (r"\b(?:prospective|retrospective)?\s*cohort\b|\bfollowed (?:up )?for \d+ years\b|\blongitudinal study\b",
     "cohort"),
    (r"\bcase-control\b", "case_control"),
    (r"\bcross-sectional\b|\bsurvey of\b", "cross_sectional"),
    (r"\bcase report\b|\bwe report (?:a|the) case\b", "case_report"),
    (r"\b(?:narrative|literature) review\b", "narrative_review"),
    (r"\beditorial\b|\bcommentary\b|\bopinion\b|\bperspective\b", "opinion"),
]

ANIMAL_MESH = {"animals", "mice", "rats", "mice, inbred c57bl", "rats, sprague-dawley", "rats, wistar", "disease models, animal",
               "zebrafish", "drosophila melanogaster", "caenorhabditis elegans", "swine", "dogs", "primates"}
ANIMAL_TEXT = re.compile(r"\b(?:mice|mouse|murine|rats?|rodents?|zebrafish|drosophila|c\. elegans|"
                         r"nematodes?|primates|macaques?|piglets?|in vivo animal)\b", re.I)
IN_VITRO_TEXT = re.compile(r"\b(?:in vitro|cell lines?|cultured (?:cells|neurons)|cell cultures?|petri)\b", re.I)
HUMAN_TEXT = re.compile(r"\b(?:adults|patients|participants|volunteers|humans?|women|men|children|"
                        r"individuals|subjects)\b", re.I)

_SIZE_WORDS = r"adults|participants|patients|subjects|volunteers|individuals|people|women|men|children|" \
              r"infants|adolescents|students|athletes|older adults|healthy adults|mice|rats|animals"
_SIZE_RE = [
    re.compile(r"\b[nN]\s*=\s*(\d{1,3}(?:,\d{3})+|\d+)"),
    re.compile(r"\b(\d{1,3}(?:,\d{3})+|\d{2,})\s+(?:[a-z\-]+\s+){0,2}(?:" + _SIZE_WORDS + r")\b", re.I),
    re.compile(r"\b(?:enrolled|recruited|randomi[sz]ed|included)\s+(\d{1,3}(?:,\d{3})+|\d{2,})\b", re.I),
]

PREREG_YES = re.compile(r"\bpre-?registered\b|\bNCT\d{8}\b|\bISRCTN\d+\b|\bPROSPERO\b|\bCRD\d{8,}\b|"
                        r"\bclinicaltrials\.gov\b|\btrial registration\b|\bregistration (?:no|number)\b", re.I)
REVIEW_REGISTRY = re.compile(r"\bPROSPERO\b|\bCRD\d{8,}\b|\bregistration (?:no|number)\b", re.I)
PREREG_NO = re.compile(r"\bnot (?:pre-?registered|registered)\b|\bwas not preregistered\b", re.I)

INDUSTRY = re.compile(r"\b(?:funded|sponsored|supported|paid) by (?:the )?(?:supplement |drug |product )?"
                      r"(?:manufacturer|company|maker|industry)\b|\bindustry-(?:funded|sponsored)\b|"
                      r"\bmanufacturer-funded\b|\bemployees? of\b|\bcompany-sponsored\b", re.I)
PUBLIC = re.compile(r"\bpublicly funded\b|\bnational institutes? of health\b|\bNIH\b|\bgovernment grant\b|"
                    r"\bresearch council\b|\bpublic funding\b|\bnon-profit\b", re.I)


class StudyProfile(BaseModel):
    design: str = "unknown"
    weight: float = DESIGN_WEIGHT["unknown"]
    sample_size: int | None = None
    preregistered: bool | None = None
    funding: str | None = None          # "industry" | "public" | None (unknown)
    species: str | None = None          # "human" | "animal" | "in_vitro" | None
    retracted: bool = False
    cues: list[str] = Field(default_factory=list)

    @property
    def label(self) -> str:
        """Short human-readable summary, e.g. 'randomized controlled trial, n=612, preregistered'."""
        bits = [DESIGN_LABEL.get(self.design, self.design)]
        if self.sample_size:
            bits.append(f"n={self.sample_size}")
        if self.preregistered is True:
            bits.append("preregistered")
        elif self.preregistered is False:
            bits.append("not preregistered")
        if self.funding == "industry":
            bits.append("industry-funded")
        elif self.funding == "public":
            bits.append("publicly funded")
        if self.species in ("animal", "in_vitro") and self.design != self.species:
            bits.append(f"{self.species.replace('_', '-')} only")
        if self.retracted:
            bits.insert(0, "RETRACTED")
        return ", ".join(bits)


def _int(s: str) -> int:
    return int(s.replace(",", ""))


def _negated(text: str, pos: int) -> bool:
    return bool(re.search(r"\b(?:no|not|never|without|lack of|absence of)\b(?:\s+\w+){0,2}\s*$", text[max(0, pos - 30):pos], re.I))


def sample_size(text: str) -> int | None:
    """Largest plausible participant count stated in the text (n=48, '612 adults')."""
    found = []
    for rx in _SIZE_RE:
        for m in rx.finditer(text):
            n = _int(m.group(1))
            if 1900 <= n <= 2100 and "," not in m.group(1):   # a year, not a sample
                continue
            if 2 <= n <= 10_000_000:
                found.append(n)
    return max(found) if found else None


def detect_species(text: str, mesh: list[str] | None = None) -> tuple[str | None, str]:
    mesh_l = {m.lower() for m in (mesh or [])}
    if mesh_l:
        if "humans" in mesh_l:
            return "human", "MeSH: Humans"
        if mesh_l & ANIMAL_MESH:
            return "animal", f"MeSH: {sorted(mesh_l & ANIMAL_MESH)[0]}"
    if IN_VITRO_TEXT.search(text) and not HUMAN_TEXT.search(text):
        return "in_vitro", "text: in-vitro cue"
    if ANIMAL_TEXT.search(text) and not HUMAN_TEXT.search(text):
        return "animal", f"text: '{ANIMAL_TEXT.search(text).group()}'"
    if HUMAN_TEXT.search(text):
        return "human", "text: human participants"
    return None, ""


def classify(title: str = "", abstract: str = "", *, coi: str = "", pub_types: list[str] | None = None,
             mesh: list[str] | None = None, source_type: str | None = None,
             retracted: bool | None = None, metadata: dict | None = None) -> StudyProfile:
    """Grade one paper or document. `metadata` holds explicit front-matter values
    (design, sample_size, preregistered, funding, species, retracted) that win
    over anything inferred. `coi` is the paper's conflict-of-interest statement.

    Reviews describe *other* studies ("a small manufacturer-funded pilot..."),
    so for reviews only registry ids and the COI statement are trusted for
    preregistration and funding, never the abstract text."""
    md = metadata or {}
    text = f"{title}. {abstract}"
    cues: list[str] = []
    pts = [p.lower() for p in (pub_types or [])]

    # ---- design
    design = None
    if md.get("design"):
        design = str(md["design"]).lower()
        cues.append("front matter: design")
    if not design and source_type in ("news", "blog", "forum", "marketing"):
        design = {"news": "news", "marketing": "marketing"}.get(source_type, "opinion")
        cues.append(f"source type: {source_type}")
    if not design:
        for pt, d in PUBTYPE_DESIGN:
            if pt in pts:
                design = d
                cues.append(f"publication type: {pt}")
                break
    if not design:
        # title first (it names the design), then abstract; ignore negated
        # mentions such as "no randomized trial has been conducted"
        for part in (title, abstract):
            for rx, d in TEXT_DESIGN:
                m = next((m for m in re.finditer(rx, part, re.I) if not _negated(part, m.start())), None)
                if m:
                    design = d
                    cues.append(f"text: '{m.group().strip()}'")
                    break
            if design:
                break
    if not design and source_type == "review":
        design = "narrative_review"
        cues.append("source type: review")
    if not design and source_type in (None, "paper", "preprint") and abstract:
        # rules found nothing: a classifier trained on PubMed-tagged abstracts fills in
        # (science/design_classifier.py; abstains below its confidence bar)
        from .design_classifier import predict
        guess = predict(title, abstract)
        if guess:
            design = guess[0]
            cues.append(f"classifier: {guess[0]} (p={guess[1]:.2f})")

    # ---- species (an RCT in mice is an animal study)
    species = md.get("species") or None
    if species:
        cues.append("front matter: species")
    else:
        species, why = detect_species(text, mesh)
        if why:
            cues.append(why)
    if species in ("animal", "in_vitro") and design not in ("meta_analysis", "systematic_review", "news",
                                                             "marketing", "opinion"):
        design = species

    design = design or "unknown"

    # ---- sample size
    n = md.get("sample_size")
    try:
        n = int(n) if n not in (None, "") else None
    except ValueError:
        n = None
    if n is None and design not in ("news", "marketing", "opinion"):
        n = sample_size(text)
        if n:
            cues.append(f"text: sample size {n}")

    is_review = design in ("meta_analysis", "systematic_review", "narrative_review")
    own_text = coi if is_review else f"{text} {coi}"

    # ---- preregistration
    prereg = md.get("preregistered")
    if prereg not in (None, ""):
        prereg = str(prereg).lower() == "true"
    elif not is_review and PREREG_NO.search(text):
        prereg = False
        cues.append("text: not preregistered")
    elif (m := (REVIEW_REGISTRY if is_review else PREREG_YES).search(text)):
        prereg = True
        cues.append(f"text: '{m.group()}'")
    else:
        prereg = None

    # ---- funding / conflicts of interest
    funding = md.get("funding") or None
    if funding:
        funding = "industry" if str(funding).lower() in ("manufacturer", "industry", "company") else \
            "public" if str(funding).lower() in ("public", "government", "nih") else str(funding).lower()
    elif (m := INDUSTRY.search(own_text)):
        funding = "industry"
        cues.append(f"text: '{m.group()}'")
    elif (m := PUBLIC.search(own_text)):
        funding = "public"
        cues.append(f"text: '{m.group()}'")

    # ---- retraction
    is_retracted = bool(retracted) or "retracted publication" in pts or \
        str(md.get("retracted", "")).lower() == "true" or title.upper().startswith("RETRACTED")
    if is_retracted:
        cues.append("retracted")

    return StudyProfile(design=design, weight=DESIGN_WEIGHT.get(design, DESIGN_WEIGHT["unknown"]),
                        sample_size=n, preregistered=prereg, funding=funding, species=species,
                        retracted=is_retracted, cues=cues)
