"""Shared, dependency-free text processing used by retrieval, claim extraction,
verification and critique. Deliberately transparent: every signal the engine
uses to judge evidence can be traced to a function in this file."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

STOPWORDS = frozenset("""
a an the and or but if then else of to in on at by for with from as is are was were be been being
this that these those it its it's he she they them his her their we our you your i me my do does did
done have has had having not no nor so than too very can could should would will shall may might must
about above after again against all am any because before below between both during each few further
here how into more most other out over own same some such through under until up what when where which
while who whom why also just only there s t don now via per one ever according said says
""".split())

NEGATIONS = frozenset({"not", "no", "never", "none", "neither", "nor", "without", "cannot", "can't",
                       "didn't", "doesn't", "isn't", "wasn't", "weren't", "aren't", "won't", "hasn't",
                       "haven't", "hadn't", "failed", "unchanged", "nobody", "nothing"})

# Pairs of words whose substitution flips the meaning of a claim.
ANTONYMS = [
    ("increase", "decrease"), ("increased", "decreased"), ("rise", "fall"), ("rose", "fell"),
    ("grew", "declined"), ("growth", "decline"), ("gain", "loss"), ("up", "down"), ("higher", "lower"),
    ("improve", "worsen"), ("improved", "worsened"), ("improves", "worsens"), ("more", "less"),
    ("success", "failure"), ("visible", "invisible"), ("true", "false"), ("stable", "unstable"),
    ("significant", "insignificant"), ("supports", "contradicts"), ("early", "late"), ("before", "after"),
    ("largest", "smallest"), ("tallest", "shortest"), ("longest", "shortest"), ("first", "last"),
    ("launched", "discontinued"), ("approved", "rejected"), ("confirmed", "refuted"),
    ("stable", "declined"), ("stable", "decline"), ("steady", "declined"), ("stable", "collapsed"),
    ("grew", "collapsed"), ("unchanged", "changed"), ("on time", "late"),
]
_ANT = {}
for a, b in ANTONYMS:
    _ANT.setdefault(a, set()).add(b)
    _ANT.setdefault(b, set()).add(a)

CAUSAL_MARKERS = re.compile(
    r"\b(because|due to|caused by|causes?|caused|led to|leads to|resulted in|results? in|driven by|"
    r"explains?|explained by|owing to|as a result of|attributable to|responsible for|therefore|thus|"
    r"so that|consequently|primary (?:contributing )?factor|main reason|reason (?:was|is|for))\b", re.I)

HEDGES = re.compile(r"\b(may|might|could|possibly|probably|likely|perhaps|suggests?|appears?|seems?|"
                    r"believed?|estimated?|approximately|about|around|roughly|reportedly|unclear)\b", re.I)

MONTHS = {m: i + 1 for i, m in enumerate(
    "january february march april may june july august september october november december".split())}
MONTHS.update({m[:3]: i for m, i in list(MONTHS.items())})

_word_re = re.compile(r"[A-Za-z][A-Za-z'\-]*|\d+(?:[.,]\d+)*")


def normalize(text: str) -> str:
    text = text.replace("’", "'").replace("–", "-").replace("—", "-")
    text = re.sub(r"[à-ÿ]", lambda m: _deaccent(m.group()), text)
    return text


def _deaccent(ch: str) -> str:
    import unicodedata
    return unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode() or ch


def stem(w: str) -> str:
    """Tiny suffix stripper: enough to match launch/launched/launches, decline/declined."""
    w = w.lower()
    if len(w) <= 3 or w.isdigit():
        return w
    for suf in ("ational", "ization", "fulness", "ousness", "iveness", "ements", "ement", "ments",
                "ment", "ingly", "edly", "ities", "ity", "ings", "ing", "ies", "ied", "ers", "er", "ed", "es", "ly", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)]
            if suf in ("ies", "ied"):
                w += "y"
            break
    if len(w) > 4 and w.endswith("e"):
        w = w[:-1]
    return w


def words(text: str) -> list[str]:
    return [w.lower() for w in _word_re.findall(normalize(text))]


def content_tokens(text: str) -> list[str]:
    return [stem(w) for w in words(text) if w not in STOPWORDS and len(w) > 1]


def sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    # protect decimals, abbreviations and initials
    protected = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", text)
    protected = re.sub(r"\b(e\.g|i\.e|etc|vs|Dr|Mr|Mrs|Ms|St|No|Inc|Ltd|Co|al)\.", r"\1<DOT>", protected)
    protected = re.sub(r"\b([A-Z])\.", r"\1<DOT>", protected)
    parts = re.split(r"(?<=[.!?])\s+(?=[\"'A-Z0-9(])", protected)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]


# --------------------------------------------------------------------------- #
# Numbers, dates and quantities
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Quantity:
    value: float
    unit: str          # "", "%", "year", "date", "usd", "unit", "km", "m", "day" ...
    raw: str

    def close_to(self, other: "Quantity", rel: float = 0.02) -> bool:
        if self.unit and other.unit and self.unit != other.unit and not ({self.unit, other.unit} <= {"", "unit"}):
            return False
        if self.value == other.value:
            return True
        denom = max(abs(self.value), abs(other.value), 1e-9)
        return abs(self.value - other.value) / denom <= rel


_SCALE = {"thousand": 1e3, "k": 1e3, "million": 1e6, "m": 1e6, "mn": 1e6, "billion": 1e9, "bn": 1e9, "b": 1e9,
          "trillion": 1e12}
_UNIT_WORDS = {
    "%": "%", "percent": "%", "per cent": "%", "percentage points": "pp",
    "km": "km", "kilometres": "km", "kilometers": "km", "metres": "m", "meters": "m", "metre": "m", "meter": "m",
    "days": "day", "day": "day", "units": "unit", "unit": "unit", "adults": "people", "people": "people",
    "participants": "people", "stores": "store", "microseconds": "us", "minutes": "min", "seconds": "s",
}
_num_re = re.compile(
    r"(?P<cur>\$|USD\s?|EUR\s?|£|€)?(?<![A-Za-z\d.\-])(?P<num>-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)"
    r"\s*(?P<scale>thousand|million|billion|trillion|bn|mn|[kKmMbB](?![a-zA-Z]))?"
    r"\s*(?P<unit>%|percent|per cent|percentage points|kilometres|kilometers|km|metres|meters|metre|meter|"
    r"days?|units?|adults|people|participants|stores|microseconds|minutes|seconds)?", re.I)


def quantities(text: str) -> list[Quantity]:
    out: list[Quantity] = []
    t = normalize(text)
    # full dates first: "March 15, 2022", "15 March 2022", "2026-09-14"
    for m in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", t):
        out.append(Quantity(float(m.group(1) + m.group(2) + m.group(3)), "date", m.group()))
    for m in re.finditer(r"\b([A-Z][a-z]+)\s+(\d{1,2}),?\s+(\d{4})\b", t):
        mo = MONTHS.get(m.group(1).lower())
        if mo:
            out.append(Quantity(float(f"{m.group(3)}{mo:02d}{int(m.group(2)):02d}"), "date", m.group()))
    for m in re.finditer(r"\b(\d{1,2})\s+([A-Z][a-z]+)\s+(\d{4})\b", t):
        mo = MONTHS.get(m.group(2).lower())
        if mo:
            out.append(Quantity(float(f"{m.group(3)}{mo:02d}{int(m.group(1)):02d}"), "date", m.group()))
    date_spans = [q.raw for q in out]
    scrub = t
    for raw in date_spans:
        scrub = scrub.replace(raw, " ")
    for m in _num_re.finditer(scrub):
        raw = m.group(0).strip()
        num_s = m.group("num").replace(",", "")
        try:
            v = float(num_s)
        except ValueError:
            continue
        scale = (m.group("scale") or "").lower()
        unit_w = (m.group("unit") or "").lower()
        if m.group("cur"):
            unit = "usd"
        else:
            unit = _UNIT_WORDS.get(unit_w, _UNIT_WORDS.get(unit_w.rstrip("s"), ""))
        if scale in _SCALE and not (scale in ("m",) and unit == ""
                                    and re.match(r"\s*m(etres|eters)?\b", scrub[m.end("num"):m.end("num") + 7] or "")):
            v *= _SCALE[scale]
        if not unit and not scale and 1000 <= v <= 2100 and float(v).is_integer() and "," not in m.group("num"):
            unit = "year"
        out.append(Quantity(round(v, 6), unit, raw))
    return out


def years(text: str) -> set[int]:
    ys = {int(q.value) for q in quantities(text) if q.unit == "year"}
    ys |= {int(str(int(q.value))[:4]) for q in quantities(text) if q.unit == "date"}
    return ys


# --------------------------------------------------------------------------- #
# Entities
# --------------------------------------------------------------------------- #

_ent_re = re.compile(r"\b(?:[A-Z][a-zA-Z0-9\-]*|[A-Z]{2,}\d*)(?:\s+(?:of|de|van|von|the|and)?\s*(?:[A-Z][a-zA-Z0-9\-]*|\d+))*")
_SENTENCE_STARTERS = frozenset({"the", "a", "an", "in", "on", "at", "it", "this", "that", "these", "those", "there",
                                "however", "but", "and", "since", "after", "before", "during", "while", "when",
                                "although", "our", "we", "they", "he", "she", "its", "their", "what", "why", "how",
                                "which", "who", "where", "is", "are", "was", "were", "did", "does", "do", "can",
                                "posted", "reply", "q7", "q8", "fun", "yes", "no", "evidence", "current", "several",
                                "many", "some", "most", "prove", "explain", "find", "tell", "give", "list"})


def entities(text: str) -> list[str]:
    ents: list[str] = []
    for s in sentences(normalize(text)) or [text]:
        for m in _ent_re.finditer(s):
            e = m.group().strip()
            first = e.split()[0].lower()
            if m.start() == 0 and first in _SENTENCE_STARTERS:
                rest = e.split()[1:]
                if not rest:
                    continue
                e = " ".join(rest)
            if e.lower() in STOPWORDS or len(e) < 2 or e.isdigit():
                continue
            ents.append(e)
    seen, out = set(), []
    for e in ents:
        k = e.lower()
        if k not in seen:
            seen.add(k)
            out.append(e)
    return out


# --------------------------------------------------------------------------- #
# Similarity and polarity
# --------------------------------------------------------------------------- #

def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def coverage(claim_tokens: list[str], passage_tokens: list[str]) -> float:
    """Fraction of the claim's content tokens that appear in the passage."""
    if not claim_tokens:
        return 0.0
    ps = set(passage_tokens)
    return sum(1 for t in set(claim_tokens) if t in ps) / len(set(claim_tokens))


def has_negation(text: str) -> bool:
    ws = set(words(text))
    return bool(ws & NEGATIONS) or "n't" in text.lower()


def antonym_conflict(a: str, b: str) -> list[tuple[str, str]]:
    wa, wb = set(words(a)), set(words(b))
    hits = []
    for w in wa:
        for opp in _ANT.get(w, ()):
            if opp in wb and w not in wb:
                hits.append((w, opp))
    return hits


def is_causal(text: str) -> bool:
    return bool(CAUSAL_MARKERS.search(text))


def hedged(text: str) -> bool:
    return bool(HEDGES.search(text))


def cosine(a, b) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    return num / (da * db) if da and db else 0.0


def short(text: str, n: int = 160) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"
