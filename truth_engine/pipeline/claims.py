"""Claim extraction for audit mode (CONCEPTS.md §4: claim decomposition,
check-worthiness).

Text -> sentences -> atomic claims, each one checkable on its own:

    "The company launched Product X in 2022, sold 4 million units, and
     discontinued it in 2024 because demand declined."
      C1 The company launched Product X in 2022.
      C2 The company sold 4 million units.
      C3 The company discontinued it in 2024.
      C4 The company discontinued it in 2024 because demand declined.  (causal)

Rules, in order:
  1. clause split on ";" and contrast words (but / while / whereas)
  2. predicate split: "X verb1 ..., verb2 ..., and verb3 ..." -> one claim per
     verb, each with the shared subject "X"
  3. causal split: "A because B" -> the fact A, plus the causal claim A-because-B
  4. pronoun subjects ("It improves...") take the previous sentence's subject
Each claim keeps the character span of its sentence so the UI can highlight
it, and gets a check-worthiness score: opinions, questions, instructions and
filler are skipped. With an LLM available, extraction can be delegated to it
(messy prose); kind tagging and spans still come from these rules.
"""
from __future__ import annotations

import re

from ..schemas import Claim, ClaimKind
from ..science.study_design import ANIMAL_TEXT, IN_VITRO_TEXT
from ..textutils import CAUSAL_MARKERS, content_tokens, entities, quantities, sentences, words

CHECK_WORTHY_THRESHOLD = 0.45

AUX = {"is", "are", "was", "were", "has", "have", "had", "can", "may", "might", "could", "will", "would",
       "does", "did", "do", "should", "must"}
VERBS = set("""
improves improve improved reduces reduce reduced increases increase increased causes cause caused prevents prevent
prevented cures cure cured treats treat treated helps help helped boosts boost boosted lowers lower lowered raises
raise raised enhances enhance enhanced decreases decrease decreased cuts cut halves halved extends extend extended
reverses reverse reversed grows grow grew protects protect protected supports support contains contain affects affect
works work sold sells sell launched launches leads lead led results doubles doubled triples slashes slashed kills kill
burns burn fights fight blocks block slows slow speeds stops stop makes make made gives give delays delay won wins
discovered developed invented built founded received died born became becomes remains remained reached reaches shows
""".split()) | AUX
NOT_VERBS = {"red", "bed", "need", "seed", "speed", "breed", "feed", "hundred", "weed", "shed", "embed"}
PRONOUN_SUBJECTS = re.compile(r"^(?:it|he|she|they|this|these|the supplement|the drug|the product|the pill|the treatment|"
                              r"the compound|this supplement|this product)\b", re.I)

SAFETY = re.compile(r"\b(?:safe|safety|side effects?|adverse|toxic\w*|harmful|harmless|dangerous|overdose|"
                    r"contraindicat\w*)\b", re.I)
ASSOCIATION = re.compile(r"\b(?:associated with|linked (?:to|with)|correlat\w*|tied to|people who|those who|"
                         r"drinkers|users (?:had|have)|more likely|less likely|"
                         # epidemiological risk wording states an association, not a mechanism
                         r"(?:increased|elevated|higher|greater|lower|reduced) risk|increases? the risk|"
                         r"at (?:higher|increased|greater|lower) risk|predict(?:s|or|ors|ive)|marker (?:of|for)|"
                         r"(?:can be )?described by)\b", re.I)
EFFICACY = re.compile(r"\b(?:improv\w*|reduc\w*|boost\w*|increas\w*|lower\w*|rais(?:e|es|ed)|prevent\w*|"
                      r"cur(?:e|es|ed)|treat\w*|help\w*|enhanc\w*|effective|works?|extends?|reverses?|cuts?|"
                      r"halv\w*|slows?|protects?|fights?|burns?|slash\w*)\b", re.I)
MECHANISM = re.compile(r"\b(?:mechanism|activat\w*|inhibit\w*|binds?|receptors?|pathways?|neurons?|brain cells|"
                       r"cells|mitochondria\w*|enzymes?|hormones?|by (?:increasing|decreasing|blocking|boosting))\b",
                       re.I)
CAUSAL_VERBS = re.compile(r"\b(?:causes?|caused|leads? to|results? in)\b", re.I)
CLAIM_HEDGE = re.compile(r"\b(?:may|might|could|possibly|suggests?|appears? to|seems? to|potentially|preliminary)\b",
                         re.I)
CERTAINTY = re.compile(r"\b(?:proven|clinically proven|science shows|studies (?:show|prove)|guaranteed|definitely|"
                       r"scientifically|clinically)\b", re.I)
OPINION = re.compile(r"\b(?:I|we|my|our)\s+(?:think|feel|believe|love|like|hope|recommend)\b|\bin my (?:opinion|view)\b|"
                     r"\b(?:delicious|amazing|awesome|wonderful|beautiful|terrible)\b", re.I)
IMPERATIVE = re.compile(r"^(?:buy|try|order|click|call|subscribe|ask|talk|consult|join|visit|see|read|sign up|"
                        r"learn|discover|get)\b", re.I)


# --------------------------------------------------------------------------- sentence spans
def sentence_spans(text: str) -> list[tuple[int, int, str]]:
    """Sentences with their character offsets in the original text."""
    out, cursor = [], 0
    for s in sentences(text):
        pattern = r"\s+".join(re.escape(tok) for tok in s.split())
        m = re.compile(pattern).search(text, cursor)
        if m:
            out.append((m.start(), m.end(), s))
            cursor = m.end()
        else:
            out.append((cursor, cursor, s))
    return out


# --------------------------------------------------------------------------- splitting
def is_verb(w: str) -> bool:
    w = w.lower()
    return w in VERBS or (w.endswith("ed") and len(w) > 4 and w not in NOT_VERBS)


def _first_verb(tokens: list[str]) -> int | None:
    for i, t in enumerate(tokens):
        if i > 0 and is_verb(re.sub(r"\W", "", t)):
            return i
    return None


def split_clauses(sentence: str) -> list[str]:
    parts = re.split(r";\s+|,\s+(?:but|while|whereas|yet)\s+|\s+(?:but|whereas)\s+", sentence)
    return [p.strip(" .") for p in parts if p and p.strip(" .")]


def split_predicates(clause: str) -> list[str]:
    """'X launched A, sold B, and discontinued C' -> ['X launched A', 'X sold B', 'X discontinued C']."""
    pieces = re.split(r",\s+(?:and\s+)?|\s+and\s+", clause)
    tokens = clause.split()
    v = _first_verb(tokens)
    if v is None or len(pieces) == 1:
        return [clause]
    subject = " ".join(tokens[:v])
    merged: list[str] = [pieces[0]]
    for piece in pieces[1:]:
        first = re.sub(r"\W", "", piece.split()[0]) if piece.split() else ""
        own_verb = _first_verb(piece.split())
        if first and is_verb(first):
            merged.append(f"{subject} {piece}")                       # shared-subject predicate
        elif own_verb is not None and own_verb <= 4 and len(piece.split()) >= 3:
            merged.append(piece)                                       # a clause with its own subject
        else:
            merged[-1] = f"{merged[-1]} and {piece}"                   # noun list: "memory and focus"
    return merged


def split_causal(claim: str) -> list[tuple[str, bool]]:
    """'A because B' -> [('A', False), ('A because B', True)]."""
    m = re.search(r"\s+(?:because(?: of)?|due to|as a result of|caused by|driven by|owing to)\s+", claim, re.I)
    if not m or m.start() < 8:
        return [(claim, False)]
    return [(claim[:m.start()], False), (claim, True)]


def _tidy(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip(" ,;:.")
    text = re.sub(r"^(?:and|but|so|also)\s+", "", text, flags=re.I)
    end = "" if text.endswith(("?", "!")) else "."
    return text[:1].upper() + text[1:] + end if text else ""


# --------------------------------------------------------------------------- tagging
def science_kind(text: str) -> str:
    if SAFETY.search(text):
        return "safety"
    if ASSOCIATION.search(text):
        return "association"
    if EFFICACY.search(text):
        return "efficacy"
    if MECHANISM.search(text):
        return "mechanism"
    if CAUSAL_MARKERS.search(text) or CAUSAL_VERBS.search(text):
        return "causal"
    if any(q.unit in ("%", "people", "pp") for q in quantities(text)):
        return "statistic"
    return "general"


PEOPLE = re.compile(r"\b(?:people|persons?|humans?|adults?|patients?|men|women|children|kids|infants?|babies|"
                    r"elderly|older adults|pregnan\w*|individuals|participants|population|you|your|consumers|"
                    r"smokers|drinkers|athletes|students|brain|brains|heart|memory|sleep|skin|hair|body|"
                    r"weight|lifespan|mood|focus|energy)\b|\bin (?:the )?(?:UK|US|USA|United \w+|[A-Z][a-z]+ia)\b",
                    re.I)


def population(text: str) -> str:
    """Who the claim is about. Health-effect and safety claims ("X improves sleep")
    are implicitly about people; molecular / mechanism claims ("ADAR1 binds
    Dicer") are about no population, so animal or cell evidence isn't penalized."""
    if IN_VITRO_TEXT.search(text):
        return "in_vitro"
    if ANIMAL_TEXT.search(text):
        return "animal"
    if PEOPLE.search(text) or science_kind(text) in ("efficacy", "safety", "association"):
        return "human"
    return "unspecified"


def check_worthiness(text: str) -> float:
    s = 0.3
    if quantities(text):
        s += 0.25
    if entities(text):
        s += 0.2
    if EFFICACY.search(text) or SAFETY.search(text) or ASSOCIATION.search(text) or MECHANISM.search(text) \
            or CERTAINTY.search(text):
        s += 0.25
    if CAUSAL_MARKERS.search(text) or CAUSAL_VERBS.search(text):
        s += 0.15
    if text.rstrip().endswith("?"):
        s -= 0.5
    if OPINION.search(text):
        s -= 0.4
    if IMPERATIVE.search(text.strip()):
        s -= 0.4
    if len(content_tokens(text)) < 3:
        s -= 0.3
    return round(max(0.0, min(1.0, s)), 2)


def make_claim(text: str, *, sentence: str = "", span=None, causal: bool = False) -> Claim:
    kind_s = science_kind(text)
    qs = quantities(text)
    is_c = causal or kind_s in ("efficacy", "causal") or bool(CAUSAL_MARKERS.search(text))
    kind = ClaimKind.CAUSAL if causal or kind_s == "causal" else \
        ClaimKind.NUMERIC if any(q.unit not in ("year", "date") for q in qs) else \
        ClaimKind.TEMPORAL if qs else \
        ClaimKind.INFERENTIAL if CLAIM_HEDGE.search(text) else ClaimKind.FACTUAL
    return Claim(text=text, kind=kind, numbers=[q.value for q in qs], entities=entities(text), is_causal=is_c,
                 sentence=sentence or text, span=span, science_kind=kind_s, population=population(text),
                 hedged=bool(CLAIM_HEDGE.search(text)), check_worthiness=check_worthiness(text),
                 importance=min(1.0, 0.4 + 0.3 * bool(qs) + 0.3 * is_c))


# --------------------------------------------------------------------------- entry points
def extract_claims(text: str, *, include_skipped: bool = False, llm=None) -> list[Claim]:
    """Atomic claims, in reading order. By default only check-worthy ones.
    `llm` is an optional (provider, ModelSpec) pair used for messy prose."""
    if llm is not None:
        got = _llm_extract(text, *llm)
        if got is not None:
            return [c for c in got if include_skipped or c.check_worthiness >= CHECK_WORTHY_THRESHOLD]
    out: list[Claim] = []
    last_subject: str | None = None
    for start, end, sent in sentence_spans(text):
        ents = entities(sent)
        for ci, clause in enumerate(split_clauses(sent)):
            # "...; he was born in Ulm" -> this sentence's subject; "It also..." -> the previous one's
            subject = ents[0] if ci > 0 and ents else last_subject
            for pred in split_predicates(clause):
                if subject and PRONOUN_SUBJECTS.match(pred):
                    pred = PRONOUN_SUBJECTS.sub(subject, pred, count=1)
                for piece, causal in split_causal(pred):
                    t = _tidy(piece)
                    if len(words(t)) >= 2:
                        out.append(make_claim(t, sentence=sent, span=(start, end), causal=causal))
        if ents:
            last_subject = ents[0]
    seen, uniq = set(), []
    for c in out:
        k = c.text.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    return [c for c in uniq if include_skipped or c.check_worthiness >= CHECK_WORTHY_THRESHOLD]


LLM_SYSTEM = """You split text into atomic, independently checkable factual claims.
Rules: one fact per claim; repeat the subject in every claim (no pronouns); keep numbers exactly;
for "A because B" output both "A" and the causal claim "A because B"; skip opinions, questions,
instructions and filler. Return JSON: {"claims": [{"claim": "...", "sentence": <0-based sentence index>}]}"""


def _llm_extract(text: str, provider, spec) -> list[Claim] | None:
    spans = sentence_spans(text)
    numbered = "\n".join(f"[{i}] {s}" for i, (_, _, s) in enumerate(spans))
    try:
        data, _ = provider.complete_json(spec, LLM_SYSTEM, numbered)
    except Exception:
        return None
    items = (data or {}).get("claims") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return None
    out = []
    for it in items:
        if not isinstance(it, dict) or not it.get("claim"):
            continue
        idx = it.get("sentence")
        start, end, sent = spans[idx] if isinstance(idx, int) and 0 <= idx < len(spans) else (None, None, "")
        out.append(make_claim(_tidy(it["claim"]), sentence=sent, span=(start, end) if start is not None else None,
                              causal=bool(CAUSAL_MARKERS.search(it["claim"]))))
    return out
