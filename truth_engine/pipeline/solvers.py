"""Independent solvers for Ask mode (spec §4, Improvement 3; CONCEPTS.md §7:
ensembles, correlated errors).

A question becomes candidate answers from solvers that work *differently*,
so their mistakes are less likely to coincide:

    claim     yes/no question -> the claim it asks about ("Does X improve Y?"
              -> "X improves Y."); the verifier then decides yes / no / unclear
    graph     multi-hop path search over the knowledge graph
    swift     single best keyword (BM25) sentence           (offline-swift)
    scholar   single best dense-vector sentence              (offline-scholar)
    sage      hybrid search, prefers high-quality sources    (offline-sage)
    LLMs      any configured model, asked for a short answer

Every candidate is only a *proposal*: Ask mode verifies it like an audited
claim before anything is said. `diversity()` measures (not assumes) how
independent the candidates are.
"""
from __future__ import annotations

import re

from ..providers import get_registry
from ..retrieval import get_graph, get_store
from ..schemas import CandidateAnswer, DiversityReport
from ..textutils import STOPWORDS, entities, jaccard, words
from .analyzer import is_question

AUX_DO = {"does": "s", "do": "", "did": "ed"}
AUX_BE = {"is", "are", "was", "were", "can", "could", "will", "would", "has", "have", "should", "may", "might"}
IRREGULAR_PAST = {"win": "won", "make": "made", "give": "gave", "grow": "grew", "lead": "led", "cause": "caused",
                  "find": "found", "take": "took", "become": "became", "have": "had"}
COMMON_VERBS = set("""improve reduce increase cause prevent cure treat help boost lower raise enhance decrease cut
extend reverse grow protect support contain affect work lead slow stop make give delay win discover develop invent
build found receive die become remain reach show have kill burn fight block speed""".split())


def _third_person(v: str) -> str:
    if v == "have":
        return "has"
    if v.endswith(("s", "sh", "ch", "x", "z", "o")):
        return v + "es"
    if v.endswith("y") and len(v) > 2 and v[-2] not in "aeiou":
        return v[:-1] + "ies"
    return v + "s"


def _past(v: str) -> str:
    if v in IRREGULAR_PAST:
        return IRREGULAR_PAST[v]
    return v + "d" if v.endswith("e") else v + "ed"


def question_to_claim(question: str) -> str | None:
    """'Does creatine improve memory?' -> 'Creatine improves memory.'  None if not a yes/no question."""
    q = question.strip().rstrip("?").strip()
    toks = q.split()
    if len(toks) < 3:
        return None
    aux = toks[0].lower()
    rest = toks[1:]
    if aux in AUX_DO:
        vi = next((i for i, t in enumerate(rest) if i > 0 and t.lower() in COMMON_VERBS), None)
        if vi is None:
            return None
        verb = rest[vi].lower()
        verb = _third_person(verb) if aux == "does" else _past(verb) if aux == "did" else verb
        out = rest[:vi] + [verb] + rest[vi + 1:]
    elif aux in AUX_BE:
        ents = entities(" ".join(rest))
        if ents and " ".join(rest).startswith(ents[0]):
            n = len(ents[0].split())
        else:
            # subject = words up to the first adjective/participle-looking word
            n = next((i for i, t in enumerate(rest) if i > 0 and (t.lower().endswith(("ed", "ive", "ful", "ous", "able"))
                                                                  or t.lower() in {"safe", "true", "real", "better",
                                                                                   "worse", "a", "an", "the", "linked",
                                                                                   "associated", "used", "more",
                                                                                   "less", "good", "bad"})), 1)
        out = rest[:n] + [aux] + rest[n:]
    else:
        return None
    text = " ".join(out)
    return text[:1].upper() + text[1:] + "."


EXPECTED_TYPE = [(r"\bwhich country\b|\bwhat country\b", {"country"}), (r"\bwhich city\b|\bwhat city\b", {"city"}),
                 (r"\bwho(?:m|se)?\b", {"person"}), (r"\bwhere\b", {"city", "country", "state"})]
DEMONYM = re.compile(r"^[A-Z][a-z]+(?:ish|ian|ese|ean|ic|i)$")


def _short(question: str, sentence: str) -> str:
    """A short answer for a wh-question: the named thing in the sentence that
    the question doesn't already mention, preferring the type the question asks
    for (who -> person, which country -> country). Nationality words
    ("Scottish") are not answers."""
    qwords = set(words(question))
    want = next((t for rx, t in EXPECTED_TYPE if re.search(rx, question, re.I)), set())
    graph = get_graph()
    best, best_score = None, -9.0
    for i, e in enumerate(entities(sentence)):
        if set(words(e)) & qwords:
            continue
        etype = graph.types.get(graph.canonical(e))
        score = -0.1 * i + (2.0 if etype in want else 0.0) - (2.0 if DEMONYM.match(e) else 0.0) \
            + (0.5 if "person" in want and len(e.split()) >= 2 else 0.0)
        if score > best_score:
            best, best_score = e, score
    if best is not None:
        return best
    nums = re.findall(r"\d[\d,.]*\s*(?:%|metres|kilometres|km|m|years?)?", sentence)
    return nums[0].strip() if nums else sentence


def solve(question: str, *, use_llms: bool = True, k: int = 5) -> list[CandidateAnswer]:
    cands: list[CandidateAnswer] = []
    claim = question_to_claim(question) if is_question(question) else question
    if claim:
        cands.append(CandidateAnswer(model="claim", role="solver", answer=claim, final_value=claim,
                                     reasoning="yes/no question rewritten as the claim it asks about"))
        return cands                       # a yes/no question is answered by verifying the claim
    graph = get_graph()
    for path in graph.multi_hop(question, k=1):
        if path.score >= 0.3:
            cands.append(CandidateAnswer(model="graph", answer=path.text, final_value=path.answer,
                                         reasoning=f"knowledge-graph path, score {path.score:.2f}"))
    store = get_store()
    for model, method in (("offline-swift", "bm25"), ("offline-scholar", "vector")):
        hits = store.search(question, k=1, method=method)
        if hits:
            cands.append(CandidateAnswer(model=model, answer=hits[0].text, final_value=_short(question, hits[0].text),
                                         reasoning=f"best {method} sentence from {hits[0].source_id}"))
    hits = store.search(question, k=k)
    if hits:
        best = max(hits[:k], key=lambda p: p.score * (store.sources[p.source_id].quality.overall + 0.2))
        cands.append(CandidateAnswer(model="offline-sage", answer=best.text, final_value=_short(question, best.text),
                                     reasoning=f"hybrid search, quality-weighted, from {best.source_id}"))
    if use_llms:
        cands += _llm_candidates(question)
    return cands


LLM_SYSTEM = ("Answer the question in one sentence, then give the bare answer. Do not hedge unless the "
              "evidence is genuinely unsettled. Return JSON: {\"answer\": \"...\", \"short\": \"...\"}")


def _llm_candidates(question: str) -> list[CandidateAnswer]:
    reg = get_registry()
    out = []
    for spec in reg.for_role("solver", llm_only=True):
        try:
            data, resp = reg.provider_for(spec).complete_json(spec, LLM_SYSTEM, question)
        except Exception:
            continue
        if isinstance(data, dict) and data.get("answer"):
            out.append(CandidateAnswer(model=spec.name, answer=data["answer"], final_value=data.get("short"),
                                       tokens_in=resp.tokens_in, tokens_out=resp.tokens_out, cost_usd=resp.cost_usd,
                                       latency_s=resp.latency_s))
    return out


# --------------------------------------------------------------------------- diversity
def _norm(v: str | None) -> set[str]:
    return {w for w in words(v or "") if w not in STOPWORDS}


def _family(model: str) -> str:
    reg = get_registry()
    return reg.models[model].family if model in reg.models else model


def diversity(cands: list[CandidateAnswer], same_threshold: float = 0.5) -> DiversityReport:
    """Cluster candidates by their short answers and measure how independent the
    agreeing ones are. Offline solvers share one corpus, so their agreement is
    weaker evidence than agreement between different model families."""
    clusters: list[list[CandidateAnswer]] = []
    for c in cands:
        for cl in clusters:
            if jaccard(_norm(c.final_value), _norm(cl[0].final_value)) >= same_threshold:
                cl.append(c)
                break
        else:
            clusters.append([c])
    clusters.sort(key=len, reverse=True)
    n = len(cands)
    pair = {}
    for i, a in enumerate(cands):
        for b in cands[i + 1:]:
            pair[f"{a.model}|{b.model}"] = round(jaccard(_norm(a.final_value), _norm(b.final_value)), 3)
    major = clusters[0] if clusters else []
    fams = {_family(c.model) for c in major}
    offline_only = all(c.model.startswith(("offline", "graph", "claim")) for c in major)
    risk = 0.0 if len(major) < 2 else round(1 - (len(fams) - 1) / (len(major) - 1), 3) * 0.5 + (0.3 if offline_only else 0)
    notes = []
    if offline_only and len(major) > 1:
        notes.append("majority comes only from offline solvers reading the same corpus: their errors correlate")
    return DiversityReport(
        n_candidates=n, agreement_rate=round(len(major) / n, 3) if n else 0.0,
        answer_clusters=[[c.id for c in cl] for cl in clusters], pairwise_similarity=pair,
        majority_value=major[0].final_value if major else None, majority_share=round(len(major) / n, 3) if n else 0.0,
        correlated_error_risk=round(min(1.0, risk), 3), notes=notes)
