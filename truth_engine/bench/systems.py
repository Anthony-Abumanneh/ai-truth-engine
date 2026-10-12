"""Systems under test (spec §15): the full engine, its ablations, and baselines.

    ours       the full pipeline (local NLI + rules + routing + calibration + judge)
    A single   one LLM, from its own knowledge
    B rag      one LLM + standard RAG (top-5 retrieved sentences, one query, no
               falsification search, no source grading, no independence)
    C vote     several LLMs from their own knowledge, majority vote
    D judge    several LLMs with the same RAG context, then a judge LLM decides

Baselines report the confidence the LLM *states* (the spec's point: asked-for
confidence vs computed confidence). Every LLM response is cached on disk
(data/cache/llm), so reruns cost nothing and results are reproducible; free
tiers are rate-limited, so 429s are retried with a long back-off.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter

from ..providers import ProviderError, extract_json, get_registry
from ..schemas import Decision, EvidenceStatus
from ..science.cache import JsonCache
from ..pipeline.audit import DEFAULT_COMPONENTS, _Run, check_claim
from ..pipeline.claims import make_claim
from ..pipeline.router import Router

LABEL = {EvidenceStatus.SUPPORTED: "SUPPORT", EvidenceStatus.PARTIALLY_SUPPORTED: "SUPPORT",
         EvidenceStatus.INFERRED: "SUPPORT", EvidenceStatus.CONTRADICTED: "CONTRADICT",
         EvidenceStatus.UNVERIFIED: "NEI", EvidenceStatus.SPECULATIVE: "NEI"}


# --------------------------------------------------------------------------- ours
def run_ours(cases: list[dict], engine, components=DEFAULT_COMPONENTS, calibrator=None, live: bool = False,
             progress: str | None = None) -> list[dict]:
    from ..pipeline.confidence import Calibrator
    calibrator = calibrator if calibrator is not None else Calibrator.load()   # same default as audit()
    run = _Run(engine, None, components)
    router = Router(live=live, has_web=live and engine.web.available,
                    has_nli=run.verifier(True, False).nli is not None)
    out, t0 = [], time.time()
    for n, c in enumerate(cases, 1):
        claim = make_claim(c["claim"].rstrip("."), sentence=c["claim"])
        t = time.time()
        v, route = check_claim(claim, run, router, components=components, calibrator=calibrator)
        pre = v.leaning or v.status
        out.append({
            "id": c["id"], "claim": c["claim"], "gold": c["gold"], "gold_docs": c.get("gold_docs", []),
            "pred": LABEL[v.status], "answered": v.decision not in (Decision.ABSTAIN, None),
            "conf_pred": LABEL[pre], "confidence": v.confidence_detail.calibrated if v.confidence_detail else None,
            "cited": list(dict.fromkeys(i.source.id for i in v.supporting + v.contradicting)),
            "red_flags": sorted({f.kind for f in v.red_flags}), "expected_flags": c.get("expected_flags"),
            "tiers": [p.get("tier") for p in route["passes"]], "latency_s": round(time.time() - t, 3),
            "cost_usd": 0.0, "tokens": 0})
        if progress and (n % 50 == 0 or n == len(cases)):
            acc = sum(r["gold"] == r["pred"] for r in out) / len(out)
            print(f"  [{progress}] {n}/{len(cases)} accuracy {acc:.3f} ({time.time() - t0:.0f}s)", flush=True)
    return out


# --------------------------------------------------------------------------- LLM plumbing
def _norm_label(x) -> str:
    s = str(x or "").upper()
    if s.startswith("SUPPORT"):
        return "SUPPORT"
    if s.startswith(("CONTRADICT", "REFUTE")):
        return "CONTRADICT"
    return "NEI"


class LLM:
    """One model with on-disk response cache and patient retries for free-tier rate limits."""

    def __init__(self, spec, provider, cache: JsonCache | None = None, min_interval_s: float = 2.0):
        self.spec, self.provider = spec, provider
        self.cache = cache or JsonCache()
        self.min_interval_s = min_interval_s
        self._last = 0.0
        self.exhausted: str | None = None      # set when a daily quota is hit; further calls are skipped

    @property
    def name(self) -> str:
        return self.spec.name

    def ask(self, system: str, user: str, max_tokens: int = 400) -> tuple[dict | None, dict]:
        key_params = {"system": system, "user": user, "model": self.spec.api_model}
        hit = self.cache.get("llm", self.spec.name, key_params)
        if hit is not None:
            d = json.loads(hit)
            return d["data"], {**d["meta"], "cached": True}
        if self.exhausted:
            return None, {"tokens": 0, "cost_usd": 0.0, "latency_s": 0.0, "error": self.exhausted}
        for attempt in range(6):
            wait = self._last + self.min_interval_s - time.time()
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                resp = self.provider.complete(self.spec, system + "\n\nRespond with a single JSON object.", user,
                                              temperature=0.0, max_tokens=max_tokens)
                data = extract_json(resp.text)
                meta = {"tokens": resp.tokens_in + resp.tokens_out, "cost_usd": resp.cost_usd,
                        "latency_s": round(resp.latency_s, 3)}
                self.cache.put("llm", self.spec.name, key_params, json.dumps({"data": data, "meta": meta}))
                return data if isinstance(data, dict) else None, meta
            except ProviderError as e:
                msg = str(e)
                if getattr(e, "status", None) == 429 and re.search(r"retry in \d+h|per day|PerDay|daily", msg):
                    self.exhausted = f"daily quota exhausted for {self.spec.name}"
                    return None, {"tokens": 0, "cost_usd": 0.0, "latency_s": 0.0, "error": self.exhausted}
                if getattr(e, "status", None) in (429, 500, 502, 503, 504, 529) and attempt < 5:
                    time.sleep(15 * (attempt + 1))         # free-tier per-minute limits
                    continue
                return None, {"tokens": 0, "cost_usd": 0.0, "latency_s": 0.0, "error": str(e)[:200]}
        return None, {"tokens": 0, "cost_usd": 0.0, "latency_s": 0.0, "error": "retries exhausted"}


# Seconds between requests per provider, to stay inside free-tier per-minute limits.
PACE = {"gemini": 6.0, "groq": 2.5, "openrouter": 4.0}


def available_llms(min_interval_s: float | None = None) -> list[LLM]:
    reg = get_registry()
    specs = sorted(reg.llms, key=lambda m: m.provider == "gemini")     # Gemini last: smallest free quota
    return [LLM(spec, reg.provider_for(spec), min_interval_s=min_interval_s or PACE.get(spec.provider, 1.0))
            for spec in specs]


VERDICT_SYSTEM = ("You check scientific and health claims. Label the claim SUPPORT, CONTRADICT or NOT_ENOUGH_INFO. "
                  "Return JSON: {\"label\": \"SUPPORT|CONTRADICT|NOT_ENOUGH_INFO\", \"confidence\": <0-1, how sure "
                  "you are that your label is right>, \"reason\": \"<one sentence>\", \"cited\": [<ids of the "
                  "evidence you relied on, if evidence was given>]}")


def _rag_context(engine, claim: str, k: int = 5) -> tuple[str, list[str]]:
    """Standard RAG: one query (the claim), top-k sentences, no falsification search."""
    hits = engine.store.search(claim, k=k)
    lines, ids = [], []
    for p in hits:
        src = engine.store.sources[p.source_id]
        lines.append(f"[{p.source_id}] ({src.title}) {p.text}")
        ids.append(p.source_id)
    return "\n".join(lines), list(dict.fromkeys(ids))


def _record(case, label, conf, cited, meta_list, t0) -> dict:
    return {"id": case["id"], "claim": case["claim"], "gold": case["gold"], "gold_docs": case.get("gold_docs", []),
            "pred": label, "answered": label != "NEI", "conf_pred": label,
            "confidence": None if conf is None else min(1.0, max(0.0, float(conf))),
            "cited": cited, "latency_s": round(time.time() - t0, 3),
            "cost_usd": sum(m.get("cost_usd", 0.0) for m in meta_list),
            "tokens": sum(m.get("tokens", 0) for m in meta_list),
            "errors": [m["error"] for m in meta_list if m.get("error")]}


def _conf(d) -> float | None:
    try:
        return float((d or {}).get("confidence"))
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- baselines
def run_single(cases, llm: LLM) -> list[dict]:
    out = []
    for c in cases:
        t0 = time.time()
        d, meta = llm.ask(VERDICT_SYSTEM, f"Claim: {c['claim']}")
        out.append(_record(c, _norm_label((d or {}).get("label")), _conf(d), None, [meta], t0))
    return out


def run_rag(cases, llm: LLM, engine) -> list[dict]:
    out = []
    for c in cases:
        t0 = time.time()
        ctx, ids = _rag_context(engine, c["claim"])
        d, meta = llm.ask(VERDICT_SYSTEM, f"Evidence:\n{ctx}\n\nClaim: {c['claim']}")
        cited = [x for x in (d or {}).get("cited", []) if isinstance(x, str) and x in ids] if d else []
        out.append(_record(c, _norm_label((d or {}).get("label")), _conf(d), cited or ids, [meta], t0))
    return out


def run_vote(cases, llms: list[LLM]) -> list[dict]:
    out = []
    for c in cases:
        t0 = time.time()
        votes, metas = [], []
        for llm in llms:
            d, meta = llm.ask(VERDICT_SYSTEM, f"Claim: {c['claim']}")
            votes.append((_norm_label((d or {}).get("label")), _conf(d)))
            metas.append(meta)
        counts = Counter(v for v, _ in votes)
        top, n = counts.most_common(1)[0]
        label = top if n > len(votes) / 2 else "NEI"           # no strict majority -> no verdict
        confs = [cf for v, cf in votes if v == label and cf is not None]
        out.append(_record(c, label, sum(confs) / len(confs) if confs else None, None, metas, t0))
    return out


JUDGE_SYSTEM = ("You are the judge. Several fact-checkers labeled a scientific claim using the same evidence. Weigh "
                "the evidence itself, not the vote count. " + VERDICT_SYSTEM.split(". ", 1)[1])


def run_judge(cases, llms: list[LLM], judge: LLM, engine) -> list[dict]:
    out = []
    for c in cases:
        t0 = time.time()
        ctx, ids = _rag_context(engine, c["claim"])
        answers, metas = [], []
        for llm in llms:
            d, meta = llm.ask(VERDICT_SYSTEM, f"Evidence:\n{ctx}\n\nClaim: {c['claim']}")
            answers.append(f"- {llm.name}: {_norm_label((d or {}).get('label'))} "
                           f"(confidence {_conf(d)}): {(d or {}).get('reason', '')}")
            metas.append(meta)
        d, meta = judge.ask(JUDGE_SYSTEM, f"Evidence:\n{ctx}\n\nClaim: {c['claim']}\n\nFact-checkers:\n"
                                          + "\n".join(answers))
        metas.append(meta)
        cited = [x for x in (d or {}).get("cited", []) if isinstance(x, str) and x in ids] if d else []
        out.append(_record(c, _norm_label((d or {}).get("label")), _conf(d), cited or ids, metas, t0))
    return out


# --------------------------------------------------------------------------- fairness + upgrade variants
def fewshot_examples(n_per_label: int = 2) -> str:
    """Worked examples from SciFact TRAIN (never dev), each with its gold evidence sentences."""
    from . import scifact as sf
    corpus = {}
    for line in open(sf.SCIFACT / "corpus.jsonl"):
        d = json.loads(line)
        corpus[d["doc_id"]] = d
    # never show an abstract that is evidence for a dev claim (it would leak a near-answer)
    dev_docs = {int(d) for line in open(sf.SCIFACT / "claims_dev.jsonl") for d in json.loads(line)["evidence"]} | \
        {int(d) for line in open(sf.SCIFACT / "claims_dev.jsonl") for d in json.loads(line)["cited_doc_ids"]}
    picked, seen = [], Counter()
    for line in open(sf.SCIFACT / "claims_train.jsonl"):
        c = json.loads(line)
        if set(int(d) for d in c["evidence"]) & dev_docs or set(c["cited_doc_ids"]) & dev_docs:
            continue
        if c["evidence"]:
            doc_id, sets = next(iter(c["evidence"].items()))
            label = sets[0]["label"]
            sents = [corpus[int(doc_id)]["abstract"][i] for i in sets[0]["sentences"]]
        else:
            label, sents = "NOT_ENOUGH_INFO", corpus[c["cited_doc_ids"][0]]["abstract"][:1]
        if seen[label] >= n_per_label:
            continue
        seen[label] += 1
        picked.append(f"Evidence: {' '.join(sents)}\nClaim: {c['claim']}\nAnswer: {label}")
        if len(picked) >= 3 * n_per_label:
            break
    return "\n\n".join(picked)


def run_rag_fewshot(cases, llm: LLM, engine) -> list[dict]:
    """Baseline B plus six labelled SciFact train examples in the prompt: the
    same kind of help our system got from being tuned on train."""
    shots = fewshot_examples()
    out = []
    for c in cases:
        t0 = time.time()
        ctx, ids = _rag_context(engine, c["claim"])
        d, meta = llm.ask(VERDICT_SYSTEM + "\n\nWorked examples:\n\n" + shots,
                          f"Evidence:\n{ctx}\n\nClaim: {c['claim']}")
        cited = [x for x in (d or {}).get("cited", []) if isinstance(x, str) and x in ids] if d else []
        out.append(_record(c, _norm_label((d or {}).get("label")), _conf(d), cited or ids, [meta], t0))
    return out


OURS_JUDGE_SYSTEM = (
    "You check scientific claims. An evidence engine has searched the literature for evidence FOR and AGAINST the "
    "claim, graded each source, and flagged distortions. Read the evidence yourself and decide. Label SUPPORT, "
    "CONTRADICT or NOT_ENOUGH_INFO (only if the evidence does not settle the claim). " + VERDICT_SYSTEM.split(". ", 1)[1])


def run_ours_llm_judge(cases, llm: LLM, engine, components=DEFAULT_COMPONENTS) -> list[dict]:
    """Our pipeline finds, grades and de-duplicates evidence (incl. the falsification
    search); an LLM reads that evidence and gives the final label."""
    from ..pipeline.confidence import Calibrator
    run = _Run(engine, None, components)
    router = Router(live=False, has_nli=run.verifier(True, False).nli is not None)
    cal = Calibrator.load()
    out = []
    for c in cases:
        t0 = time.time()
        claim = make_claim(c["claim"].rstrip("."), sentence=c["claim"])
        v, _ = check_claim(claim, run, router, components=components, calibrator=cal)
        items = sorted(v.supporting + v.contradicting, key=lambda i: -i.weight)[:8]
        if len(items) < 3:                       # thin: show the most relevant retrieved sentences too
            ctx, _ids = _rag_context(engine, c["claim"])
        else:
            ctx = ""
        lines = [f"[{i.source.id}] ({i.source.title}; {i.source.metadata.get('study_label') or i.source.source_type}; "
                 f"engine read it as {i.stance.value}) {i.passage.text}" for i in items]
        flags = "; ".join(f"{f.kind}: {f.message}" for f in v.red_flags) or "none"
        user = (f"Evidence found by the engine:\n" + "\n".join(lines) + (f"\n{ctx}" if ctx else "") +
                f"\n\nRed flags: {flags}\nEngine's own verdict: {v.status.value} (confidence {v.confidence:.2f})"
                f"\n\nClaim: {c['claim']}")
        d, meta = llm.ask(OURS_JUDGE_SYSTEM, user)
        ids = [i.source.id for i in items]
        cited = [x for x in (d or {}).get("cited", []) if isinstance(x, str) and x in ids] if d else []
        rec = _record(c, _norm_label((d or {}).get("label")), _conf(d), cited or ids, [meta], t0)
        rec["engine_pred"] = LABEL[v.status]
        out.append(rec)
    return out
