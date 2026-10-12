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

    @property
    def name(self) -> str:
        return self.spec.name

    def ask(self, system: str, user: str, max_tokens: int = 400) -> tuple[dict | None, dict]:
        key_params = {"system": system, "user": user, "model": self.spec.api_model}
        hit = self.cache.get("llm", self.spec.name, key_params)
        if hit is not None:
            d = json.loads(hit)
            return d["data"], {**d["meta"], "cached": True}
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
                if getattr(e, "status", None) in (429, 500, 502, 503, 504, 529) and attempt < 5:
                    time.sleep(15 * (attempt + 1))         # free-tier per-minute limits
                    continue
                return None, {"tokens": 0, "cost_usd": 0.0, "latency_s": 0.0, "error": str(e)[:200]}
        return None, {"tokens": 0, "cost_usd": 0.0, "latency_s": 0.0, "error": "retries exhausted"}


def available_llms(min_interval_s: float = 2.0) -> list[LLM]:
    reg = get_registry()
    return [LLM(spec, reg.provider_for(spec), min_interval_s=min_interval_s) for spec in reg.llms]


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
