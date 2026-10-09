"""Orchestrator (spec §7, §12, §14): one entry point for both modes.

    audit(text)      claims -> verify (parallel when live) -> evidence tree -> trace
    ask(question)    yes/no: the question's claim, verified -> yes / no / can't say
                     open:   solvers propose answers -> each answer is turned back
                             into a statement and verified -> the judge picks the
                             best-verified one, or abstains
    recheck_watched  re-verify saved claims, report verdicts that moved

Every run emits StageEvents (for a live UI log), and is saved as a Trace
(spec §14 record + full JSON) in the SQLite store, where feedback attaches.
"""
from __future__ import annotations

import os
import re
import time

from ..config import settings
from ..feedback.traces import TraceStore
from ..retrieval import EvidenceEngine
from ..schemas import (ConfidenceBreakdown, Decision, EvidenceStatus, FinalAnswer, StageEvent, Trace)
from .analyzer import analyze
from .audit import ALL_COMPONENTS, audit, audit_claims
from .claims import make_claim
from .judge import judge_candidates
from .solvers import answer_type_ok, diversity, question_to_claim, solve
from .tree import build_tree

YES = {EvidenceStatus.SUPPORTED, EvidenceStatus.PARTIALLY_SUPPORTED, EvidenceStatus.INFERRED}


def to_statement(question: str, answer: str) -> str:
    """Put a short answer back into its question so it can be verified as a claim:
    'Who discovered penicillin?' + 'Alexander Fleming' -> 'Alexander Fleming discovered penicillin.'"""
    q = question.strip().rstrip("?").strip()
    patterns = [
        (r"^who (?:was|is) (.+)$", lambda m: f"{answer} was {m[1]}"),
        (r"^who (.+)$", lambda m: f"{answer} {m[1]}"),
        (r"^what (?:is|was|are|were) (the .+)$", lambda m: f"{m[1][0].upper() + m[1][1:]} is {answer}"),
        (r"^(?:in )?(?:which|what) (?:country|city|state|place|year) (?:was|is|did) (.+?) (born|located|founded|"
         r"built|discovered|released|launched|die|died)$", lambda m: f"{m[1]} {'was ' if m[2] != 'died' else ''}{m[2]} in {answer}"),
        (r"^where (?:was|is) (.+?) (born|located|founded|built)$", lambda m: f"{m[1]} was {m[2]} in {answer}"),
        (r"^when (?:was|did) (.+?) (born|founded|built|released|launched|discovered|die)$",
         lambda m: f"{m[1]} was {m[2]} in {answer}"),
        (r"^how (tall|long|high|old|big|large) (?:is|was) (.+)$", lambda m: f"{m[2]} is {answer} {m[1]}"),
    ]
    for rx, fmt in patterns:
        m = re.match(rx, q, re.I)
        if m:
            s = fmt(m)
            return s[0].upper() + s[1:] + "."
    return f"{q}: {answer}."


class Orchestrator:
    def __init__(self, *, live: bool = False, max_tier=None, components=ALL_COMPONENTS,
                 store: TraceStore | None | bool = None, engine: EvidenceEngine | None = None,
                 workers: int | None = None, llm_verifier=None, llm_extractor=None, calibrator=None):
        self.live, self.max_tier, self.components = live, max_tier, components
        self.engine = engine or EvidenceEngine()
        self.store = TraceStore() if store is None else (store or None)
        self.workers = workers or (settings.max_parallel if live else 1)
        self.llm_verifier, self.llm_extractor, self.calibrator = llm_verifier, llm_extractor, calibrator

    # ------------------------------------------------------------ audit
    def audit(self, text: str, on_event=None) -> Trace:
        trace = Trace(question=text, mode="audit")
        emit = self._emitter(trace, on_event)
        emit(StageEvent(stage="audit", status="started", message="audit started"))
        rep = audit(text, live=self.live, max_tier=self.max_tier, engine=self.engine, components=self.components,
                    llm_extractor=self.llm_extractor, llm_verifier=self.llm_verifier, calibrator=self.calibrator,
                    workers=self.workers, emit=emit)
        self._fill(trace, rep)
        trace.tree = build_tree(text, rep.claims, rep.verdicts)
        counts = {}
        for v in rep.verdicts:
            counts[v.status.value] = counts.get(v.status.value, 0) + 1
        emit(StageEvent(stage="audit", status="completed", message=f"done: {counts}", data=counts))
        return self._save(trace)

    # ------------------------------------------------------------ ask
    def ask(self, question: str, on_event=None) -> Trace:
        trace = Trace(question=question, mode="ask", analysis=analyze(question))
        emit = self._emitter(trace, on_event)
        claim_text = question_to_claim(question)
        if claim_text:
            emit(StageEvent(stage="solve", status="completed", message=f"yes/no question -> claim: {claim_text}"))
            rep = self._verify([claim_text], emit)
            v = rep.verdicts[0]
            self._fill(trace, rep)
            trace.final = self._yes_no(v)
        else:
            cands = solve(question, use_llms=self.llm_verifier is not None)
            trace.candidates = cands
            trace.diversity = diversity(cands) if cands else None
            emit(StageEvent(stage="solve", status="completed",
                            message=f"{len(cands)} candidate(s): " + ", ".join(f"{c.model}={c.final_value}" for c in cands)))
            values = list(dict.fromkeys(c.final_value or c.answer for c in cands))
            wrong_type = [v for v in values if not answer_type_ok(question, v)]
            if wrong_type:
                emit(StageEvent(stage="judge", status="completed",
                                message=f"rejected (wrong kind of answer for the question): {wrong_type}"))
            statements = {val: to_statement(question, val) for val in values if val not in wrong_type}
            rep = self._verify(list(statements.values()), emit) if statements else None
            if rep:
                self._fill(trace, rep)
            by_statement = {c.sentence: v for c, v in zip(rep.claims, rep.verdicts)} if rep else {}
            verified = {c.id: by_statement.get(statements.get(c.final_value or c.answer)) for c in cands}
            verified = {k: v for k, v in verified.items() if v is not None}
            trace.judge = judge_candidates(cands, verified, trace.diversity,
                                           threshold=self._threshold())
            trace.final = self._open_answer(trace, cands, verified)
        trace.tree = build_tree(question, trace.claims, trace.verdicts)
        emit(StageEvent(stage="answer", status="completed", message=trace.final.summary))
        return self._save(trace)

    # ------------------------------------------------------------ watched claims
    def watch(self, text: str) -> str:
        return self.store.watch(text)

    def recheck_watched(self, on_event=None) -> list[dict]:
        """Re-verify every watched claim; return the ones whose verdict changed."""
        changes = []
        for w in self.store.watched():
            trace = Trace(question=w["text"], mode="audit")
            rep = self._verify([w["text"]], self._emitter(trace, on_event))
            self._fill(trace, rep)
            trace.tree = build_tree(w["text"], rep.claims, rep.verdicts)
            self._save(trace)
            v = rep.verdicts[0]
            change = self.store.record_check(w["text"], v.status.value, v.confidence, trace.query_id)
            if change:
                changes.append(change)
        return changes

    # ------------------------------------------------------------ helpers
    def _verify(self, statements: list[str], emit):
        claims = [make_claim(s.rstrip("."), sentence=s) for s in statements]
        return audit_claims(claims, live=self.live, max_tier=self.max_tier, engine=self.engine,
                            components=self.components, llm_verifier=self.llm_verifier, calibrator=self.calibrator,
                            workers=self.workers, emit=emit)

    def _threshold(self) -> float | None:
        from .confidence import Calibrator
        cal = self.calibrator or Calibrator.load()
        return cal.threshold if cal else None

    @staticmethod
    def _emitter(trace: Trace, on_event):
        def emit(ev: StageEvent):
            trace.events.append(ev)
            if on_event:
                on_event(ev)
        return emit

    @staticmethod
    def _fill(trace: Trace, rep) -> None:
        trace.claims, trace.verdicts, trace.sources = rep.claims, rep.verdicts, rep.sources
        trace.routing, trace.skipped, trace.verifiers = rep.routing, rep.skipped, rep.verifiers
        trace.errors, trace.latency_s = rep.errors, rep.latency_s

    def _save(self, trace: Trace) -> Trace:
        trace.latency_s = round(time.time() - trace.created_at, 3)
        if self.store:
            self.store.save_trace(trace)
        return trace

    @staticmethod
    def _citations(verdicts) -> dict[str, list[str]]:
        return {v.claim_id: list(dict.fromkeys(i.source.id for i in v.supporting + v.contradicting))[:5]
                for v in verdicts}

    def _yes_no(self, v) -> FinalAnswer:
        conf = v.confidence_detail or _empty_conf()
        if v.decision == Decision.ABSTAIN:
            lean = f" The evidence leans {v.leaning.value.replace('_', ' ')}." if v.leaning else ""
            summary, decision = "The evidence isn't good enough to say." + lean, Decision.ABSTAIN
        elif v.status in YES:
            summary, decision = "Yes" + (", partly." if v.status != EvidenceStatus.SUPPORTED else "."), v.decision
        else:
            summary, decision = "No.", v.decision
        return FinalAnswer(decision=decision, summary=summary, body=v.rationale, confidence=conf,
                           caveats=[f.message for f in v.red_flags],
                           uncertainty=(["Strong evidence points both ways."] if v.conflicting else []),
                           citations=self._citations([v]))

    def _open_answer(self, trace: Trace, cands, verified) -> FinalAnswer:
        j = trace.judge
        if j.decision == Decision.ABSTAIN or j.selected_candidate_id is None:
            best = max(verified.values(), key=lambda v: v.confidence, default=None)
            return FinalAnswer(decision=Decision.ABSTAIN, summary="The evidence isn't good enough to say.",
                               body=j.rationale, confidence=(best.confidence_detail if best and best.confidence_detail
                                                             else _empty_conf()),
                               uncertainty=[f"{r['model']}: {r['reason']}" for r in j.rejected],
                               citations=self._citations(trace.verdicts))
        c = next(c for c in cands if c.id == j.selected_candidate_id)
        v = verified[c.id]
        return FinalAnswer(decision=j.decision, summary=str(c.final_value or c.answer), body=v.rationale,
                           confidence=v.confidence_detail or _empty_conf(), caveats=[f.message for f in v.red_flags],
                           citations=self._citations([v]))


def _empty_conf() -> ConfidenceBreakdown:
    zero = dict.fromkeys(["model_agreement", "evidence_support", "source_quality", "verifier_agreement",
                          "historical_reliability", "source_independence", "contradiction", "unresolved_uncertainty",
                          "difficulty_penalty"], 0.0)
    return ConfidenceBreakdown(**zero, raw_score=0.0, calibrated=0.0, label="VERY LOW")
