"""Computed, calibrated confidence (spec §4, Improvement 9; CONCEPTS.md §8).

Confidence is never asked of a model. It answers one question: *how likely
is this verdict to be right?* It's built in three steps:

  1. signals - the spec's list, each in [0, 1]:
       evidence_support     strength of the winning side (S or C; for UNVERIFIED, 1 - max)
       contradiction        strength of the losing side
       source_quality       best quality on the winning side
       source_independence  1 - 0.5^(independent lineages on the winning side)
       verifier_agreement   rules vs NLI/LLM agreement on this claim's evidence
       model_agreement      the NLI model's own probability for the winning label
       historical_reliability  how often verdicts of this status were right before
       unresolved_uncertainty  critic findings + red flags
       difficulty_penalty   analyzer difficulty
  2. raw score - a fixed, hand-weighted logistic combination (works with no data)
  3. calibration - fitted on stored outcomes (SciFact train): logistic
     regression on the signals, or Platt scaling of the raw score, whichever
     cross-validates better. Measured with expected calibration error (ECE),
     Brier score and a reliability table: "80%" should be right ~80% of the time.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..config import DATA_DIR
from ..schemas import ClaimVerdict, ConfidenceBreakdown, CritiqueReport, EvidenceStatus, QuestionAnalysis

STATUSES = [s.value for s in EvidenceStatus]
SIGNALS = ["evidence_support", "contradiction", "source_quality", "source_independence", "verifier_agreement",
           "model_agreement", "historical_reliability", "unresolved_uncertainty", "difficulty_penalty"]
# hand-set weights for the uncalibrated score (sign = direction of effect)
RAW_WEIGHTS = {"evidence_support": 2.0, "contradiction": -1.5, "source_quality": 1.0, "source_independence": 0.8,
               "verifier_agreement": 0.8, "model_agreement": 1.2, "historical_reliability": 1.0,
               "unresolved_uncertainty": -1.2, "difficulty_penalty": -0.8}
RAW_BIAS = -1.5
CALIBRATOR_PATH = DATA_DIR / "calibrator.json"


def _sig(z):
    return 1 / (1 + np.exp(-z))


def label(p: float) -> str:
    return "VERY LOW" if p < 0.3 else "LOW" if p < 0.5 else "MODERATE" if p < 0.7 else "HIGH" if p < 0.85 \
        else "VERY HIGH"


def signals(verdict: ClaimVerdict, critique: CritiqueReport | None, analysis: QuestionAnalysis | None,
            history: dict[str, float] | None = None) -> dict[str, float]:
    st = verdict.status
    s, c = verdict.support_score, verdict.contradiction_score
    if st == EvidenceStatus.CONTRADICTED:
        win_items, win, lose, nli_key = verdict.contradicting, c, s, "contradiction"
    elif st == EvidenceStatus.UNVERIFIED:
        win_items, win, lose, nli_key = [], 1 - max(s, c), min(s, c), "neutral"
    else:
        win_items, win, lose, nli_key = verdict.supporting, s, c, "entailment"
    lineages = len({i.lineage for i in win_items if i.lineage is not None})
    pool = win_items if win_items else verdict.supporting + verdict.contradicting
    nli_probs = [i.nli[nli_key] for i in pool if i.nli]
    # critic findings when the critic ran, else the red flags directly. Red flags all
    # describe weaknesses of the claim's *support* (echo chamber, tiny or industry-funded
    # pilot, mouse-only, retracted...): they make a CONTRADICTED verdict more certain,
    # so they don't count as unresolved uncertainty there.
    if st == EvidenceStatus.CONTRADICTED:
        severity = 0.0
    elif critique:
        severity = sum(f.severity for f in critique.findings)
    else:
        severity = sum(f.severity for f in verdict.red_flags)
    return {
        "evidence_support": round(win, 4),
        "contradiction": round(lose, 4),
        "source_quality": max((i.source.quality.overall for i in win_items if i.source.quality), default=0.0),
        "source_independence": round(1 - 0.5 ** lineages, 4),
        "verifier_agreement": verdict.verifier_agreement if verdict.verifier_agreement is not None else 0.5,
        "model_agreement": round(max(nli_probs), 4) if nli_probs else 0.5,
        "historical_reliability": (history or {}).get(st.value, 0.5),
        "unresolved_uncertainty": round(min(1.0, severity / 2), 4),
        "difficulty_penalty": analysis.difficulty if analysis else 0.5,
    }


def raw_score(sig: dict[str, float]) -> float:
    return float(_sig(RAW_BIAS + sum(RAW_WEIGHTS[k] * sig[k] for k in SIGNALS)))


def feature_vector(sig: dict[str, float], status: str) -> np.ndarray:
    return np.array([1.0] + [sig[k] for k in SIGNALS] + [float(status == s) for s in STATUSES])


# --------------------------------------------------------------------------- calibrators
def _fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 500) -> np.ndarray:
    """L2-regularized logistic regression by Newton's method (no bias penalty)."""
    w = np.zeros(X.shape[1])
    reg = np.eye(X.shape[1]) * l2
    reg[0, 0] = 0.0
    for _ in range(iters):
        p = _sig(X @ w)
        grad = X.T @ (p - y) + reg @ w
        H = X.T @ (X * (p * (1 - p))[:, None]) + reg + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(H, grad)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


class Calibrator:
    """method = "logistic" (signals + status) or "platt" (raw score only)."""

    def __init__(self, method: str = "logistic", weights: list[float] | None = None,
                 history: dict[str, float] | None = None, threshold: float | None = None, meta: dict | None = None):
        self.method, self.w = method, np.array(weights) if weights is not None else None
        self.history = history or {}
        self.threshold = threshold          # abstention threshold chosen by the judge's fit
        self.meta = meta or {}

    def _x(self, sig, status) -> np.ndarray:
        if self.method == "platt":
            r = raw_score(sig)
            return np.array([1.0, math.log(r / (1 - r + 1e-9) + 1e-9)])
        return feature_vector(sig, status)

    def fit(self, sigs: list[dict], statuses: list[str], correct: list[bool], l2: float = 1.0) -> "Calibrator":
        X = np.array([self._x(s, st) for s, st in zip(sigs, statuses)])
        self.w = _fit_logistic(X, np.array(correct, dtype=float), l2)
        return self

    def __call__(self, sig: dict[str, float], status: str) -> float:
        if self.w is None:
            return raw_score(sig)
        return float(_sig(self._x(sig, status) @ self.w))

    def save(self, path: Path = CALIBRATOR_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"method": self.method, "weights": self.w.tolist() if self.w is not None else None,
                                    "history": self.history, "threshold": self.threshold, "meta": self.meta},
                                   indent=1))

    @classmethod
    def load(cls, path: Path = CALIBRATOR_PATH) -> "Calibrator | None":
        """The fitted calibrator, or None (raw scores) if absent, or if TE_CALIBRATOR=0
        for the default file."""
        import os
        disabled = os.getenv("TE_CALIBRATOR", "1").lower() in ("0", "off", "false") and path == CALIBRATOR_PATH
        if disabled or not path.exists():
            return None
        d = json.loads(path.read_text())
        return cls(d["method"], d["weights"], d.get("history"), d.get("threshold"), d.get("meta"))


def history_from(statuses: list[str], correct: list[bool]) -> dict[str, float]:
    """Per-status accuracy with a Laplace prior: the 'historical reliability' signal."""
    out = {}
    for s in set(statuses):
        hits = [c for st, c in zip(statuses, correct) if st == s]
        out[s] = round((sum(hits) + 1) / (len(hits) + 2), 4)
    return out


def cross_val_brier(method: str, sigs, statuses, correct, folds: int = 5, l2: float = 1.0) -> float:
    n = len(correct)
    idx = np.arange(n)
    losses = []
    for k in range(folds):
        test = idx[k::folds]
        train = np.setdiff1d(idx, test)
        cal = Calibrator(method).fit([sigs[i] for i in train], [statuses[i] for i in train],
                                     [correct[i] for i in train], l2)
        losses += [(cal(sigs[i], statuses[i]) - correct[i]) ** 2 for i in test]
    return float(np.mean(losses))


# --------------------------------------------------------------------------- metrics
def reliability(probs: list[float], correct: list[bool], bins: int = 10) -> list[dict]:
    rows = []
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [(p, c) for p, c in zip(probs, correct) if lo <= p < hi or (hi == 1 and p == 1)]
        if sel:
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": len(sel), "mean_confidence": round(np.mean([p for p, _ in sel]), 3),
                         "accuracy": round(np.mean([c for _, c in sel]), 3)})
    return rows


def ece(probs: list[float], correct: list[bool], bins: int = 10) -> float:
    n = len(probs)
    return round(sum(r["n"] / n * abs(r["mean_confidence"] - r["accuracy"]) for r in reliability(probs, correct, bins)), 4)


def brier(probs: list[float], correct: list[bool]) -> float:
    return round(float(np.mean([(p - c) ** 2 for p, c in zip(probs, correct)])), 4)


# --------------------------------------------------------------------------- assemble
def breakdown(verdict: ClaimVerdict, critique: CritiqueReport | None, analysis: QuestionAnalysis | None,
              calibrator: Calibrator | None) -> ConfidenceBreakdown:
    sig = signals(verdict, critique, analysis, calibrator.history if calibrator else None)
    raw = raw_score(sig)
    cal = calibrator(sig, verdict.status.value) if calibrator else raw
    return ConfidenceBreakdown(**sig, raw_score=round(raw, 4), calibrated=round(cal, 4), label=label(cal))

