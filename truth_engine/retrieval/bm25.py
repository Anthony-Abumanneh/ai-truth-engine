"""Okapi BM25 keyword retrieval over stemmed content tokens.

Uses an inverted index (term -> documents containing it), so scoring a query
touches only the documents that share a word with it: fast enough for the
~50k-sentence SciFact corpus."""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from ..textutils import content_tokens


class BM25:
    def __init__(self, k1: float = 1.4, b: float = 0.7):
        self.k1, self.b = k1, b
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.lens = np.zeros(0)
        self.df: Counter = Counter()
        self.avgdl = 0.0
        self.n = 0

    def fit(self, texts: list[str]) -> "BM25":
        post: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lens = []
        for i, t in enumerate(texts):
            c = Counter(content_tokens(t))
            lens.append(sum(c.values()))
            for term, f in c.items():
                post[term].append((i, f))
        self.postings = {t: (np.array([d for d, _ in p], dtype=np.int64), np.array([f for _, f in p], dtype=float))
                         for t, p in post.items()}
        self.df = Counter({t: len(p[0]) for t, p in self.postings.items()})
        self.lens = np.array(lens, dtype=float)
        self.n = len(texts)
        self.avgdl = float(self.lens.mean()) if self.n else 0.0
        return self

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n)
        norm = self.k1 * (1 - self.b + self.b * self.lens / (self.avgdl or 1))
        for t in content_tokens(query):
            if t not in self.postings:
                continue
            idx, f = self.postings[t]
            out[idx] += self.idf(t) * f * (self.k1 + 1) / (f + norm[idx])
        return out
