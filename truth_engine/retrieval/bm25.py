"""Okapi BM25 keyword retrieval over stemmed content tokens."""
from __future__ import annotations

import math
from collections import Counter

from ..textutils import content_tokens


class BM25:
    def __init__(self, k1: float = 1.4, b: float = 0.7):
        self.k1, self.b = k1, b
        self.docs: list[Counter] = []
        self.lens: list[int] = []
        self.df: Counter = Counter()
        self.avgdl = 0.0

    def fit(self, texts: list[str]) -> "BM25":
        self.docs, self.lens, self.df = [], [], Counter()
        for t in texts:
            c = Counter(content_tokens(t))
            self.docs.append(c)
            self.lens.append(sum(c.values()))
            self.df.update(c.keys())
        self.avgdl = sum(self.lens) / max(len(self.lens), 1)
        return self

    def idf(self, term: str) -> float:
        n = len(self.docs)
        df = self.df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> list[float]:
        q = content_tokens(query)
        out = []
        for doc, dl in zip(self.docs, self.lens):
            s = 0.0
            for t in q:
                f = doc.get(t, 0)
                if not f:
                    continue
                s += self.idf(t) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
            out.append(s)
        return out
