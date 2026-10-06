"""Embedding backends.

Order of preference:
  1. OpenAI embeddings      (if OPENAI_API_KEY and TE_EMBEDDINGS=openai)
  2. sentence-transformers   (if installed and TE_EMBEDDINGS=st)
  3. HashingEmbedder         (always available; deterministic, offline)

The hashing embedder is a signed feature-hashing model over stemmed unigrams,
bigrams and character trigrams with sublinear TF weighting. It is not as good
as a neural encoder, but it is fast, reproducible, and captures morphology
(launch/launched) and partial-word overlap, which BM25 alone misses.
"""
from __future__ import annotations

import hashlib
import math
import os
from functools import lru_cache

import numpy as np

from ..textutils import content_tokens, normalize


class Embedder:
    name = "base"
    dim = 0

    def embed(self, texts: list[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


def _h(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")


class HashingEmbedder(Embedder):
    name = "hashing-v1"

    def __init__(self, dim: int = 2048):
        self.dim = dim

    def _features(self, text: str) -> dict[str, float]:
        toks = content_tokens(text)
        feats: dict[str, float] = {}
        for t in toks:
            feats["u:" + t] = feats.get("u:" + t, 0) + 1.0
        for a, b in zip(toks, toks[1:]):
            k = f"b:{a}_{b}"
            feats[k] = feats.get(k, 0) + 0.7
        flat = " " + " ".join(toks) + " "
        for i in range(len(flat) - 2):
            k = "c:" + flat[i:i + 3]
            feats[k] = feats.get(k, 0) + 0.15
        return {k: 1 + math.log(v) if v >= 1 else v for k, v in feats.items()}

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for f, w in self._features(normalize(t)).items():
                h = _h(f)
                out[i, h % self.dim] += w if (h >> 63) & 1 else -w
            n = np.linalg.norm(out[i])
            if n:
                out[i] /= n
        return out


class OpenAIEmbedder(Embedder):
    name = "openai"

    def __init__(self, model: str = "text-embedding-3-small"):
        import httpx
        self.model = model
        self.dim = 1536
        self.client = httpx.Client(timeout=60)
        self.key = os.environ["OPENAI_API_KEY"]

    def embed(self, texts: list[str]) -> np.ndarray:
        r = self.client.post("https://api.openai.com/v1/embeddings",
                             headers={"Authorization": f"Bearer {self.key}"},
                             json={"model": self.model, "input": texts})
        r.raise_for_status()
        arr = np.array([d["embedding"] for d in r.json()["data"]], dtype=np.float32)
        return arr / np.linalg.norm(arr, axis=1, keepdims=True)


class STEmbedder(Embedder):
    name = "sentence-transformers"

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # type: ignore
        self.m = SentenceTransformer(model)
        self.dim = self.m.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.m.encode(texts, normalize_embeddings=True), dtype=np.float32)


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    choice = os.getenv("TE_EMBEDDINGS", "hashing").lower()
    try:
        if choice == "openai" and os.getenv("OPENAI_API_KEY"):
            return OpenAIEmbedder()
        if choice in ("st", "sentence-transformers"):
            return STEmbedder()
    except Exception:
        pass
    return HashingEmbedder()
