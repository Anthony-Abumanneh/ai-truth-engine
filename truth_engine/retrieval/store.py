"""Document store with sentence-level passages and hybrid (BM25 + dense) search.

Documents are Markdown files with a small front-matter header (see
data/corpus/). Drop new files in, or add them at runtime via `add_document`
or the /api/documents endpoint. Each sentence becomes a passage, which keeps
evidence precise: the verifier judges one statement at a time instead of a
whole page that merely *mentions* the claim (spec §1).
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import DATA_DIR
from ..schemas import Passage, Source
from ..science.study_design import classify
from ..textutils import sentences
from .bm25 import BM25
from .embeddings import get_embedder
from .source_quality import score_source

TYPE_DEFAULT_AUTHORITY = {
    "primary_record": 0.95, "filing": 0.9, "database": 0.9, "internal": 0.85, "paper": 0.8, "review": 0.88,
    "government": 0.9, "encyclopedia": 0.8, "press_release": 0.7, "analyst_report": 0.65, "reference": 0.6,
    "news": 0.5, "web": 0.45, "blog": 0.25, "forum": 0.15, "marketing": 0.1, "document": 0.5,
}


def parse_front_matter(text: str) -> tuple[dict, str]:
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.S)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, m.group(2)


def _as_list(v) -> list[str]:
    if not v:
        return []
    if isinstance(v, list):
        return v
    return [x.strip() for x in str(v).strip("[]").split(",") if x.strip()]


@dataclass
class _Row:
    passage: Passage
    doc_text: str


class DocumentStore:
    def __init__(self):
        self.sources: dict[str, Source] = {}
        self.rows: list[_Row] = []
        self.bm25 = BM25()
        self.vecs: np.ndarray | None = None
        self.embedder = get_embedder()
        self._dirty = True
        self._pos: dict[str, int] = {}
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- loading
    def load_dir(self, path: Path) -> int:
        n = 0
        for f in sorted(Path(path).rglob("*.md")):
            meta, body = parse_front_matter(f.read_text())
            meta.setdefault("id", f.stem)
            meta.setdefault("title", f.stem.replace("_", " ").title())
            self.add_document(body, **meta)
            n += 1
        return n

    def add_document(self, body: str, *, id: str, title: str, type: str = "document", url: str | None = None,
                     publisher: str | None = None, published: str | None = None, authority=None,
                     primary=None, cites=None, derived_from=None, **extra) -> Source:
        stype = str(type)
        auth = float(authority) if authority not in (None, "") else TYPE_DEFAULT_AUTHORITY.get(stype, 0.5)
        is_primary = str(primary).lower() in ("true", "1", "yes") if primary is not None else stype in (
            "primary_record", "filing", "database", "internal", "paper")
        src = Source(id=id, title=title, url=url, source_type=stype, publisher=publisher, published=published,
                     authority=auth, is_primary=is_primary, cites=_as_list(cites),
                     derived_from=_as_list(derived_from), metadata=extra)
        if stype in ("paper", "review", "news", "blog", "marketing", "forum"):
            prof = classify(title, body, source_type=stype, metadata=extra)
            src.metadata["study"] = prof.model_dump()
            src.metadata["study_label"] = prof.label
        src.quality = score_source(src)
        with self._lock:
            self.sources[id] = src
            self.rows = [r for r in self.rows if r.passage.source_id != id]
            for i, s in enumerate(sentences(body)):
                self.rows.append(_Row(Passage(id=f"{id}#s{i}", source_id=id, text=s), body))
            self._dirty = True
        return src

    def remove_document(self, doc_id: str) -> bool:
        with self._lock:
            existed = doc_id in self.sources
            self.sources.pop(doc_id, None)
            self.rows = [r for r in self.rows if r.passage.source_id != doc_id]
            self._dirty = True
        return existed

    def _ensure_index(self):
        if not self._dirty:
            return
        with self._lock:
            texts = [self._index_text(r) for r in self.rows]
            self.bm25.fit(texts)
            self._pos = {r.passage.id: i for i, r in enumerate(self.rows)}
            self.vecs = self._embed_cached(texts)
            self._dirty = False

    def _embed_cached(self, texts: list[str]) -> np.ndarray:
        """Neural embeddings are slow to compute: cache them on disk, keyed by model + texts."""
        if not texts:
            return np.zeros((0, self.embedder.dim), dtype=np.float32)
        if self.embedder.name.startswith("hashing"):
            return self.embedder.embed(texts)
        import hashlib
        key = hashlib.sha1((self.embedder.name + "\n" + "\n".join(texts)).encode()).hexdigest()[:20]
        path = DATA_DIR / "cache" / "embeddings" / f"{key}.npy"
        if path.exists():
            return np.load(path)
        vecs = self.embedder.embed(texts)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vecs)
        return vecs

    def _index_text(self, r: _Row) -> str:
        # title gives each sentence its subject (pronoun-only sentences become findable)
        return f"{self.sources[r.passage.source_id].title}. {r.passage.text}"

    # ---------------------------------------------------------------- search
    def search(self, query: str, k: int = 8, method: str = "hybrid", source_ids: set[str] | None = None,
               exclude_types: set[str] | None = None) -> list[Passage]:
        self._ensure_index()
        if not self.rows:
            return []
        idx = [i for i, r in enumerate(self.rows)
               if (source_ids is None or r.passage.source_id in source_ids)
               and (not exclude_types or self.sources[r.passage.source_id].source_type not in exclude_types)]
        if not idx:
            return []
        bm = self.bm25.scores(query)[idx] if method in ("hybrid", "bm25") else None
        dv = (self.vecs[idx] @ self.embedder.embed_query(query)) if method in ("hybrid", "vector") else None

        if method == "bm25":
            fused = bm
        elif method == "vector":
            fused = dv
        else:
            # Reciprocal-rank fusion: robust to the two scorers' different scales.
            fused = np.zeros(len(idx))
            for arr, w in ((bm, 1.0), (dv, 1.0)):
                order = np.argsort(-arr)
                ranks = np.empty(len(idx))
                ranks[order] = np.arange(len(idx))
                fused += w / (60 + ranks)
            # require at least some lexical or semantic signal
            fused = np.where((bm > 0) | (dv > 0.12), fused, 0)
        order = np.argsort(-fused)[:k]
        out = []
        for j in order:
            if fused[j] <= 0:
                continue
            r = self.rows[idx[j]].passage
            out.append(Passage(id=r.id, source_id=r.source_id, text=r.text, score=float(fused[j]),
                               retrieval_method=method))
        return out

    def neighbors(self, passage_id: str, window: int = 1) -> list[Passage]:
        """Adjacent sentences from the same document, for context."""
        self._ensure_index()
        i = self._pos.get(passage_id)
        if i is None:
            return []
        r = self.rows[i]
        lo, hi = max(0, i - window), min(len(self.rows), i + window + 1)
        return [x.passage for x in self.rows[lo:hi] if x.passage.source_id == r.passage.source_id]

    def document_text(self, source_id: str) -> str:
        return " ".join(r.passage.text for r in self.rows if r.passage.source_id == source_id)

    def stats(self) -> dict:
        types: dict[str, int] = {}
        for s in self.sources.values():
            types[s.source_type] = types.get(s.source_type, 0) + 1
        return {"documents": len(self.sources), "passages": len(self.rows), "embedder": self.embedder.name,
                "by_type": types}


_store: DocumentStore | None = None
_store_lock = threading.Lock()


def get_store() -> DocumentStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = DocumentStore()
            corpus = DATA_DIR / "corpus"
            if corpus.exists():
                _store.load_dir(corpus)
        return _store
