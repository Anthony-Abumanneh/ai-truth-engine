"""Retrieval recall on SciFact: does an abstract that holds the gold evidence
appear among the top-k retrieved sentences? Compares embedders (and BM25-only,
vector-only, hybrid). Uses TRAIN claims by default so dev stays untouched.

    python scripts/eval_retrieval_scifact.py [n_claims] [model ...]
    SPLIT=dev python scripts/eval_retrieval_scifact.py 300 hashing BAAI/bge-base-en-v1.5   # final held-out check
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from truth_engine.bench.scifact import build_engine, load_claims          # noqa: E402
from truth_engine.retrieval.embeddings import HashingEmbedder, STEmbedder  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
models = sys.argv[2:] or ["hashing", "sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5",
                          "BAAI/bge-base-en-v1.5", "pritamdeka/S-PubMedBert-MS-MARCO"]
claims = [c for c in load_claims(os.getenv("SPLIT", "train")) if c["evidence_docs"]][:n]
eng = build_engine()
store = eng.store
for model in models:
    t = time.time()
    store.embedder = HashingEmbedder() if model == "hashing" else STEmbedder(model)
    store._dirty = True
    store._ensure_index()
    build = time.time() - t
    for method in (["bm25"] if model == "hashing" else []) + ["vector", "hybrid"]:
        hits = {5: 0, 10: 0, 20: 0}
        for c in claims:
            docs = [p.source_id for p in store.search(c["claim"], k=20, method=method)]
            for k in hits:
                hits[k] += bool(set(docs[:k]) & set(c["evidence_docs"]))
        print(f"{model:45} {method:7} recall@5 {hits[5]/len(claims):.3f}  @10 {hits[10]/len(claims):.3f}  "
              f"@20 {hits[20]/len(claims):.3f}   (index {build:.0f}s)", flush=True)
