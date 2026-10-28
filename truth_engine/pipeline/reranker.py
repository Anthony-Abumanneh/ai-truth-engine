"""Evidence reranker: "is this sentence about the claim?" (step 8.5).

A small cross-encoder (MiniLM, 22M parameters) reads the claim and a sentence
together and scores relevance. It replaces word overlap as the gate deciding
which retrieved sentences the NLI model reads. scripts/diagnose_scifact.py
found the word-overlap gate discarded the expert's evidence sentence for 27% of
SciFact dev claims (paraphrases share few words with the claim).

    TE_RERANKER=0 disables it; TE_RERANKER_MODEL picks a key of MODELS.
"""
from __future__ import annotations

import os
import threading
from functools import lru_cache

from ..config import DATA_DIR

MODELS = {
    "msmarco": "cross-encoder/ms-marco-MiniLM-L6-v2",          # off the shelf (web search relevance)
    "ft": "local/truth-engine-reranker",                        # fine-tuned on SciFact train (if trained)
}
MAX_TOKENS = 256


def model_dir(key: str):
    return DATA_DIR / "models" / MODELS[key].replace("/", "__")


def download(key: str = "msmarco"):
    """Fetch the off-the-shelf reranker from Hugging Face (91 MB)."""
    import httpx
    out = model_dir(key)
    out.mkdir(parents=True, exist_ok=True)
    for f in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
              "special_tokens_map.json", "vocab.txt"):
        if (out / f).exists():
            continue
        from ..models_registry import revision
        r = httpx.get(f"https://huggingface.co/{MODELS[key]}/resolve/{revision(MODELS[key])}/{f}",
                      follow_redirects=True, timeout=120)
        if r.status_code == 200:
            (out / f).write_bytes(r.content)
        elif f in ("config.json", "model.safetensors"):
            r.raise_for_status()
    return out


class Reranker:
    def __init__(self, key: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        path = model_dir(key)
        self.torch, self.name = torch, MODELS[key]
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.float32).eval()
        # Copy weights out of the memory-mapped file: on this Mac's BLAS the mapped
        # (unaligned) MiniLM tensors produced NaN in every matmul; copies are exact and fine.
        with torch.no_grad():
            for prm in self.model.parameters():
                prm.data = prm.data.clone()
        self._lock = threading.Lock()
        self._memo: dict[tuple[str, str], float] = {}

    def score(self, pairs: list[tuple[str, str]], batch: int = 32) -> list[float]:
        """pairs = [(claim, sentence)] -> relevance in [0, 1] (sigmoid of the model's logit)."""
        todo = sorted((p for p in dict.fromkeys(pairs) if p not in self._memo), key=lambda p: len(p[1]))
        with self._lock, self.torch.inference_mode():
            for i in range(0, len(todo), batch):
                chunk = todo[i:i + batch]
                enc = self.tok([c for c, _ in chunk], [s for _, s in chunk], truncation=True, max_length=MAX_TOKENS,
                               padding=True, return_tensors="pt")
                logits = self.model(**enc).logits[:, 0]
                for p, v in zip(chunk, self.torch.sigmoid(logits).tolist()):
                    self._memo[p] = round(float(v), 4)
        return [self._memo[p] for p in pairs]


def model_key() -> str:
    key = os.getenv("TE_RERANKER_MODEL")
    if key:
        return key
    return "ft" if (model_dir("ft") / "model.safetensors").exists() else "msmarco"


@lru_cache(maxsize=1)
def get_reranker() -> Reranker | None:
    if os.getenv("TE_RERANKER", "1").lower() in ("0", "false", "no"):
        return None
    key = model_key()
    if not (model_dir(key) / "model.safetensors").exists():
        return None
    try:
        return Reranker(key)
    except Exception:
        return None
