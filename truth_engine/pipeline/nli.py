"""Local neural NLI model (CONCEPTS.md §4: natural language inference).

A DeBERTa-v3 cross-encoder fine-tuned on NLI and fact-verification data reads
an evidence sentence and a claim *together* and outputs P(entailment),
P(neutral), P(contradiction). Unlike the word-overlap rules, it handles
paraphrase and implied contradiction.

Runs on CPU with PyTorch + transformers. (An 8-bit ONNX export was tried in
step 4.5 and dropped: it lost real accuracy.)

    python -m truth_engine.pipeline.nli download [model]

Env: TE_NLI=0 disables it; TE_NLI_MODEL picks a key of MODELS;
TE_DEVICE=auto|cpu|mps|cuda (auto: CUDA when present, else CPU. Apple's GPU
("mps") is opt-in: it pins memory that can't be swapped, which starved an
8 GB Mac running the large model, for only ~25% speedup).
"""
from __future__ import annotations

import os
import sys
import threading
import warnings
from functools import lru_cache
from pathlib import Path

from ..config import DATA_DIR

MODELS = {
    "base": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    "large": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
}
DEFAULT_MODEL = "large"
REQUIRED = ("config.json", "model.safetensors")
OPTIONAL = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "spm.model")
MAX_TOKENS = 256          # premise (sentence + neighbours) + claim fit comfortably


def model_key() -> str:
    return os.getenv("TE_NLI_MODEL", DEFAULT_MODEL)


def model_dir(key: str | None = None) -> Path:
    return DATA_DIR / "models" / MODELS[key or model_key()].replace("/", "__")


def is_downloaded(key: str | None = None) -> bool:
    return all((model_dir(key) / f).exists() for f in REQUIRED)


def download(key: str | None = None) -> Path:
    import httpx
    repo = MODELS[key or model_key()]
    out = model_dir(key)
    for f in REQUIRED + OPTIONAL:
        dest = out / f
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with httpx.stream("GET", f"https://huggingface.co/{repo}/resolve/main/{f}", follow_redirects=True,
                          timeout=120) as r:
            if r.status_code == 404 and f in OPTIONAL:
                continue
            r.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in r.iter_bytes(1 << 20):
                    fh.write(chunk)
        tmp.rename(dest)
    return out


class NLIModel:
    """DeBERTa cross-encoder on CPU; answers are memoized per (premise, claim)."""

    def __init__(self, path: Path, batch_size: int = 16):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
        self.torch = torch
        self.device = _device(torch)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.float32).eval()
        self.model.to(self.device)
        self.labels = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.name = path.name.replace("__", "/")
        self.batch_size = batch_size
        self._lock = threading.Lock()
        self._memo: dict[tuple[str, str], dict[str, float]] = {}

    def predict(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        """pairs = [(premise/evidence, hypothesis/claim)] -> [{"entailment": p, "neutral": p, "contradiction": p}]"""
        todo = [p for p in dict.fromkeys(pairs) if p not in self._memo]
        todo.sort(key=lambda p: len(p[0]) + len(p[1]))      # similar lengths per batch: less padding
        with self._lock, self.torch.inference_mode():
            for i in range(0, len(todo), self.batch_size):
                chunk = todo[i:i + self.batch_size]
                enc = self.tok([p for p, _ in chunk], [h for _, h in chunk], truncation=True,
                               max_length=MAX_TOKENS, padding=True, return_tensors="pt").to(self.device)
                for pair, row in zip(chunk, self.torch.softmax(self.model(**enc).logits, dim=-1).tolist()):
                    self._memo[pair] = {self.labels[j]: round(float(p), 4) for j, p in enumerate(row)}
        return [self._memo[p] for p in pairs]


def device() -> str:
    import torch
    return _device(torch)


def _device(torch) -> str:
    choice = os.getenv("TE_DEVICE", "auto").lower()
    if choice != "auto":
        return choice
    return "cuda" if torch.cuda.is_available() else "cpu"


def load(key: str | None = None) -> NLIModel | None:
    if not is_downloaded(key):
        return None
    try:
        return NLIModel(model_dir(key))
    except Exception:
        return None


@lru_cache(maxsize=1)
def get_nli() -> NLIModel | None:
    """The local NLI model, or None when disabled or not downloaded."""
    if os.getenv("TE_NLI", "1").lower() in ("0", "false", "no"):
        return None
    return load()


if __name__ == "__main__":
    if sys.argv[1:2] == ["download"]:
        print("saved to", download(sys.argv[2] if len(sys.argv) > 2 else None))
    else:
        print(__doc__)
