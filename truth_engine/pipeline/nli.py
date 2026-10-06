"""Local neural NLI model (CONCEPTS.md §4: natural language inference).

A DeBERTa-v3 cross-encoder fine-tuned on MNLI, FEVER (fact verification)
and ANLI (adversarial NLI) reads an evidence sentence and a claim *together*
and outputs P(entailment), P(neutral), P(contradiction). Unlike the word-
overlap rules, it handles paraphrase and implied contradiction.

Two backends, same model:
  torch  original weights (369 MB) via PyTorch + transformers. Default when
         installed: the reference implementation, and the one fine-tuning needs.
  onnx   8-bit quantized export (244 MB) via ONNX Runtime; no PyTorch needed.

    python -m truth_engine.pipeline.nli download [torch|onnx]

Env: TE_NLI=0 disables it; TE_NLI_BACKEND=torch|onnx|auto (default auto).
"""
from __future__ import annotations

import json
import os
import sys
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..config import DATA_DIR

BACKENDS = {
    "torch": ("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
              ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "added_tokens.json", "spm.model")),
    "onnx": ("Xenova/DeBERTa-v3-base-mnli-fever-anli",
             ("config.json", "tokenizer.json", "onnx/model_quantized.onnx")),
}
MAX_TOKENS = 384


def model_dir(backend: str) -> Path:
    return DATA_DIR / "models" / BACKENDS[backend][0].replace("/", "__")


def is_downloaded(backend: str) -> bool:
    return all((model_dir(backend) / f).exists() for f in BACKENDS[backend][1])


def download(backend: str = "torch") -> Path:
    import httpx
    repo, files = BACKENDS[backend]
    out = model_dir(backend)
    for f in files:
        dest = out / f
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with httpx.stream("GET", f"https://huggingface.co/{repo}/resolve/main/{f}", follow_redirects=True,
                          timeout=120) as r:
            r.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in r.iter_bytes(1 << 20):
                    fh.write(chunk)
        tmp.rename(dest)
    return out


class _Base:
    name = ""
    backend = ""

    def __init__(self):
        self._lock = threading.Lock()
        self._memo: dict[tuple[str, str], dict[str, float]] = {}

    def predict(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        """pairs = [(premise/evidence, hypothesis/claim)] -> [{"entailment": p, "neutral": p, "contradiction": p}]"""
        todo = [p for p in dict.fromkeys(pairs) if p not in self._memo]
        if todo:
            with self._lock:
                for pair, row in zip(todo, self._run(todo)):
                    self._memo[pair] = {self.labels[j]: round(float(p), 4) for j, p in enumerate(row)}
        return [self._memo[p] for p in pairs]

    def _run(self, pairs):  # pragma: no cover - interface
        raise NotImplementedError


class TorchNLIModel(_Base):
    backend = "torch"

    def __init__(self, path: Path, batch_size: int = 16):
        super().__init__()
        import warnings

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.float32).eval()
        self.labels = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.name = BACKENDS["torch"][0]
        self.batch_size = batch_size

    def _run(self, pairs):
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(pairs), self.batch_size):
                chunk = pairs[i:i + self.batch_size]
                enc = self.tok([p for p, _ in chunk], [h for _, h in chunk], truncation=True,
                               max_length=MAX_TOKENS, padding=True, return_tensors="pt")
                out += self.torch.softmax(self.model(**enc).logits, dim=-1).tolist()
        return out


class OnnxNLIModel(_Base):
    backend = "onnx"

    def __init__(self, path: Path):
        super().__init__()
        import onnxruntime as ort
        from tokenizers import Tokenizer
        cfg = json.loads((path / "config.json").read_text())
        self.labels = {int(k): v.lower() for k, v in cfg["id2label"].items()}
        self.tok = Tokenizer.from_file(str(path / "tokenizer.json"))
        # No padding: this export answers differently for a padded sentence
        # than for the same sentence alone, so each pair runs on its own.
        self.tok.enable_truncation(MAX_TOKENS)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, (os.cpu_count() or 2) // 2)
        self.session = ort.InferenceSession(str(path / "onnx/model_quantized.onnx"), opts,
                                            providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.session.get_inputs()}
        self.name = BACKENDS["onnx"][0] + " (int8)"

    def _run(self, pairs):
        out = []
        for e in self.tok.encode_batch(pairs):
            feed = {"input_ids": np.array([e.ids], dtype=np.int64),
                    "attention_mask": np.array([e.attention_mask], dtype=np.int64)}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.array([e.type_ids], dtype=np.int64)
            logits = self.session.run(None, feed)[0][0]
            p = np.exp(logits - logits.max())
            out.append(p / p.sum())
        return out


def load(backend: str) -> _Base | None:
    if not is_downloaded(backend):
        return None
    try:
        return TorchNLIModel(model_dir("torch")) if backend == "torch" else OnnxNLIModel(model_dir("onnx"))
    except Exception:
        return None


@lru_cache(maxsize=1)
def get_nli() -> _Base | None:
    """The local NLI model, or None when disabled or not downloaded.
    auto = PyTorch backend if available, else ONNX."""
    if os.getenv("TE_NLI", "1").lower() in ("0", "false", "no"):
        return None
    choice = os.getenv("TE_NLI_BACKEND", "auto").lower()
    for backend in (["torch", "onnx"] if choice == "auto" else [choice]):
        model = load(backend)
        if model is not None:
            return model
    return None


if __name__ == "__main__":
    if sys.argv[1:2] == ["download"]:
        print("saved to", download(sys.argv[2] if len(sys.argv) > 2 else "torch"))
    else:
        print(__doc__)
