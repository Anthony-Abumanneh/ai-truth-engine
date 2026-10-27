"""Model versioning (step 11): pinned revisions + a checksum manifest.

Every downloaded model is pinned to an exact Hugging Face commit, so a rebuild
next year gets the same weights. `write_manifest()` records a SHA-256 for each
model file plus the locally trained artifacts (calibrator, design classifier,
fine-tuned verifiers); the API's /api/health reports it, and `verify()` checks
the files on disk against it (CI and the Docker build run it).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from .config import DATA_DIR

# Hugging Face commits (checked 2026-10-27). Bump deliberately, then rerun the benchmark.
REVISIONS = {
    "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli": "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7",
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli": "6f5cf0a2b59cabb106aca4c287eed12e357e90eb",
    "cross-encoder/ms-marco-MiniLM-L6-v2": "233902d25c440f23af6f7d6e94d2946bac0bee0a",
    "BAAI/bge-base-en-v1.5": "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
}
MANIFEST = DATA_DIR / "models" / "MANIFEST.json"
TRACKED = ("model.safetensors", "config.json", "tokenizer.json", "spm.model", "vocab.txt")


def revision(repo: str) -> str:
    return REVISIONS.get(repo, "main")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def artifacts() -> dict[str, Path]:
    """Every model artifact present on disk, by a stable name."""
    out: dict[str, Path] = {}
    models = DATA_DIR / "models"
    if models.exists():
        for d in sorted(p for p in models.iterdir() if p.is_dir()):
            for f in TRACKED:
                if (d / f).exists():
                    out[f"{d.name}/{f}"] = d / f
        clf = models / "study_design_clf.joblib"
        if clf.exists():
            out[clf.name] = clf
    cal = DATA_DIR / "calibrator.json"
    if cal.exists():
        out["calibrator.json"] = cal
    return out


def write_manifest() -> dict:
    entries = {name: {"sha256": _sha256(p), "bytes": p.stat().st_size} for name, p in artifacts().items()}
    m = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "revisions": REVISIONS, "files": entries}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=1))
    return m


def verify() -> list[str]:
    """Files whose checksum no longer matches the manifest (empty = all good)."""
    if not MANIFEST.exists():
        return ["no manifest: run `truth-engine setup`"]
    m = json.loads(MANIFEST.read_text())
    bad = []
    for name, p in artifacts().items():
        want = m["files"].get(name)
        if want and want["sha256"] != _sha256(p):
            bad.append(name)
    return bad


def summary() -> dict:
    """Short version info for /api/health."""
    if not MANIFEST.exists():
        return {"manifest": None}
    m = json.loads(MANIFEST.read_text())
    return {"manifest": m["created"], "files": len(m["files"]),
            "revisions": {k.split("/")[-1]: v[:10] for k, v in m["revisions"].items()}}
