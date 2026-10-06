"""Runtime configuration, read from environment variables.

Set any of these to enable real models (nothing else needs to change):

    ANTHROPIC_API_KEY        enables claude-* models
    OPENAI_API_KEY           enables gpt-* / o* models
    OPENAI_COMPAT_BASE_URL   any OpenAI-compatible server (vLLM, Together,
    OPENAI_COMPAT_API_KEY    DashScope/Qwen, Ollama, LM Studio ...)
    OPENAI_COMPAT_MODELS     comma list, e.g. "qwen2.5-math-72b,qwen2.5-72b"
    TAVILY_API_KEY / BRAVE_API_KEY / SERPER_API_KEY   live web retrieval
    NCBI_API_KEY             optional: PubMed at 10 req/s instead of 3
    OPENALEX_EMAIL           optional: OpenAlex "polite pool"

Without keys the engine runs fully offline on deterministic reasoners.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("TRUTH_ENGINE_DATA", ROOT / "data"))
DB_PATH = Path(os.getenv("TRUTH_ENGINE_DB", DATA_DIR / "truth_engine.sqlite"))


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@dataclass
class Settings:
    anthropic_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    openai_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    compat_base_url: str = field(default_factory=lambda: _env("OPENAI_COMPAT_BASE_URL"))
    compat_key: str = field(default_factory=lambda: _env("OPENAI_COMPAT_API_KEY", "none"))
    compat_models: list[str] = field(default_factory=lambda: [m.strip() for m in _env("OPENAI_COMPAT_MODELS").split(",") if m.strip()])
    tavily_key: str = field(default_factory=lambda: _env("TAVILY_API_KEY"))
    brave_key: str = field(default_factory=lambda: _env("BRAVE_API_KEY"))
    serper_key: str = field(default_factory=lambda: _env("SERPER_API_KEY"))
    ncbi_key: str = field(default_factory=lambda: _env("NCBI_API_KEY"))
    openalex_email: str = field(default_factory=lambda: _env("OPENALEX_EMAIL"))

    # Model choices per role when real providers are present (override via env)
    anthropic_strong: str = field(default_factory=lambda: _env("TE_ANTHROPIC_STRONG", "claude-opus-5"))
    anthropic_fast: str = field(default_factory=lambda: _env("TE_ANTHROPIC_FAST", "claude-haiku-4-5-20251001"))
    anthropic_mid: str = field(default_factory=lambda: _env("TE_ANTHROPIC_MID", "claude-sonnet-5-5"))
    openai_strong: str = field(default_factory=lambda: _env("TE_OPENAI_STRONG", "gpt-4.1"))
    openai_fast: str = field(default_factory=lambda: _env("TE_OPENAI_FAST", "gpt-4.1-mini"))

    request_timeout_s: float = float(_env("TE_TIMEOUT", "90"))
    max_parallel: int = int(_env("TE_MAX_PARALLEL", "6"))
    force_offline: bool = _env("TE_OFFLINE", "0") in {"1", "true", "yes"}

    @property
    def has_web(self) -> bool:
        return bool(self.tavily_key or self.brave_key or self.serper_key)


settings = Settings()


def reload_settings() -> Settings:
    global settings
    settings = Settings()
    return settings
