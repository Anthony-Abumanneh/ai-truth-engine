"""Model registry: which models exist in this environment and what each is good at.

Prices are approximate list prices in USD per million tokens and only feed the
cost metrics; override them with TE_PRICE_<NAME>="in,out" if they drift.
Offline reasoners carry *simulated* prices scaled to their tier so that
cost/accuracy trade-offs remain measurable without any API spend.
"""
from __future__ import annotations

import os
from functools import lru_cache

from ..config import settings
from .base import LLMProvider, ModelSpec
from .http_providers import AnthropicProvider, OfflineProvider, OpenAIChatProvider

FREE_ENDPOINTS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}
ALL_ROLES = frozenset({"solver", "verifier", "critic", "judge", "analyzer", "extractor", "synthesizer"})


def _price(name: str, default: tuple[float, float]) -> tuple[float, float]:
    raw = os.getenv(f"TE_PRICE_{name.upper().replace('-', '_').replace('.', '_')}")
    if raw:
        a, b = raw.split(",")
        return float(a), float(b)
    return default


OFFLINE_MODELS = [
    ModelSpec("offline-swift", "offline", "swift", "small", "offline-lexical",
              frozenset({"general", "geography", "history"}), *_price("offline-swift", (0.25, 1.25)),
              strategy="quick", roles=frozenset({"solver", "verifier"}),
              description="Fast lexical reasoner: answers from the single best keyword match."),
    ModelSpec("offline-scholar", "offline", "scholar", "medium", "offline-semantic",
              frozenset({"science", "medicine", "history", "technology", "general"}), *_price("offline-scholar", (1.0, 5.0)),
              strategy="semantic", roles=frozenset({"solver", "verifier"}),
              description="Semantic reasoner: dense retrieval, aggregates several passages."),
    ModelSpec("offline-sage", "offline", "sage", "large", "offline-hybrid",
              frozenset({"general", "science", "business", "finance", "history", "technology", "law", "medicine"}),
              *_price("offline-sage", (3.0, 15.0)), strategy="careful", roles=ALL_ROLES,
              description="Careful reasoner: hybrid retrieval, cross-checks passages, prefers primary sources."),
    ModelSpec("offline-math", "offline", "math", "medium", "offline-symbolic",
              frozenset({"mathematics"}), *_price("offline-math", (1.0, 5.0)), strategy="symbolic",
              roles=frozenset({"solver"}), description="Mathematics specialist backed by a CAS."),
    ModelSpec("offline-numeric", "offline", "numeric", "small", "offline-numeric",
              frozenset({"mathematics", "finance"}), *_price("offline-numeric", (0.25, 1.25)), strategy="numeric",
              roles=frozenset({"solver"}), description="Independent numeric solver (sampling + root finding)."),
    ModelSpec("offline-coder", "offline", "coder", "medium", "offline-exec",
              frozenset({"coding"}), *_price("offline-coder", (1.0, 5.0)), strategy="code",
              roles=frozenset({"solver"}), description="Debugging specialist: executes code, reproduces failures."),
    ModelSpec("offline-static", "offline", "static", "small", "offline-static",
              frozenset({"coding"}), *_price("offline-static", (0.25, 1.25)), strategy="static",
              roles=frozenset({"solver"}), description="Independent static analyzer for code (AST lint rules)."),
    ModelSpec("offline-analyst", "offline", "analyst", "medium", "offline-sql",
              frozenset({"business", "finance"}), *_price("offline-analyst", (1.0, 5.0)), strategy="analyst",
              roles=frozenset({"solver"}), description="Business analyst: SQL variance decomposition over company data."),
    ModelSpec("offline-analyst-b", "offline", "analyst_b", "small", "offline-sql-b",
              frozenset({"business", "finance"}), *_price("offline-analyst-b", (0.25, 1.25)), strategy="analyst_top",
              roles=frozenset({"solver"}), description="Second analyst: top-down contribution ranking only."),
]


def _llm_models() -> list[ModelSpec]:
    s = settings
    out: list[ModelSpec] = []
    if s.force_offline:
        return out
    if s.anthropic_key:
        out += [
            ModelSpec("claude-strong", "anthropic", s.anthropic_strong, "large", "anthropic",
                      frozenset({"mathematics", "coding", "science", "business", "finance", "law", "medicine", "general", "technology", "history"}),
                      *_price("claude-strong", (5.0, 25.0)), roles=ALL_ROLES, description=f"Anthropic {s.anthropic_strong}"),
            ModelSpec("claude-mid", "anthropic", s.anthropic_mid, "medium", "anthropic",
                      frozenset({"coding", "general", "science", "business", "technology"}),
                      *_price("claude-mid", (3.0, 15.0)), roles=ALL_ROLES, description=f"Anthropic {s.anthropic_mid}"),
            ModelSpec("claude-fast", "anthropic", s.anthropic_fast, "small", "anthropic",
                      frozenset({"general", "geography", "history"}),
                      *_price("claude-fast", (1.0, 5.0)), roles=ALL_ROLES, description=f"Anthropic {s.anthropic_fast}"),
        ]
    if s.openai_key:
        out += [
            ModelSpec("gpt-strong", "openai", s.openai_strong, "large", "openai",
                      frozenset({"mathematics", "coding", "science", "business", "finance", "general", "technology"}),
                      *_price("gpt-strong", (2.0, 8.0)), roles=ALL_ROLES, description=f"OpenAI {s.openai_strong}"),
            ModelSpec("gpt-fast", "openai", s.openai_fast, "small", "openai",
                      frozenset({"general", "geography", "history"}),
                      *_price("gpt-fast", (0.4, 1.6)), roles=ALL_ROLES, description=f"OpenAI {s.openai_fast}"),
        ]
    # Free tiers: real calls, $0 actual cost (prices left at 0; rate limits apply)
    for prov, key, models, family in (("gemini", s.gemini_key, s.gemini_models, "google"),
                                      ("groq", s.groq_key, s.groq_models, None),
                                      ("openrouter", s.openrouter_key, s.openrouter_models, None)):
        if not key:
            continue
        for m in models:
            fam = family or m.lower().split("/")[-1].split("-")[0]       # llama / qwen / gemma ...
            big = any(x in m.lower() for x in ("70b", "72b", "pro", "405b", "235b", "120b"))
            out.append(ModelSpec(f"{prov}:{m}", prov, m, "large" if big else "medium", fam,
                                 frozenset({"general", "science", "medicine", "history", "geography", "technology"}),
                                 0.0, 0.0, roles=ALL_ROLES, description=f"{prov} free tier: {m}"))
    if s.compat_base_url and s.compat_models:
        for m in s.compat_models:
            lower = m.lower()
            strengths = {"general"}
            if "math" in lower:
                strengths = {"mathematics"}
            elif "coder" in lower or "code" in lower:
                strengths = {"coding"}
            else:
                strengths |= {"science", "business", "technology", "history"}
            tier = "large" if any(x in lower for x in ("72b", "70b", "110b", "235b", "405b", "max")) else "medium"
            out.append(ModelSpec(f"compat:{m}", "compat", m, tier, f"compat-{lower.split('-')[0]}",
                                 frozenset(strengths), *_price(f"compat-{m}", (0.5, 1.5)), roles=ALL_ROLES,
                                 description=f"OpenAI-compatible {m}"))
    return out


class Registry:
    def __init__(self):
        self.models: dict[str, ModelSpec] = {}
        for m in OFFLINE_MODELS + _llm_models():
            self.models[m.name] = m
        self.providers: dict[str, LLMProvider] = {"offline": OfflineProvider()}
        if settings.anthropic_key and not settings.force_offline:
            self.providers["anthropic"] = AnthropicProvider(settings.anthropic_key)
        if settings.openai_key and not settings.force_offline:
            self.providers["openai"] = OpenAIChatProvider(settings.openai_key)
        if settings.compat_base_url and not settings.force_offline:
            self.providers["compat"] = OpenAIChatProvider(settings.compat_key, settings.compat_base_url)
        for name, key, url in (("gemini", settings.gemini_key, FREE_ENDPOINTS["gemini"]),
                               ("groq", settings.groq_key, FREE_ENDPOINTS["groq"]),
                               ("openrouter", settings.openrouter_key, FREE_ENDPOINTS["openrouter"])):
            if key and not settings.force_offline:
                self.providers[name] = OpenAIChatProvider(key, url)

    def get(self, name: str) -> ModelSpec:
        return self.models[name]

    def provider_for(self, spec: ModelSpec) -> LLMProvider:
        return self.providers[spec.provider]

    @property
    def llms(self) -> list[ModelSpec]:
        return [m for m in self.models.values() if m.is_llm]

    @property
    def has_llm(self) -> bool:
        return bool(self.llms)

    def for_role(self, role: str, *, llm_only: bool = False) -> list[ModelSpec]:
        return [m for m in self.models.values() if role in m.roles and (m.is_llm or not llm_only)]

    def describe(self) -> list[dict]:
        return [{"name": m.name, "provider": m.provider, "api_model": m.api_model, "tier": m.tier,
                 "family": m.family, "strengths": sorted(m.strengths), "roles": sorted(m.roles),
                 "price_in_per_m": m.cost_in_per_m, "price_out_per_m": m.cost_out_per_m,
                 "simulated_price": not m.is_llm, "description": m.description}
                for m in self.models.values()]


@lru_cache(maxsize=1)
def get_registry() -> Registry:
    return Registry()


def reset_registry() -> None:
    get_registry.cache_clear()
