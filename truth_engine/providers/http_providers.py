"""Real model adapters: Anthropic Messages API, OpenAI Chat Completions, and any
OpenAI-compatible server (vLLM, Together, DashScope/Qwen, Ollama, LM Studio)."""
from __future__ import annotations

import time

import httpx

from ..config import settings
from .base import LLMProvider, LLMResponse, ModelSpec, ProviderError, with_retries


def _err(resp: httpx.Response) -> ProviderError:
    e = ProviderError(f"HTTP {resp.status_code}: {resp.text[:400]}")
    e.status = resp.status_code  # type: ignore[attr-defined]
    return e


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    url = "https://api.anthropic.com/v1/messages"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = httpx.Client(timeout=settings.request_timeout_s)

    def complete(self, spec: ModelSpec, system: str, user: str, *, temperature: float = 0.2,
                 max_tokens: int = 1500) -> LLMResponse:
        t0 = time.time()

        def call():
            r = self.client.post(self.url, headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }, json={
                "model": spec.api_model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            })
            if r.status_code != 200:
                raise _err(r)
            return r.json()

        data = with_retries(call)
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage", {})
        ti, to = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        return LLMResponse(text=text, model=spec.name, tokens_in=ti, tokens_out=to,
                           cost_usd=spec.price(ti, to), latency_s=time.time() - t0, raw=data)


class OpenAIChatProvider(LLMProvider):
    name = "openai"

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=settings.request_timeout_s)

    def complete(self, spec: ModelSpec, system: str, user: str, *, temperature: float = 0.2,
                 max_tokens: int = 1500) -> LLMResponse:
        t0 = time.time()
        body: dict = {
            "model": spec.api_model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        # Reasoning-family models reject temperature/max_tokens; use the newer field.
        if spec.api_model.startswith(("o1", "o3", "o4", "gpt-5")):
            body["max_completion_tokens"] = max_tokens
        else:
            body["temperature"] = temperature
            body["max_tokens"] = max_tokens

        def call():
            r = self.client.post(f"{self.base_url}/chat/completions", headers={
                "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, json=body)
            if r.status_code != 200:
                raise _err(r)
            return r.json()

        data = with_retries(call)
        text = data["choices"][0]["message"].get("content") or ""
        usage = data.get("usage", {}) or {}
        ti, to = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        return LLMResponse(text=text, model=spec.name, tokens_in=ti, tokens_out=to,
                           cost_usd=spec.price(ti, to), latency_s=time.time() - t0, raw=data)


class OfflineProvider(LLMProvider):
    """Placeholder for offline models. Offline models never generate free text;
    each pipeline stage runs its deterministic algorithm instead."""
    name = "offline"

    def complete(self, spec, system, user, **kw) -> LLMResponse:
        raise ProviderError("offline models do not generate free text")
