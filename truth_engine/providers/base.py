"""Provider abstraction: one interface for every model the engine can route to."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class LLMResponse:
    text: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    raw: Any = None


@dataclass
class ModelSpec:
    """A routable model. `family` is used to measure correlated-error risk:
    two models from the same family are assumed less independent."""
    name: str
    provider: Literal["anthropic", "openai", "compat", "gemini", "groq", "openrouter", "offline"]
    api_model: str
    tier: Literal["small", "medium", "large"]
    family: str
    strengths: frozenset[str] = frozenset()
    cost_in_per_m: float = 0.0
    cost_out_per_m: float = 0.0
    strategy: str | None = None        # offline reasoning strategy
    roles: frozenset[str] = frozenset({"solver"})
    description: str = ""

    @property
    def is_llm(self) -> bool:
        return self.provider != "offline"

    def price(self, tokens_in: int, tokens_out: int) -> float:
        return tokens_in / 1e6 * self.cost_in_per_m + tokens_out / 1e6 * self.cost_out_per_m


class ProviderError(RuntimeError):
    pass


class LLMProvider:
    name = "base"

    def complete(self, spec: ModelSpec, system: str, user: str, *, temperature: float = 0.2,
                 max_tokens: int = 1500) -> LLMResponse:  # pragma: no cover - interface
        raise NotImplementedError

    def complete_json(self, spec: ModelSpec, system: str, user: str, *, temperature: float = 0.0,
                      max_tokens: int = 1500) -> tuple[dict | list | None, LLMResponse]:
        sys = system + "\n\nRespond with a single valid JSON value and nothing else."
        resp = self.complete(spec, sys, user, temperature=temperature, max_tokens=max_tokens)
        return extract_json(resp.text), resp


def extract_json(text: str) -> Any:
    """Pull the first JSON object/array out of model text (handles ```json fences)."""
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1)
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start = text.find(open_c)
        if start < 0:
            continue
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == open_c:
                depth += 1
            elif ch == close_c:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except Exception:
                        break
    return None


def with_retries(fn, *, attempts: int = 4, base_delay: float = 1.5, retry_on=(429, 500, 502, 503, 504, 529)):
    """Exponential backoff wrapper for HTTP calls that raise ProviderError(status)."""
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except ProviderError as e:
            last = e
            status = getattr(e, "status", None)
            if status not in retry_on or i == attempts - 1:
                raise
            time.sleep(base_delay * (2 ** i))
    raise last  # type: ignore[misc]
