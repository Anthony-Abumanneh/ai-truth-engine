"""On-disk JSON cache for every external call (PubMed, OpenAlex).

Two jobs:
  1. Reproducibility: an investigation that hit the network once can be
     replayed later, offline, with byte-identical inputs.
  2. Politeness: PubMed and OpenAlex are free public services; we never ask
     them the same question twice.

Keys are a hash of (namespace, url, params) with secrets (api_key, email)
removed, so caches can be shared without leaking keys. In offline mode
(TE_OFFLINE=1) clients read from the cache only and never touch the network.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

from ..config import DATA_DIR

SECRET_PARAMS = {"api_key", "mailto", "email", "key"}


class JsonCache:
    def __init__(self, root: Path | None = None):
        self.root = Path(root or DATA_DIR / "cache")
        self._lock = threading.Lock()

    @staticmethod
    def key(namespace: str, url: str, params: dict | None = None) -> str:
        clean = {k: v for k, v in sorted((params or {}).items()) if k not in SECRET_PARAMS}
        raw = json.dumps([namespace, url, clean], sort_keys=True, default=str)
        return hashlib.sha1(raw.encode()).hexdigest()

    def _path(self, namespace: str, key: str) -> Path:
        return self.root / namespace / f"{key}.json"

    def get(self, namespace: str, url: str, params: dict | None = None) -> str | None:
        p = self._path(namespace, self.key(namespace, url, params))
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())["body"]
        except Exception:
            return None

    def put(self, namespace: str, url: str, params: dict | None, body: str) -> None:
        p = self._path(namespace, self.key(namespace, url, params))
        clean = {k: v for k, v in (params or {}).items() if k not in SECRET_PARAMS}
        with self._lock:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"url": url, "params": clean, "fetched_at": time.time(), "body": body}))


class RateLimiter:
    """Minimum spacing between requests (PubMed: 3/s without a key, 10/s with one)."""

    def __init__(self, per_second: float):
        self.interval = 1.0 / per_second
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            delay = self._last + self.interval - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._last = time.monotonic()


class CachedFetcher:
    """GET with cache, rate limit, retries and offline mode. Shared by the
    PubMed and OpenAlex clients. Failures never raise: they return None and are
    recorded in `errors`, so a dead network degrades retrieval instead of
    crashing an investigation."""

    def __init__(self, namespace: str, *, per_second: float, cache: JsonCache | None = None,
                 client=None, offline: bool | None = None, timeout: float = 20.0):
        import httpx

        from ..config import settings
        self.namespace = namespace
        self.cache = cache or JsonCache()
        self.client = client or httpx.Client(timeout=timeout, headers={"User-Agent": "ai-truth-engine/0.3"})
        self.offline = settings.force_offline if offline is None else offline
        self.limiter = RateLimiter(per_second)
        self.errors: list[str] = []
        self.network_calls = 0

    def get(self, url: str, params: dict | None = None, attempts: int = 3) -> str | None:
        hit = self.cache.get(self.namespace, url, params)
        if hit is not None:
            return hit
        if self.offline:
            return None
        for i in range(attempts):
            self.limiter.wait()
            try:
                self.network_calls += 1
                r = self.client.get(url, params=params)
            except Exception as e:  # network down, DNS, timeout
                self.errors.append(f"{self.namespace}: {type(e).__name__}: {e}")
                return None
            if r.status_code == 200:
                self.cache.put(self.namespace, url, params, r.text)
                return r.text
            if r.status_code in (429, 500, 502, 503, 504) and i < attempts - 1:
                time.sleep(1.0 * (2 ** i))
                continue
            self.errors.append(f"{self.namespace}: HTTP {r.status_code} for {url}")
            return None
        return None
