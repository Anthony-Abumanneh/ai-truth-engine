"""Monitoring and abuse protection for the API (step 11).

    Metrics      requests and latency per endpoint, verdict mix, abstention rate,
                 mean confidence, errors; served at /metrics (Prometheus text
                 format) and /api/stats (JSON, plus accuracy from user feedback).
    Drift        the verdict mix and abstention rate are compared with the SciFact
                 dev benchmark; a large gap means the inputs (or the models) changed.
    Logs         one JSON line per request on stdout (path, status, latency, client hash).
    Rate limit   per-client token bucket: TE_RATE_LIMIT requests per minute on the
                 expensive endpoints (0 = off, the local default).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from collections import defaultdict

# Verdict mix and abstention of the full system on SciFact dev (data/bench/sys_ours_dev.jsonl,
# 2026-10-27, n=300): the drift baseline. "support" groups supported / partially / inferred.
BENCHMARK_MIX = {"support": 0.260, "contradicted": 0.103, "unverified": 0.637}
BENCHMARK_ABSTAIN = 0.637
GROUP = {"supported": "support", "partially_supported": "support", "inferred": "support",
         "contradicted": "contradicted", "unverified": "unverified", "speculative": "unverified"}
LATENCY_BUCKETS = (0.5, 1, 2, 5, 10, 30, 60, 120)
EXPENSIVE = ("/api/audit", "/api/audit/stream", "/api/ask", "/api/watch/recheck")


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.started = time.time()
        self.requests: dict[tuple[str, int], int] = defaultdict(int)
        self.latency_sum: dict[str, float] = defaultdict(float)
        self.latency_buckets: dict[tuple[str, float], int] = defaultdict(int)
        self.verdicts: dict[str, int] = defaultdict(int)
        self.decisions: dict[str, int] = defaultdict(int)
        self.confidence_sum = 0.0
        self.claims = 0
        self.live_runs = 0
        self.errors = 0
        self.rate_limited = 0

    def request(self, path: str, status: int, seconds: float) -> None:
        with self._lock:
            self.requests[(path, status)] += 1
            self.latency_sum[path] += seconds
            for b in LATENCY_BUCKETS:
                if seconds <= b:
                    self.latency_buckets[(path, b)] += 1
            if status >= 500:
                self.errors += 1

    def result(self, view: dict, live: bool) -> None:
        with self._lock:
            self.live_runs += int(live)
            for c in view.get("claims", []):
                self.claims += 1
                self.verdicts[c["status"]] += 1
                self.decisions[c["decision"] or "none"] += 1
                self.confidence_sum += c["confidence"]

    def stats(self) -> dict:
        with self._lock:
            n = max(1, self.claims)
            mix = {k: round(v / n, 3) for k, v in self.verdicts.items()}
            grouped: dict[str, float] = defaultdict(float)
            for k, v in self.verdicts.items():
                grouped[GROUP.get(k, k)] += v / n
            abstain = round(self.decisions.get("abstain", 0) / n, 3)
            drift = {k: round(grouped.get(k, 0.0) - v, 3) for k, v in BENCHMARK_MIX.items()} if self.claims else {}
            return {"uptime_s": round(time.time() - self.started), "claims_checked": self.claims,
                    "requests": sum(self.requests.values()), "errors_5xx": self.errors,
                    "rate_limited": self.rate_limited, "live_runs": self.live_runs,
                    "verdict_mix": mix, "abstention_rate": abstain,
                    "mean_confidence": round(self.confidence_sum / n, 3) if self.claims else None,
                    "drift_vs_benchmark": {"verdict_mix": drift,
                                           "abstention": round(abstain - BENCHMARK_ABSTAIN, 3) if self.claims else None,
                                           "alert": bool(self.claims >= 50 and max(abs(x) for x in drift.values()) > 0.25)}}

    def prometheus(self) -> str:
        lines = ["# TYPE truth_engine_requests_total counter"]
        with self._lock:
            for (path, status), k in sorted(self.requests.items()):
                lines.append(f'truth_engine_requests_total{{path="{path}",status="{status}"}} {k}')
            lines.append("# TYPE truth_engine_request_seconds histogram")
            for path, total in sorted(self.latency_sum.items()):
                count = sum(k for (p, _), k in self.requests.items() if p == path)
                for b in LATENCY_BUCKETS:
                    lines.append(f'truth_engine_request_seconds_bucket{{path="{path}",le="{b}"}} '
                                 f'{self.latency_buckets[(path, b)]}')
                lines.append(f'truth_engine_request_seconds_bucket{{path="{path}",le="+Inf"}} {count}')
                lines.append(f'truth_engine_request_seconds_sum{{path="{path}"}} {total:.3f}')
                lines.append(f'truth_engine_request_seconds_count{{path="{path}"}} {count}')
            lines.append("# TYPE truth_engine_verdicts_total counter")
            for v, k in sorted(self.verdicts.items()):
                lines.append(f'truth_engine_verdicts_total{{verdict="{v}"}} {k}')
            lines.append("# TYPE truth_engine_decisions_total counter")
            for d, k in sorted(self.decisions.items()):
                lines.append(f'truth_engine_decisions_total{{decision="{d}"}} {k}')
            lines += ["# TYPE truth_engine_rate_limited_total counter", f"truth_engine_rate_limited_total {self.rate_limited}",
                      "# TYPE truth_engine_live_runs_total counter", f"truth_engine_live_runs_total {self.live_runs}"]
        return "\n".join(lines) + "\n"


class RateLimiter:
    """Token bucket per client: `per_minute` requests, refilling continuously."""

    def __init__(self, per_minute: int):
        self.rate = per_minute / 60.0
        self.capacity = float(per_minute)
        self._lock = threading.Lock()
        self.buckets: dict[str, tuple[float, float]] = {}

    def allow(self, client: str) -> bool:
        if self.rate <= 0:
            return True
        now = time.monotonic()
        with self._lock:
            tokens, last = self.buckets.get(client, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            ok = tokens >= 1
            self.buckets[client] = (tokens - 1 if ok else tokens, now)
            if len(self.buckets) > 10_000:                 # bound memory under many clients
                self.buckets = dict(list(self.buckets.items())[-5_000:])
            return ok


def client_id(ip: str | None) -> str:
    """Hashed client address for logs (no raw IPs stored)."""
    return hashlib.sha256((ip or "?").encode()).hexdigest()[:12]


def log_request(path: str, status: int, seconds: float, client: str) -> None:
    if os.getenv("TE_ACCESS_LOG", "1") != "0":
        print(json.dumps({"ts": round(time.time(), 3), "path": path, "status": status,
                          "latency_s": round(seconds, 3), "client": client}), file=sys.stdout, flush=True)
