"""Metric collection. Goodput-first, because throughput lies under overload.

The number that matters is **goodput**: the fraction of requests that get a
*safe* outcome *within the responsiveness budget*. A saturated server can show
100% throughput (everything eventually returns) while goodput is zero (all of it
too slow to be useful). The collapse-vs-failsafe chart is fundamentally a
goodput-over-time chart.

Standard library only; percentiles computed by hand.
"""
from __future__ import annotations

import math
from typing import Dict, List

from .config import SimConfig
from .types import Outcome, Request


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


class MetricsRecorder:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.nbins = int(math.ceil(cfg.horizon_ms / cfg.bin_ms))
        z = lambda: [0.0] * self.nbins
        self.offered = z()
        self.shed = z()
        self.timeouts = z()
        self.dropped = z()
        self.served_primary = z()
        self.served_fallback = z()
        self.served_retrieval = z()
        self.served_cache = z()
        self.preemptions = z()
        self.ttft_by_bin: List[List[float]] = [[] for _ in range(self.nbins)]
        # system samples (averaged per bin)
        self._kv_max_acc = z(); self._kv_max_n = z()
        self._queue_acc = z(); self._queue_n = z()
        self._effcap_acc = z(); self._effcap_n = z()
        self.terminal: List[Request] = []
        self.first_token_within_slo = 0
        self.n_retries = 0

    def _bin(self, t_ms: float) -> int:
        return min(self.nbins - 1, max(0, int(t_ms / self.cfg.bin_ms)))

    def _rps(self, count: float) -> float:
        return count / (self.cfg.bin_ms / 1000.0)

    # ---- event hooks called by the engine ----
    def on_arrival(self, req: Request):
        self.offered[self._bin(req.arrival_ms)] += 1
        if req.attempt > 0:
            self.n_retries += 1

    def on_first_token(self, req: Request, now: float):
        self.ttft_by_bin[self._bin(now)].append(req.ttft_ms)
        if req.ttft_ms is not None and req.ttft_ms <= self.cfg.slo_ttft_ms:
            self.first_token_within_slo += 1

    def on_terminal(self, req: Request, now: float):
        self.terminal.append(req)
        b = self._bin(now)
        o = req.outcome
        if o == Outcome.SHED:
            self.shed[b] += 1
        elif o == Outcome.TIMEOUT:
            self.timeouts[b] += 1
        elif o == Outcome.DROPPED_EXPIRED:
            self.dropped[b] += 1
        elif o == Outcome.SERVED_PRIMARY:
            self.served_primary[b] += 1
        elif o == Outcome.SERVED_FALLBACK:
            self.served_fallback[b] += 1
        elif o == Outcome.SERVED_RETRIEVAL:
            self.served_retrieval[b] += 1
        elif o == Outcome.SERVED_CACHE:
            self.served_cache[b] += 1

    def on_preemption(self, now: float, n: int = 1):
        self.preemptions[self._bin(now)] += n

    def sample_system(self, now: float, kv_util_max: float, queue_depth: float, eff_cap: float):
        b = self._bin(now)
        self._kv_max_acc[b] += kv_util_max; self._kv_max_n[b] += 1
        self._queue_acc[b] += queue_depth; self._queue_n[b] += 1
        self._effcap_acc[b] += eff_cap; self._effcap_n[b] += 1

    # ---- summaries ----
    def _avg(self, acc, n):
        return [(acc[i] / n[i]) if n[i] else 0.0 for i in range(self.nbins)]

    def series(self) -> Dict[str, List[float]]:
        t_s = [i * self.cfg.bin_ms / 1000.0 for i in range(self.nbins)]
        safe = [
            self.served_primary[i] + self.served_fallback[i]
            + self.served_retrieval[i] + self.served_cache[i]
            for i in range(self.nbins)
        ]
        fail = [self.timeouts[i] + self.dropped[i] for i in range(self.nbins)]
        return {
            "t_s": t_s,
            "offered_rps": [self._rps(x) for x in self.offered],
            "goodput_safe_rps": [self._rps(x) for x in safe],
            "primary_rps": [self._rps(x) for x in self.served_primary],
            "fallback_rps": [self._rps(x) for x in self.served_fallback],
            "retrieval_rps": [self._rps(x) for x in self.served_retrieval],
            "cache_rps": [self._rps(x) for x in self.served_cache],
            "shed_rps": [self._rps(x) for x in self.shed],
            "failure_rps": [self._rps(x) for x in fail],
            "ttft_p95_ms": [percentile(self.ttft_by_bin[i], 95) for i in range(self.nbins)],
            "kv_util_max": self._avg(self._kv_max_acc, self._kv_max_n),
            "queue_depth": self._avg(self._queue_acc, self._queue_n),
            "eff_cap": self._avg(self._effcap_acc, self._effcap_n),
            "preemptions": list(self.preemptions),
        }

    def window_summary(self) -> Dict[str, Dict[str, float]]:
        """Goodput / latency split into calm | spike | recovery windows, keyed by
        each request's arrival time. The recovery window is the punchline: a
        fail-safe system snaps back to calm-window numbers the moment the burst
        ends; a collapsed one stays underwater."""
        cfg = self.cfg

        def bucket(r: Request) -> str:
            if r.arrival_ms < cfg.spike_start_ms:
                return "calm"
            if r.arrival_ms < cfg.spike_end_ms:
                return "spike"
            return "recovery"

        out: Dict[str, Dict[str, float]] = {}
        for w in ("calm", "spike", "recovery"):
            rs = [r for r in self.terminal if bucket(r) == w]
            n = len(rs)
            good = [r for r in rs if r.outcome.is_served_safe
                    and r.ttft_ms is not None and r.ttft_ms <= cfg.slo_ttft_ms]
            safe = [r for r in rs if r.outcome.is_served_safe]
            primary = [r for r in rs if r.outcome == Outcome.SERVED_PRIMARY]
            ttfts = [r.ttft_ms for r in rs if r.ttft_ms is not None]
            out[w] = {
                "n": float(n),
                "goodput_pct": 100.0 * len(good) / n if n else 0.0,
                "safe_pct": 100.0 * len(safe) / n if n else 0.0,
                "primary_pct": 100.0 * len(primary) / n if n else 0.0,
                "ttft_p95_ms": percentile(ttfts, 95),
            }
        return out

    def summary(self) -> Dict[str, float]:
        n = len(self.terminal)
        served = [r for r in self.terminal if r.outcome.is_served_safe]
        primary = [r for r in self.terminal if r.outcome == Outcome.SERVED_PRIMARY]
        failures = [r for r in self.terminal if r.outcome.is_failure]
        shed = [r for r in self.terminal if r.outcome == Outcome.SHED]
        ttfts = [r.ttft_ms for r in self.terminal if r.ttft_ms is not None]
        good = [r for r in served if (r.ttft_ms is not None and r.ttft_ms <= self.cfg.slo_ttft_ms)]
        hits = [r for r in self.terminal if r.cache_hit is True]
        seen = [r for r in self.terminal if r.cache_hit is not None]
        # post-spike recovery: goodput in the last 20 s vs the calm pre-spike window
        return {
            "total_requests": float(n),
            "served_safe": float(len(served)),
            "served_safe_pct": 100.0 * len(served) / n if n else 0.0,
            "primary_pct": 100.0 * len(primary) / n if n else 0.0,
            "goodput_pct": 100.0 * len(good) / n if n else 0.0,   # safe AND within 160 ms
            "shed_pct": 100.0 * len(shed) / n if n else 0.0,
            "failure_pct": 100.0 * len(failures) / n if n else 0.0,
            "prefix_cache_hit_pct": 100.0 * len(hits) / len(seen) if seen else 0.0,
            "retries": float(self.n_retries),
            "ttft_p50_ms": percentile(ttfts, 50),
            "ttft_p95_ms": percentile(ttfts, 95),
            "ttft_p99_ms": percentile(ttfts, 99),
            "preemptions": float(sum(self.preemptions)),
        }
