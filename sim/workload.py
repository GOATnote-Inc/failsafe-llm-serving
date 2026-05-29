"""Deterministic, seeded workload generator.

Produces a medical-RAG-shaped arrival stream:
  * Zipf-popular topics  -> a few clinical questions dominate traffic, so prompt
    prefixes are heavily reusable (prefix caching is a real lever, not a toy).
  * Long shared prefixes -> system prompt + retrieved NEJM/JAMA passages.
  * A burst              -> a drug-recall headline spikes ONE brand-new topic,
    whose prefix is COLD on every replica (worst case for KV + prefill).

All randomness flows through a single seeded `random.Random`, so a run is
byte-for-byte reproducible — which is what lets the tests assert "naive
collapses, fail-safe does not."
"""
from __future__ import annotations

import random
from typing import List

from .config import SimConfig
from .types import Request


class WorkloadGenerator:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        # Precompute a Zipf popularity distribution over background topics.
        weights = [1.0 / (i + 1) ** cfg.zipf_s for i in range(cfg.n_topics)]
        total = sum(weights)
        self._topic_cum = []
        acc = 0.0
        for w in weights:
            acc += w / total
            self._topic_cum.append(acc)
        # The spike rides on a fresh topic id that never appears in the baseline.
        self.spike_topic_id = cfg.n_topics  # one past the background range -> cold

    def _pick_topic(self) -> int:
        u = self.rng.random()
        for i, c in enumerate(self._topic_cum):
            if u <= c:
                return i
        return self.cfg.n_topics - 1

    def _sizes(self):
        c = self.cfg
        shared = max(1_000, int(self.rng.gauss(c.shared_prefix_mean, c.shared_prefix_jitter)))
        unique = max(64, int(self.rng.gauss(c.unique_mean, c.unique_jitter)))
        out = max(64, int(self.rng.gauss(c.output_mean, c.output_jitter)))
        return shared, unique, out

    def _make(self, rid: int, t_ms: float, topic: int) -> Request:
        shared, unique, out = self._sizes()
        return Request(
            rid=rid,
            prefix_id=topic,
            arrival_ms=t_ms,
            prompt_tokens=shared + unique,
            shared_prefix_tokens=shared,
            output_tokens=out,
            deadline_ms=t_ms + self.cfg.deadline_budget_ms,
        )

    def generate(self) -> List[Request]:
        """Return all *first-attempt* requests, sorted by arrival time.

        Retries are not pre-generated — they are injected dynamically by the
        router/client during the run, because whether a retry happens depends on
        the policy under test (that feedback loop is the whole point).
        """
        c = self.cfg
        reqs: List[Request] = []
        rid = 0

        # --- baseline Poisson stream over the full horizon ---
        t = 0.0
        rate_per_ms = c.lambda_base_rps / 1000.0
        while t < c.horizon_ms:
            t += self.rng.expovariate(rate_per_ms)
            if t >= c.horizon_ms:
                break
            reqs.append(self._make(rid, t, self._pick_topic()))
            rid += 1

        # --- spike stream: extra load on the cold recall topic ---
        spike_rate = (c.lambda_base_rps * (c.spike_multiplier - 1.0)) / 1000.0
        if spike_rate > 0:                       # multiplier <= 1 means "no burst"
            t = c.spike_start_ms
            while t < c.spike_end_ms:
                t += self.rng.expovariate(spike_rate)
                if t >= c.spike_end_ms:
                    break
                topic = self.spike_topic_id if c.spike_is_cold else self._pick_topic()
                reqs.append(self._make(rid, t, topic))
                rid += 1

        reqs.sort(key=lambda r: r.arrival_ms)
        return reqs
