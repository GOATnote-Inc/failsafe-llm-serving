"""Fixed-step discrete-time simulation loop.

Closes the loop that makes saturation *metastable*: a request that misses its
deadline is retried by the client (capped, with backoff+jitter). Under the naive
policy those retries pile onto an already-overloaded fleet and sustain the
collapse after the burst is gone. Under the fail-safe policy almost nothing times
out (it is shed or laddered to a degraded-but-safe tier instead), so there is
nothing to retry and the system snaps back the instant the burst ends.

Both policies use the *same* client retry behavior — the only difference is the
server. That keeps the comparison honest.
"""
from __future__ import annotations

import heapq
import random
from typing import List

from .config import SimConfig
from .metrics import MetricsRecorder
from .model_node import ModelNode
from .router import Router
from .types import Outcome, Phase, Request
from .workload import WorkloadGenerator


class Simulation:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.nodes = [ModelNode(i, cfg) for i in range(cfg.n_replicas)]
        # Pre-warm each replica with the hot prefixes that affinity-route to it,
        # mirroring a real fleet that keeps popular contexts resident.
        for i, node in enumerate(self.nodes):
            hot = list(range(i, cfg.n_topics, cfg.n_replicas))[: cfg.prefix_cache_slots]
            node.warm(hot)
        self.router = Router(cfg)
        self.metrics = MetricsRecorder(cfg)
        self.rng = random.Random(cfg.seed + 2)

    def _maybe_retry(self, req: Request, now: float, heap: list, counter: List[int]):
        """A real client retries a timed-out request — capped, with jittered
        backoff. This is the load amplifier; fail-safe avoids it by rarely
        producing a timeout in the first place."""
        if req.attempt + 1 >= self.cfg.naive_max_attempts:
            return
        backoff = self.cfg.naive_retry_backoff_ms * (1.0 + 0.5 * self.rng.random())
        retry = Request(
            rid=req.rid,
            prefix_id=req.prefix_id,
            arrival_ms=now + backoff,
            prompt_tokens=req.prompt_tokens,
            shared_prefix_tokens=req.shared_prefix_tokens,
            output_tokens=req.output_tokens,
            deadline_ms=now + backoff + self.cfg.deadline_budget_ms,
            attempt=req.attempt + 1,
        )
        counter[0] += 1
        heapq.heappush(heap, (retry.arrival_ms, counter[0], retry))

    def _handle_node_timeouts(self, now: float, heap: list, counter: List[int]):
        for node in self.nodes:
            still_running = []
            for r in node.running:
                if now >= r.deadline_ms:
                    r.outcome = Outcome.TIMEOUT
                    r.finish_ms = now
                    self.metrics.on_terminal(r, now)
                    self._maybe_retry(r, now, heap, counter)
                else:
                    still_running.append(r)
            node.running = still_running

    def run(self) -> MetricsRecorder:
        cfg = self.cfg
        reqs = WorkloadGenerator(cfg).generate()
        heap = [(r.arrival_ms, i, r) for i, r in enumerate(reqs)]
        heapq.heapify(heap)
        counter = [len(reqs)]  # mutable tiebreaker for heap pushes

        now = 0.0
        while now < cfg.horizon_ms:
            # 1) inject arrivals (and any retries) due by now
            while heap and heap[0][0] <= now:
                _, _, r = heapq.heappop(heap)
                self.router.submit(r, now, self.metrics)

            # 2) dispatch: admission control, routing, fallback ladder
            terminal, aux_admitted = self.router.step(now, self.nodes, self.metrics)
            for r in terminal:                       # SHED / DROPPED_EXPIRED
                self.metrics.on_terminal(r, now)
                if r.outcome == Outcome.DROPPED_EXPIRED:
                    self._maybe_retry(r, now, heap, counter)
            for r in aux_admitted:                   # entered fallback/retrieval
                self.metrics.on_first_token(r, now)
            self.router.note_first_tokens(aux_admitted, now)

            # 3) advance every primary replica one engine step
            preempted: List[Request] = []
            for node in self.nodes:
                fts, completed, pre = node.tick(cfg.tick_ms, now)
                if fts:
                    for r in fts:
                        self.metrics.on_first_token(r, now)
                    self.router.note_first_tokens(fts, now)
                for r in completed:
                    r.outcome = Outcome.SERVED_PRIMARY
                    r.served_tier = "primary"
                    self.metrics.on_terminal(r, now)
                preempted.extend(pre)
            if preempted:
                self.metrics.on_preemption(now, len(preempted))
                self.router.requeue_preempted(preempted, now)

            # 4) aux-tier completions (fallback / retrieval)
            for r in self.router.aux_step(now):
                self.metrics.on_terminal(r, now)

            # 5) timeouts on in-service requests -> retry storm feedback
            self._handle_node_timeouts(now, heap, counter)

            # 6) sample system state for the time series
            kv_max = max(n.util() for n in self.nodes)
            self.metrics.sample_system(now, kv_max, self.router.queue_depth(),
                                       self.router.per_replica_cap)
            now += cfg.tick_ms

        # Drain at horizon end. Distinguish genuine failures from censoring:
        # a request that already got its first token and is decoding normally
        # would have completed — counting it as a failure is a measurement
        # artifact, not a system failure. One still in prefill never started.
        for r in list(self.router.queue):
            r.outcome = Outcome.DROPPED_EXPIRED
            r.finish_ms = now
            self.metrics.on_terminal(r, now)
        for node in self.nodes:
            for r in node.running:
                if r.phase == Phase.DECODE and r.ttft_ms is not None:
                    r.outcome = Outcome.SERVED_PRIMARY      # progressing fine; censored by horizon
                    r.served_tier = "primary"
                else:
                    r.outcome = Outcome.DROPPED_EXPIRED      # never got off the ground
                r.finish_ms = now
                self.metrics.on_terminal(r, now)
        return self.metrics
