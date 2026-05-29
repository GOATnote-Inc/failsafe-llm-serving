"""The router / conductor — the fail-safe brain in front of the GPU fleet.

In Baseten terms this is the entrypoint Chainlet of a Chain: it does admission
control, KV-cache-affinity routing, circuit breaking, and the fallback ladder
before ever touching a primary GPU. Every mechanism here exists to convert the
one failure mode that matters — a saturating node dragging the whole system into
metastable collapse — into smooth, bounded, *recoverable* degradation.

Two policies share this code, selected by flags in SimConfig:
  * NAIVE   — no admission cap, unbounded queue, round-robin, client retries.
              This is the strawman that melts down.
  * FAILSAFE — concurrency capped at the latency knee (optionally AIMD-adaptive),
              bounded deadline-aware queue, prefix-affinity + power-of-two-choices,
              per-replica circuit breakers, and a primary->fallback->retrieval->503
              ladder so overflow degrades in quality, never in safety.
"""
from __future__ import annotations

import random
from collections import deque
from typing import List, Optional

from .config import SimConfig
from .metrics import percentile
from .model_node import ModelNode
from .types import Outcome, Request


class CircuitBreaker:
    """Per-replica breaker. Opens when a node stays saturated, so we stop piling
    work onto a node that cannot drain — give it room to recover instead."""
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.bad = 0.0
        self.open_until = -1.0

    def observe(self, node: ModelNode, now: float):
        if node.util() > 0.97:
            self.bad = min(20.0, self.bad + 1.0)
        else:
            self.bad = max(0.0, self.bad - 1.0)
        if self.bad >= 10.0 and now > self.open_until:
            self.open_until = now + self.cfg.breaker_cooldown_ms

    def is_open(self, now: float) -> bool:
        return now < self.open_until


class AuxTier:
    """A degraded service tier (the small-model fallback, or retrieval-only).

    Modeled simply: a finite number of concurrent slots and a fixed service time.
    The fallback pool is a smaller/quantized model (faster, cheaper, own replicas).
    The retrieval tier returns the ranked peer-reviewed sources with no generation
    GPU at all — the clinically-safe floor."""
    def __init__(self, name: str, slots: int, service_ms: float, ttft_ms: float, outcome: Outcome):
        self.name = name
        self.slots = slots
        self.service_ms = service_ms
        self.ttft_ms = ttft_ms
        self.outcome = outcome
        self.busy: List[tuple] = []  # (finish_ms, request)

    def free(self) -> bool:
        return len(self.busy) < self.slots

    def admit(self, req: Request, now: float) -> bool:
        if not self.free():
            return False
        req.ttft_ms = self.ttft_ms
        req.served_tier = self.name
        self.busy.append((now + self.service_ms, req))
        return True

    def tick(self, now: float) -> List[Request]:
        done = [r for (f, r) in self.busy if f <= now]
        if done:
            self.busy = [(f, r) for (f, r) in self.busy if f > now]
            for r in done:
                r.outcome = self.outcome
                r.finish_ms = now
        return done


class Router:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed + 1)
        self.queue: "deque[Request]" = deque()
        self.eff_cap = float(cfg.predict_concurrency)
        self.breakers = [CircuitBreaker(cfg) for _ in range(cfg.n_replicas)]
        # AIMD keys off QUEUE WAIT (dispatch - enqueue), not total TTFT: queue wait
        # is the true saturation signal. Prefill time is inherent to the prompt and
        # must not be mistaken for overload, or the controller throttles itself into
        # the very congestion it is meant to prevent.
        self._recent_qwait: "deque[tuple]" = deque()  # (now, queue_wait_ms)
        # Fallback pool: a smaller/quantized model — faster decode, own capacity.
        fb_slots = int(cfg.n_replicas * cfg.predict_concurrency * cfg.fallback_extra_capacity)
        self.fallback = AuxTier("fallback", fb_slots, service_ms=1_500.0,
                                ttft_ms=90.0, outcome=Outcome.SERVED_FALLBACK)
        # Retrieval-only floor: ranked citations, no generation GPU. Large but finite.
        self.retrieval = AuxTier("retrieval", slots=cfg.retrieval_slots, service_ms=cfg.retrieval_only_ms,
                                 ttft_ms=cfg.retrieval_only_ms, outcome=Outcome.SERVED_RETRIEVAL)

    # ---------- admission ----------
    def submit(self, req: Request, now: float, metrics) -> Optional[Request]:
        metrics.on_arrival(req)
        req.enqueue_ms = now
        self.queue.append(req)
        return None

    @property
    def per_replica_cap(self) -> int:
        if not self.cfg.use_admission_control:
            return self.cfg.engine_max_running   # naive: only the engine's own ceiling
        if self.cfg.use_adaptive_concurrency:
            return max(self.cfg.aimd_floor, int(round(self.eff_cap)))
        return self.cfg.predict_concurrency

    def _avg_service_ms(self) -> float:
        cap = max(1, self.per_replica_cap)
        itl = self.cfg.decode_base_ms * (1.0 + cap / self.cfg.decode_softcap)
        return self.cfg.output_mean * itl

    def _projected_wait_ms(self, qlen: int) -> float:
        cap = self.per_replica_cap
        drain_rps = (self.cfg.n_replicas * cap) / max(1.0, self._avg_service_ms() / 1000.0)
        return 1000.0 * qlen / max(1e-6, drain_rps)

    # ---------- adaptive concurrency (AIMD) ----------
    def note_first_tokens(self, reqs: List[Request], now: float):
        # Kept for API symmetry; AIMD is driven by queue wait recorded at admit.
        return

    def _record_qwait(self, req: Request, now: float):
        if req.enqueue_ms is not None:
            self._recent_qwait.append((now, now - req.enqueue_ms))

    def _update_adaptive(self, now: float):
        if not (self.cfg.use_admission_control and self.cfg.use_adaptive_concurrency):
            return
        while self._recent_qwait and self._recent_qwait[0][0] < now - 2_000.0:
            self._recent_qwait.popleft()
        if len(self._recent_qwait) < 5:
            return
        p90 = percentile([q for _, q in self._recent_qwait], 90)
        # Queue is backing up -> shed capacity (MD). Queue is empty -> probe up (AI).
        if p90 > 0.5 * self.cfg.slo_ttft_ms:
            self.eff_cap = max(self.cfg.aimd_floor, self.eff_cap * 0.92)
        elif p90 < 0.15 * self.cfg.slo_ttft_ms:
            self.eff_cap = min(float(self.cfg.predict_concurrency), self.eff_cap + 0.4)

    # ---------- routing ----------
    def _pick_replica(self, req: Request, nodes: List[ModelNode], now: float) -> Optional[int]:
        cap = self.per_replica_cap
        healthy = [i for i, n in enumerate(nodes)
                   if n.can_accept() and n.in_flight < cap
                   and not (self.cfg.use_circuit_breaker and self.breakers[i].is_open(now))]
        if not healthy:
            return None
        if self.cfg.use_prefix_affinity:
            pref = req.prefix_id % self.cfg.n_replicas
            if pref in healthy:
                least = min(nodes[i].in_flight for i in healthy)
                # honor affinity unless the preferred replica is a clear hotspot
                if nodes[pref].in_flight <= least + self.cfg.affinity_imbalance_slack:
                    return pref
            # power-of-two-choices among healthy replicas (load balance)
            a, b = self.rng.choice(healthy), self.rng.choice(healthy)
            return a if nodes[a].in_flight <= nodes[b].in_flight else b
        # naive: least-loaded (approximates round-robin under even load)
        return min(healthy, key=lambda i: nodes[i].in_flight)

    def _ladder(self, req: Request, now: float, admitted: List[Request]) -> Optional[Request]:
        """Overflow handling: primary -> fallback -> retrieval -> 503.

        Appends to `admitted` if a degraded tier accepted it (its first token is
        now known). Returns the request iff it became terminal here (SHED)."""
        if not self.cfg.use_fallback_ladder:
            return None  # naive: no ladder; request stays queued and eventually times out
        if self.fallback.admit(req, now):
            admitted.append(req)
            return None
        if self.retrieval.admit(req, now):
            admitted.append(req)
            return None
        # even the cheap tiers are saturated -> honest fast rejection
        req.outcome = Outcome.SHED
        req.served_tier = "shed"
        req.finish_ms = now
        return req

    def step(self, now: float, nodes: List[ModelNode], metrics):
        """Dispatch queued work. Returns (terminal, aux_admitted):
          terminal     — became terminal this step (SHED / DROPPED_EXPIRED)
          aux_admitted — just entered fallback/retrieval (first token known)."""
        aux_admitted: List[Request] = []
        self._update_adaptive(now)
        if self.cfg.use_circuit_breaker:
            for i, n in enumerate(nodes):
                self.breakers[i].observe(n, now)

        terminal: List[Request] = []
        # Drain the queue oldest-first; bounded work since slots are finite.
        requeue: "deque[Request]" = deque()
        while self.queue:
            req = self.queue.popleft()
            # Drop anything already past its deadline — never run worthless work.
            if now >= req.deadline_ms:
                req.outcome = Outcome.DROPPED_EXPIRED
                req.finish_ms = now
                terminal.append(req)
                continue
            idx = self._pick_replica(req, nodes, now)
            if idx is not None:
                self._record_qwait(req, now)
                nodes[idx].admit(req, now)
                continue
            # No primary slot. Fail-safe: if the queue can still drain inside the
            # responsiveness budget, hold it; otherwise ladder it out NOW.
            if self.cfg.use_admission_control:
                proj = self._projected_wait_ms(len(requeue) + 1)
                if proj <= self.cfg.shed_at_projected_ms and not self._all_breakers_open(now):
                    requeue.append(req)          # bounded wait — a real slot will free up
                else:
                    t = self._ladder(req, now, aux_admitted)   # primary->fallback->retrieval->503
                    if t is not None:
                        terminal.append(t)
            else:
                requeue.append(req)              # naive: unbounded queue, no ladder
        self.queue = requeue
        return terminal, aux_admitted

    def _all_breakers_open(self, now: float) -> bool:
        return self.cfg.use_circuit_breaker and all(b.is_open(now) for b in self.breakers)

    def requeue_preempted(self, reqs: List[Request], now: float):
        # Recompute-preempted sequences go back to the *front* — they were already
        # in service and are closest to their deadline.
        for r in reqs:
            r.enqueue_ms = now
            self.queue.appendleft(r)

    def aux_step(self, now: float) -> List[Request]:
        return self.fallback.tick(now) + self.retrieval.tick(now)

    def queue_depth(self) -> int:
        return len(self.queue)
