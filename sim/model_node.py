"""A single GPU replica running a continuous-batching LLM engine.

This is where the saturation physics live. It reproduces the four behaviors that
make a real serving node fall off a cliff, without simulating a real engine:

  1. Decode slows as the batch grows. Inter-token latency = base * (1 + B/softcap).
     This is the throughput<->latency feedback: a bigger batch is more efficient
     per-GPU but slower per-token, so chasing throughput past the knee makes
     *every* request slower, which needs an even bigger batch to keep up -> runaway.

  2. KV cache is the real concurrency limit. Each sequence holds (unique prompt +
     generated) tokens; the shared prefix is counted once per distinct prefix on
     the node (paged KV block sharing). When utilization crosses `preempt_util`,
     the engine preempts.

  3. The "swapping cliff": past `preempt_util`, a multiplier blows up step time to
     model recompute thrash + fragmentation. (vLLM V1 default is RECOMPUTE
     preemption: a preempted sequence loses its generated KV and is re-prefilled.)

  4. Prefix caching: if the node already holds a prompt's shared prefix, that
     prefix is skipped at prefill -> dramatically lower TTFT and KV footprint.
     A cold prefix (a fresh drug-recall topic) pays the full prefill on first use.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import List, Tuple

from .config import SimConfig
from .types import Phase, Request


class ModelNode:
    def __init__(self, idx: int, cfg: SimConfig):
        self.idx = idx
        self.cfg = cfg
        self.running: List[Request] = []
        self.prefix_cache: "OrderedDict[int, bool]" = OrderedDict()  # LRU of hot prefixes

    # ---- introspection used by the router ----
    @property
    def in_flight(self) -> int:
        return len(self.running)

    def decode_count(self) -> int:
        return sum(1 for r in self.running if r.phase == Phase.DECODE)

    def prefill_count(self) -> int:
        return sum(1 for r in self.running if r.phase == Phase.PREFILL)

    def has_prefix(self, prefix_id: int) -> bool:
        return prefix_id in self.prefix_cache

    def warm(self, prefix_ids):
        """Pre-load hot prefixes. A real deployment keeps these warm (min_replica
        >= 2 plus prefix-cache offload), so the system prompt + popular retrieved
        contexts don't pay a cold prefill on the first request after a deploy."""
        for pid in prefix_ids:
            self._cache_prefix(pid)

    def kv_used(self) -> float:
        """KV tokens resident. Shared prefix counted once per distinct prefix."""
        own = 0.0
        shared = {}
        for r in self.running:
            own += r.unique_tokens + r.tokens_done
            if r.shared_prefix_tokens > shared.get(r.prefix_id, 0):
                shared[r.prefix_id] = r.shared_prefix_tokens
        return own + sum(shared.values())

    def util(self) -> float:
        return self.kv_used() / self.cfg.kv_capacity_tokens

    def can_accept(self) -> bool:
        return len(self.running) < self.cfg.engine_max_running

    # ---- admission ----
    def admit(self, req: Request, now: float):
        cfg = self.cfg
        req.cache_hit = self.has_prefix(req.prefix_id)
        uncached = req.unique_tokens if req.cache_hit else req.prompt_tokens
        req.phase = Phase.PREFILL
        req.assigned = self.idx
        req.dispatch_ms = now
        req.tokens_done = 0.0
        req.prefill_total_ms = cfg.prefill_ms_per_ktok * uncached / 1000.0
        req.prefill_remaining_ms = req.prefill_total_ms
        self.running.append(req)

    def _cache_prefix(self, prefix_id: int):
        self.prefix_cache[prefix_id] = True
        self.prefix_cache.move_to_end(prefix_id)
        while len(self.prefix_cache) > self.cfg.prefix_cache_slots:
            self.prefix_cache.popitem(last=False)

    def _cliff(self, util: float) -> float:
        cfg = self.cfg
        if util <= cfg.preempt_util:
            return 1.0
        over = (min(util, 1.0) - cfg.preempt_util) / max(1e-6, 1.0 - cfg.preempt_util)
        return 1.0 + cfg.cliff_steepness * over

    # ---- one engine step (dt ms) ----
    def tick(self, dt: float, now: float) -> Tuple[List[Request], List[Request], List[Request]]:
        first_tokens: List[Request] = []
        completed: List[Request] = []
        preempted: List[Request] = []

        # 1) Preempt to fit KV. RECOMPUTE semantics: victim loses generated tokens
        #    and must be re-prefilled from scratch (handed back to the router).
        #    `used` is tracked incrementally so a deep preemption storm stays O(n).
        cap = self.cfg.kv_capacity_tokens
        used = self.kv_used()
        while used > cap and len(self.running) > 1:
            victim = None  # prefer newest decode seq, else newest prefill seq
            for r in reversed(self.running):
                if r.phase == Phase.DECODE:
                    victim = r
                    break
            if victim is None:
                victim = self.running[-1]
            self.running.remove(victim)
            used -= victim.unique_tokens + victim.tokens_done   # approx (ignores shared-prefix release)
            victim.tokens_done = 0.0
            victim.phase = None
            victim.assigned = None
            preempted.append(victim)

        B = len(self.running)
        util = self.util()
        cliff = self._cliff(util)
        decode_step_ms = self.cfg.decode_base_ms * (1.0 + B / self.cfg.decode_softcap) * cliff
        prefill_divisor = (1.0 + self.prefill_count() / self.cfg.prefill_softcap) * cliff

        for r in list(self.running):
            if r.phase == Phase.PREFILL:
                r.prefill_remaining_ms -= dt / prefill_divisor
                if r.prefill_remaining_ms <= 0:
                    r.phase = Phase.DECODE
                    r.ttft_ms = now - r.arrival_ms        # queue wait + prefill
                    self._cache_prefix(r.prefix_id)        # shared prefix now hot here
                    first_tokens.append(r)
            else:  # DECODE
                r.tokens_done += dt / decode_step_ms
                if r.tokens_done >= r.output_tokens:
                    r.finish_ms = now
                    self.running.remove(r)
                    completed.append(r)

        return first_tokens, completed, preempted
