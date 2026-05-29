"""Baseten Chains skeleton of the fail-safe query path (illustrative).

This is the deployable shape of what ../sim/router.py models. The Conductor is
the @mark_entrypoint Chainlet; it does admission control + the fallback ladder in
plain async control flow (Chains has no central orchestrator — Chainlets call
each other directly, which is what keeps per-hop latency low). Each downstream
tier is its own Chainlet/Stub that scales independently.

SDK surface verified against docs.baseten.co/reference/sdk/chains (2026-05-28):
  chains.ChainletBase, @chains.mark_entrypoint, chains.depends, chains.StubBase,
  chains.RemoteConfig, chains.Compute(predict_concurrency=...),
  chains.RPCOptions(retries=, timeout_sec=).  Per-Chainlet min/max replica counts
  are set in the Baseten UI / autoscaling API (not in the SDK) — see config.yaml.
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional

import truss_chains as chains
import pydantic


class Query(pydantic.BaseModel):
    text: str
    specialty: str = "general"
    prefix_id: str = ""          # stable hash of (system prompt + retrieved-doc set)
    deadline_ms: float = 10_000  # client patience; past this the answer is worthless


class Answer(pydantic.BaseModel):
    text: str
    tier: str                    # primary | fallback | retrieval | shed
    citations: list[str] = []


# ----- downstream tiers (each its own independently-scaled Chainlet) -----

class PrimarySpecialist(chains.ChainletBase):
    """Full fine-tuned specialist (~30B). predict_concurrency is its latency knee."""
    remote_config = chains.RemoteConfig(
        compute=chains.Compute(gpu="H100", gpu_count=1, predict_concurrency=12),
    )

    async def run_remote(self, q: Query) -> Answer:
        ...  # call the served TRT-LLM/vLLM engine; stream tokens upstream
        return Answer(text="<full grounded answer>", tier="primary", citations=["NEJM..."])


class FallbackModel(chains.ChainletBase):
    """Smaller/quantized model: faster, cheaper, higher effective knee. Still grounded."""
    remote_config = chains.RemoteConfig(
        compute=chains.Compute(gpu="L4", gpu_count=1, predict_concurrency=48),
    )

    async def run_remote(self, q: Query) -> Answer:
        return Answer(text="<concise grounded answer>", tier="fallback", citations=["JAMA..."])


class RetrievalOnly(chains.ChainletBase):
    """The clinically-safe FLOOR: ranked peer-reviewed sources, no generation GPU.
    Cheap and horizontally scalable (CPU + vector store + Baseten BEI embeddings)."""
    remote_config = chains.RemoteConfig(compute=chains.Compute(cpu_count=4, memory="16Gi"))

    async def run_remote(self, q: Query) -> Answer:
        return Answer(text="I can't fully synthesize right now — here are the most relevant "
                           "peer-reviewed sources for your question.",
                      tier="retrieval", citations=["...ranked citations..."])


# ----- the conductor: admission control + KV-affinity routing + fallback ladder -----

@chains.mark_entrypoint
class Conductor(chains.ChainletBase):
    def __init__(
        self,
        primary: PrimarySpecialist = chains.depends(PrimarySpecialist),
        fallback: FallbackModel = chains.depends(FallbackModel),
        retrieval: RetrievalOnly = chains.depends(RetrievalOnly),
    ):
        self._primary = primary
        self._fallback = fallback
        self._retrieval = retrieval
        # Global admission gate. Sized to the fleet's aggregate knee; when it is
        # exhausted we ladder out fast instead of growing an unbounded queue.
        self._gate = asyncio.Semaphore(48)         # n_replicas * predict_concurrency
        # Per-tier RPC discipline: tight timeout, NO blind retries onto a hot tier.
        self._primary_rpc = chains.RPCOptions(timeout_sec=2.0, retries=0)
        self._fallback_rpc = chains.RPCOptions(timeout_sec=1.5, retries=0)

    async def run_remote(self, q: Query) -> Answer:
        # (1) Deadline-aware admission: never start work that can't finish in time.
        if _now_ms() > q.deadline_ms:
            return Answer(text="System busy — please retry.", tier="shed")

        # (2) Try to enter the primary gate without blocking forever. If we can't
        #     grab a slot quickly, the fleet is saturated -> go straight to the
        #     ladder rather than queueing (this is the load-shed decision).
        if self._gate.locked() and self._gate._value == 0:   # illustrative fast-path check
            return await self._degrade(q)

        async with self._gate:
            try:
                # Prefix-affinity is realized by Baseten's KV-aware router; we
                # cooperate by passing a stable prefix_id as the routing key so
                # same-context queries land on the replica that already holds them.
                return await asyncio.wait_for(
                    self._primary.run_remote(q), timeout=self._primary_rpc.timeout_sec)
            except (asyncio.TimeoutError, Exception):
                return await self._degrade(q)

    async def _degrade(self, q: Query) -> Answer:
        """primary unavailable -> fallback model -> retrieval-only -> 503."""
        try:
            return await asyncio.wait_for(
                self._fallback.run_remote(q), timeout=self._fallback_rpc.timeout_sec)
        except (asyncio.TimeoutError, Exception):
            pass
        try:
            return await self._retrieval.run_remote(q)        # the safe floor
        except Exception:
            return Answer(text="System busy — please retry shortly.", tier="shed")


def _now_ms() -> float:
    return time.monotonic() * 1000.0
