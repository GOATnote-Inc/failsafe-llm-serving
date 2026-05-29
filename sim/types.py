"""Core value types shared across the simulator.

Kept dependency-free (standard library only) so the whole project runs with a
bare Python interpreter — no numpy, no pip install. That portability is a
deliberate design choice: an interview reviewer should be able to clone and run
`make` with zero setup.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Phase(Enum):
    """Where a request is in the engine's two-phase lifecycle."""
    PREFILL = "prefill"   # processing the prompt, building the KV cache (compute-bound)
    DECODE = "decode"     # generating tokens one at a time (memory-bandwidth-bound)


class Outcome(Enum):
    """Terminal disposition of a request.

    The four SERVED_* outcomes are all *clinically safe*: the user gets either a
    grounded answer (primary / fallback / cache) or the ranked source documents
    (retrieval-only). SHED is an honest, fast rejection (HTTP 503 + Retry-After):
    not "good", but bounded and recoverable. TIMEOUT / DROPPED_EXPIRED are the
    *bad* outcomes — the slow-death failures a naive system produces under load.
    """
    SERVED_PRIMARY = "served_primary"        # full conductor->specialist ensemble answer
    SERVED_FALLBACK = "served_fallback"      # smaller/quantized model, still grounded
    SERVED_RETRIEVAL = "served_retrieval"    # ranked citations only, no synthesis (no gen GPU)
    SERVED_CACHE = "served_cache"            # exact prior answer for an identical prompt
    SHED = "shed"                            # fast 503 + Retry-After (admission control)
    TIMEOUT = "timeout"                      # exceeded deadline during/after service
    DROPPED_EXPIRED = "dropped_expired"      # dequeued past its deadline; never worth running

    @property
    def is_served_safe(self) -> bool:
        return self in (
            Outcome.SERVED_PRIMARY,
            Outcome.SERVED_FALLBACK,
            Outcome.SERVED_RETRIEVAL,
            Outcome.SERVED_CACHE,
        )

    @property
    def is_failure(self) -> bool:
        """Slow-death failures — the thing we are engineering to eliminate."""
        return self in (Outcome.TIMEOUT, Outcome.DROPPED_EXPIRED)


@dataclass
class Request:
    """A single clinician query flowing through the inference path.

    `shared_prefix_tokens` is the portion of the prompt that is identical across
    many queries (system prompt + retrieved literature for a given topic). It is
    the part that prefix caching can reuse — the dominant latency lever for a
    high-traffic medical-RAG workload.
    """
    rid: int
    prefix_id: int                 # which "topic"; drives cache affinity + sharing
    arrival_ms: float
    prompt_tokens: int             # total input length (shared_prefix + unique question)
    shared_prefix_tokens: int      # cacheable, reusable portion of the prompt
    output_tokens: int             # tokens to generate
    deadline_ms: float             # arrival + client patience budget; past this it's worthless
    attempt: int = 0               # 0 = first try; >0 = a retry (load amplification)

    # --- runtime state (mutated as it flows through the system) ---
    phase: Optional[Phase] = None
    assigned: Optional[int] = None       # replica index it was dispatched to
    enqueue_ms: Optional[float] = None
    dispatch_ms: Optional[float] = None
    prefill_total_ms: float = 0.0
    prefill_remaining_ms: float = 0.0
    tokens_done: float = 0.0
    ttft_ms: Optional[float] = None      # time to first token (queue wait + prefill)
    finish_ms: Optional[float] = None
    outcome: Optional[Outcome] = None
    served_tier: Optional[str] = None    # human-readable tier label for plots
    cache_hit: Optional[bool] = None     # was the prompt prefix already resident on the replica?

    @property
    def unique_tokens(self) -> int:
        return max(0, self.prompt_tokens - self.shared_prefix_tokens)
