"""Truss model.py for a PRIMARY specialist replica (illustrative).

For a pure TensorRT-LLM Engine-Builder deployment you often don't hand-write
predict() at all — the engine serves an OpenAI-compatible endpoint directly. This
file shows the Custom-Server shape (vLLM behind Truss) to make two fail-safe
points concrete:

  1. predict() streams tokens (SSE) so TTFT is decoupled from total generation —
     the clinician sees the first token fast even when the full answer is long.
  2. The replica exposes a cheap saturation signal (KV-cache utilization). The
     router reads this to route for cache affinity and to trip its circuit
     breaker BEFORE the engine falls onto the recompute-preemption cliff.

Baseten's standard dashboard exposes "Time to first byte", "Concurrent requests"
and "GPU memory" but NOT a first-party KV-utilization metric, so surfacing it
here (and scraping it via the Prometheus /metrics export) is deliberate.
"""
from __future__ import annotations

from typing import AsyncGenerator, Dict


class Model:
    def __init__(self, **kwargs):
        self._engine = None
        self._config = kwargs.get("config", {})

    def load(self):
        """Called once at startup. Heavy weight-load happens here; Baseten's
        network accelerator / BDN make this fast, but min_replica>=2 keeps it off
        the request path entirely."""
        # from vllm import AsyncLLMEngine, AsyncEngineArgs
        # self._engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        #     model="/packages/specialist-cardiology",
        #     enable_prefix_caching=True,      # reuse shared system prompt + retrieved-doc KV
        #     gpu_memory_utilization=0.85,     # headroom before preemption (mirrors config.yaml)
        #     max_num_seqs=16,
        # ))
        self._engine = _FakeEngine()            # placeholder so this file imports cleanly

    def kv_cache_utilization(self) -> float:
        """0..1 fraction of KV blocks in use. The single most useful backpressure
        signal — see DESIGN.md. Exported for the router + Prometheus scrape."""
        return self._engine.kv_utilization()

    async def predict(self, model_input: Dict) -> AsyncGenerator[bytes, None]:
        """Stream tokens as Server-Sent Events. The router enforces admission;
        by the time we get here we are inside predict_concurrency, so the engine
        stays in its linear regime."""
        prompt = model_input["prompt"]
        max_tokens = model_input.get("max_tokens", 512)
        async for token in self._engine.generate_stream(prompt, max_tokens):
            yield f"data: {token}\n\n".encode()
        yield b"data: [DONE]\n\n"


class _FakeEngine:
    """Stand-in so the module imports without a GPU. Replace with a real engine."""
    def kv_utilization(self) -> float:
        return 0.0

    async def generate_stream(self, prompt: str, max_tokens: int):
        for i in range(min(8, max_tokens)):
            yield f"token_{i}"
