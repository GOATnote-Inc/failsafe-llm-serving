# `baseten/` — the deployable shape

These are the concrete Baseten artifacts the simulator argues for. They are
**illustrative skeletons** (no real weights/engine), but every field name and SDK
symbol is real and current as of 2026-05-28 — see [`../BASETEN_MAPPING.md`](../BASETEN_MAPPING.md) for the
mechanism→field table and citations.

| File | What it is | Load-bearing fail-safe content |
|---|---|---|
| [`config.yaml`](config.yaml) | Truss config for one **primary specialist** replica | `runtime.predict_concurrency: 12` (the knee); `min_replica: 2` (no cold start); `trt_llm` FP8 KV + chunked prefill + paged-context reuse; `concurrency_target ≤ predict_concurrency`; slow `scale_down_delay`. |
| [`model.py`](model.py) | Truss `Model` (`load`/`predict`) | Streams tokens (SSE) so TTFT ≠ total time; exposes **KV-cache utilization** as a backpressure signal (Baseten has no first-party KV metric). |
| [`chains_router.py`](chains_router.py) | Baseten **Chains** conductor + tiers | `@chains.mark_entrypoint` conductor with admission gate, deadline check, and the `primary → fallback → retrieval → 503` ladder using `chains.RPCOptions(timeout_sec=, retries=)`. |

## How it would deploy (sketch)
```bash
# each specialist / tier is its own Truss; the Chain wires them together
truss push                       # primary specialist (config.yaml + model.py)
truss chains push chains_router.py   # conductor + fallback + retrieval, scaled independently
```
Per-Chainlet **replica counts** (min/max) are set in the Baseten UI or the
autoscaling API — they are not in the Chains Python SDK. The autoscaling block in
`config.yaml` is the per-model equivalent.

## What's real vs. illustrative
- **Real:** field names, SDK symbols (`chains.ChainletBase`, `mark_entrypoint`,
  `depends`, `StubBase`, `RemoteConfig`, `Compute`, `RPCOptions`), the engine-builder
  KV/quant/prefill knobs, the async priority/queue fields.
- **Illustrative:** the engine bodies (a `_FakeEngine` placeholder), the admission
  gate's internals (a semaphore stands in for what the simulator models precisely),
  and the specialty routing. Treat this as the wiring diagram, not a turnkey deploy.
