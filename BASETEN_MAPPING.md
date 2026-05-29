# Mechanism → Baseten primitive

Every fail-safe mechanism in this design maps to a real Baseten / Truss / Chains
knob. Field names verified against `docs.baseten.co` on 2026-05-28; where a name is
ambiguous across pages or unverified, it's flagged. The simulator column points to
where each is modeled in [`sim/`](sim/).

| # | Mechanism | Baseten primitive (exact field) | Source | Modeled in |
|---|---|---|---|---|
| 1 | **Concurrency cap at the knee** | `runtime.predict_concurrency` (per-replica admission ceiling) | [concurrency](https://docs.baseten.co/performance/concurrency) | `config.predict_concurrency`, `router.per_replica_cap` |
| 1b | Autoscale trigger ≤ the cap | `concurrency_target` (≤ `predict_concurrency`), `target_utilization_percentage` (default 70) | [concurrency](https://docs.baseten.co/performance/concurrency) | `router`, `config` |
| 2 | **Bounded, deadline-aware queue + shed** | App logic in the entrypoint Chainlet (Baseten's own queue is unbounded at `max_replica`: *"requests queue rather than triggering new replicas"*) | [autoscaling](https://docs.baseten.co/deployment/autoscaling) | `router.step` (CoDel-style drain + shed) |
| 3 | **Prefix-cache-affinity routing** | NVIDIA-Dynamo **KV-cache-aware router** (overlap score + radix tree); cooperate by passing a stable prefix key | [Dynamo blog](https://www.baseten.co/blog/how-baseten-achieved-2x-faster-inference-with-nvidia-dynamo/) | `router._pick_replica`, `model_node.prefix_cache` |
| 3b | Prefix KV reuse on the engine | `trt_llm.plugin_configuration.use_paged_context_fmha`, `paged_kv_cache`; vLLM `enable_prefix_caching` | [engine builder](https://docs.baseten.co/engines/engine-builder-llm/engine-builder-config) | `model_node` (cached prefix ⇒ skip prefill, share KV) |
| 3c | KV offload (spill, don't recompute) | `trt_llm.runtime.kv_cache_host_memory_bytes`; KV-cache-offload-to-storage | [Dynamo blog](https://www.baseten.co/blog/how-baseten-achieved-2x-faster-inference-with-nvidia-dynamo/) | abstracted (larger effective cache) |
| 3d | More KV headroom | `trt_llm.build.quantization_type: fp8_kv`; `runtime.kv_cache_free_gpu_mem_fraction: 0.85` | [engine builder](https://docs.baseten.co/engines/engine-builder-llm/engine-builder-config) | `config.kv_capacity_tokens`, `preempt_util` |
| — | Long prompts don't spike everyone's ITL | `trt_llm.runtime.enable_chunked_context` (chunked prefill, Sarathi-style) | [engine builder](https://docs.baseten.co/engines/engine-builder-llm/engine-builder-config) | `model_node` prefill/decode interleave |
| 4 | **Keep hot path warm (no cold start)** | `min_replica ≥ 2`; network accelerator (on by default); Baseten Delivery Network (BDN) 3-tier cache | [cold starts](https://docs.baseten.co/performance/cold-starts) | engine pre-warms hot prefixes |
| 5 | **Circuit breaker / RPC discipline** | `chains.RPCOptions(retries=, timeout_sec=, concurrency_limit=)` between Chainlets | [Chains SDK](https://docs.baseten.co/reference/sdk/chains) | `router.CircuitBreaker`, retry budget |
| 6 | **Ensemble orchestration (conductor → specialists)** | **Chains**: `chains.ChainletBase`, `@chains.mark_entrypoint`, `chains.depends`, `chains.StubBase`; each Chainlet scales independently via `chains.Compute(...)` | [Chains overview](https://docs.baseten.co/development/chain/overview) | `baseten/chains_router.py` (topology) |
| 7 | **Async for batch (don't starve interactive)** | `async_predict` with `priority` (0/1/2, default 1; sync takes priority), `max_time_in_queue_seconds`, `inference_retry_config` | [async](https://docs.baseten.co/inference/async) | not in sim (interactive-only) |
| — | **Autoscaling as the slow background fix** | `autoscaling_window` (60 s), `scale_down_delay` (900 s, removes ⌈excess/2⌉), SLA-based planner | [autoscaling](https://docs.baseten.co/deployment/autoscaling) | discussed; sim covers the sub-minute gap |
| — | **Multi-region / failover** | **Multi-Cloud Capacity Management (MCM)** — active-active, latency-aware routing, automatic failover (used in production by large medical-AI platforms) | [MCM](https://www.baseten.co/products/multi-cloud-capacity-management/) | out of scope (single region) |
| — | **Cheap embeddings hot path** | **Baseten Embeddings Inference (BEI)** (TRT-LLM-based, OpenAI-compatible) | [BEI blog](https://www.baseten.co/blog/how-we-built-bei-high-throughput-embedding-inference/) | retrieval tier service time |
| — | **Saturation signal** | Dashboard *Concurrent requests*, *Time to first byte*, *GPU memory*; export via Prometheus `/metrics` (OpenTelemetry) | [metrics](https://docs.baseten.co/observability/metrics) | `metrics.sample_system` |

## Two platform details worth noting

1. **The concurrency-constraint direction is stated inconsistently across Baseten's
   own pages.** The current `performance/concurrency` page says
   **`concurrency_target ≤ predict_concurrency`** (autoscale trigger ≤ in-container
   ceiling); some older guides phrase it the other way. This config follows the
   current page.

2. **There is no first-party KV-cache-utilization metric** on the standard
   dashboard — *GPU memory* is the closest proxy. Since KV utilization is the single
   best backpressure signal, this design surfaces it from the model server and
   scrapes it (see [`baseten/model.py`](baseten/model.py)). Worth proposing as a platform improvement.

## Things I could not fully verify (flagged honestly)

- Per-Chainlet **min/max replica** autoscaling is set in the UI / autoscaling API,
  not the Chains Python SDK (`Compute`/`RemoteConfig` expose no replica counts).
- Async **max queue depth** isn't a documented configurable field — only
  `max_time_in_queue_seconds` + an org-level rate limit bound it.
- Exact exported Prometheus metric identifiers (`baseten_*`) aren't enumerated on
  the export doc page.
- Doc pages carry no visible "last updated" date, so field names are "current as of
  the 2026-05-28 fetch."
