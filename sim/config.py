"""Tunable constants for the simulator, modeling a high-traffic medical-RAG workload.

Every number here is a knob you would actually set on a real deployment; the
`baseten/` directory maps each one to its concrete Baseten / Truss / Chains
field. Latency units are milliseconds, capacity units are "tokens" (we set
1 KV slot == 1 token to keep the arithmetic legible).

The headline SLO is a ~160 ms responsiveness target — a figure publicly cited by
a leading medical-AI platform for clinician-facing answers. We interpret it as
p95 time-to-first-token <= 160 ms.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SimConfig:
    # ---- horizon / resolution ----
    horizon_ms: float = 90_000.0       # 90 s run
    tick_ms: float = 10.0              # fixed simulation step
    bin_ms: float = 1_000.0           # time-series aggregation bucket
    seed: int = 7

    # ---- SLOs ----
    slo_ttft_ms: float = 160.0         # p95 TTFT target (the "160 ms" number)
    slo_itl_ms: float = 25.0           # smooth-streaming inter-token budget
    deadline_budget_ms: float = 10_000.0  # completion timeout: clinician abandons after ~10 s of nothing

    # ---- fleet ----
    n_replicas: int = 4                # matches Baseten's 4-replica Dynamo benchmark
    kv_capacity_tokens: float = 90_000.0   # per-replica KV budget (paged KV on one H100-class GPU)
    engine_max_running: int = 256      # engine's own max concurrent sequences (max_num_seqs)
    preempt_util: float = 0.90         # KV utilization where the engine starts preempting
    cliff_steepness: float = 12.0      # how violently latency explodes past preempt_util ("swapping cliff")

    # ---- engine cost model ----
    prefill_ms_per_ktok: float = 20.0  # 8k *uncached* tokens -> ~160 ms prefill (right at the SLO!)
    decode_base_ms: float = 6.0        # inter-token latency at batch=1
    decode_softcap: float = 24.0       # batch size at which decode step-time has ~doubled
    prefill_softcap: float = 8.0       # concurrent prefills sharing the compute lane

    # ---- prefix cache ----
    prefix_cache_slots: int = 12       # distinct prefixes a replica keeps hot (LRU + KV offload)

    # ---- fail-safe admission control (maps to predict_concurrency / concurrency_target) ----
    predict_concurrency: int = 12      # hard per-replica ceiling (Truss runtime.predict_concurrency)
    use_admission_control: bool = True
    # Static cap at the measured knee is the default fail-safe primitive (it maps
    # 1:1 to predict_concurrency). AIMD is an opt-in refinement — see the writeup
    # for why a latency-signalled controller must keep its floor above baseline
    # demand or it self-locks below it.
    use_adaptive_concurrency: bool = False
    aimd_floor: int = 8                # floor MUST exceed baseline demand or AIMD self-locks
    shed_at_projected_ms: float = 800.0   # shed if projected time-to-first-token exceeds 5x the SLO

    # ---- fail-safe routing ----
    use_prefix_affinity: bool = True   # hash prompt prefix -> preferred replica (KV-cache affinity)
    affinity_imbalance_slack: int = 6  # if preferred replica is this much busier, fall back to P2C

    # ---- fail-safe resilience ----
    use_circuit_breaker: bool = True
    breaker_error_rate: float = 0.5    # open if >50% of recent dispatches fail
    breaker_cooldown_ms: float = 3_000.0
    retry_budget_frac: float = 0.10    # server-side retries capped at 10% of traffic
    use_fallback_ladder: bool = True
    fallback_extra_capacity: float = 1.6  # fallback pool is cheaper/faster -> higher effective knee
    retrieval_only_ms: float = 40.0    # latency of the sources-only safe answer (no gen GPU)
    retrieval_slots: int = 240         # concurrent capacity of the retrieval-only floor

    # ---- naive client behavior (the load amplifier) ----
    naive_max_attempts: int = 3        # naive client retries timeouts -> retry storm
    naive_retry_backoff_ms: float = 250.0

    # ---- workload ----
    n_topics: int = 40                 # background clinical topics (Zipf-popular)
    zipf_s: float = 1.1                # popularity skew (few hot topics dominate)
    lambda_base_rps: float = 8.0       # baseline arrivals/sec (~63% of the 4 x knee capacity)
    shared_prefix_mean: int = 7_000    # system prompt + retrieved NEJM/JAMA passages
    shared_prefix_jitter: int = 1_000
    unique_mean: int = 900             # the clinician's actual question
    unique_jitter: int = 500
    output_mean: int = 420
    output_jitter: int = 180

    # ---- the burst: a drug-recall headline spikes ONE brand-new (cold) topic ----
    spike_start_ms: float = 30_000.0
    spike_end_ms: float = 50_000.0
    spike_multiplier: float = 5.0      # 5x extra load during the spike
    spike_is_cold: bool = True         # the hot topic's prefix starts uncached everywhere


# A naive policy: everything off. This is the strawman that melts down.
def naive_config(base: SimConfig | None = None) -> SimConfig:
    c = base or SimConfig()
    return SimConfig(
        **{
            **c.__dict__,
            "use_admission_control": False,
            "use_adaptive_concurrency": False,
            "use_prefix_affinity": False,     # round-robin
            "use_circuit_breaker": False,
            "use_fallback_ladder": False,
            "predict_concurrency": 10_000,    # effectively unbounded admission
        }
    )
