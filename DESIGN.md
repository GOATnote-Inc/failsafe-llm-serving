# Design: a fail-safe, low-latency inference path on Baseten

> **Thesis.** Bound every queue, cap concurrency at the *latency knee* (not max
> throughput), shed early at the edge with deadline-awareness, route for
> prefix-cache affinity, and always keep a cheaper degraded path — so that when a
> GPU node saturates, the system loses **quality** gracefully instead of falling
> over.

This document designs the query path, explains the failure it prevents, and maps
every mechanism to a concrete Baseten primitive. A runnable simulator in [`sim/`](sim/)
reproduces the failure and demonstrates the fix; numbers below come from it.

---

## 1. The problem & workload assumptions

A clinician asks a question. The system retrieves peer-reviewed passages, a
**conductor** model routes to one or more fine-tuned **specialist** sub-models,
and a synthesized, citation-grounded answer is streamed back.

The workload below is *representative* of high-traffic, RAG-based medical-AI
assistants, drawn from public sources (Baseten case studies, engineering talks,
and founder interviews — cited in §10). It is a plausible model to design
against, not a claim about any single company's internal architecture.

| Assumption (from public sources) | Design consequence |
|---|---|
| **~160 ms responsiveness target** — publicly cited by a leading medical-AI platform for clinician-facing answers (with a stated end-to-end 700 ms → 160 ms reduction on Baseten) | The SLO is **p95 TTFT ≈ 160 ms**, treated as a hard product constraint. |
| **Conductor → several specialist models** in a directed-tree topology (publicly described) | "A node saturates" can happen at *any hop*; tail latency = slowest specialist. Fail-safe must be **per-stage**. |
| **Long, shared prompt prefixes** (system prompt + retrieved NEJM/JAMA text) | Prefill-dominated TTFT; prefixes are heavily reused → **prefix caching is the dominant lever**. |
| **Bursty traffic** (viral spikes; GPUs are the hard part to scale elastically) | Must absorb bursts faster than GPU autoscaling can react. |
| **Billions of LLM calls/week**, grounded only in a licensed corpus with inline citations | Cost-per-query matters; safety (grounding/citations) **cannot be dropped under load**. |

## 2. The failure we are preventing

Under a load spike (e.g., a drug-recall headline floods one topic), a naive path
walks itself into **metastable collapse**:

```
spike / inflated context
  → KV-cache utilization on the hot replica crosses ~90%
  → engine begins RECOMPUTE-preemption — the "swapping cliff"     [vLLM V1 default]
  → TTFT and inter-token latency spike non-linearly
  → Baseten autoscaler won't react for up to autoscaling_window (60 s),
    then a large model "can take minutes" to cold-start                 [Baseten docs]
  → at max_replica, "requests queue rather than triggering new replicas" [Baseten docs]
  → clients time out → RETRY → load amplifies (>50% of metastable
    sustaining effects are retry-induced)                                [OSDI '22]
  → COLLAPSE persists after the spike is gone — the system cannot self-recover
```

The simulator reproduces this exactly: the naive path hits **2.9M preemptions**,
**2,011 client retries**, **93.5% failures**, and a p95 TTFT of **9.7 s** — and it
**stays collapsed** through the entire recovery window (0% goodput at t > 50 s).

The key insight: the platform's own scale-out is **structurally too slow to catch
a burst**, and its default behavior at the ceiling is an **unbounded queue**. That
gap is what the fail-safe layer owns.

## 3. The query path

```
Clinician ── SSE stream
   │
[Edge / API gateway]   authn · per-tenant token-bucket · stamp DEADLINE · classify interactive|async
   │
[Conductor / Router]   ← Baseten Chains entrypoint Chainlet
   │   ADMISSION    concurrency capped at the latency KNEE (predict_concurrency);
   │                bounded, DEADLINE-AWARE queue (drop expired; shed if projected wait > budget)
   │   ROUTE        prefix-cache affinity (stable hash) → cooperate w/ Dynamo KV-aware router;
   │                power-of-two-choices fallback when a replica is a hotspot
   │   RESILIENCE   per-replica circuit breaker · retry budget ≤10% · hedge only with slack
   │   PER-STAGE    deadlines across specialists; synthesize partial results if one is slow
   │
   ├─► PRIMARY specialists   (TRT-LLM/vLLM, min_replica≥2, FP8 KV, paged + chunked prefill)
   ├─► FALLBACK pool         (smaller/quantized model — faster, cheaper, own Chainlet)
   └─► SAFE DEGRADE          retrieval-only ranked citations · cached prior answer · 503+Retry-After
```

## 4. The mechanisms (each → why it fails safe → Baseten knob)

1. **Cap concurrency at the latency knee — not max throughput.** *The* foundational
   move. Load-test to find the largest per-replica batch where p95 TTFT/ITL stay in
   SLO and KV utilization stays < ~85%. Set `predict_concurrency` there.
   → A replica **cannot** enter the swapping cliff because we never admit enough
   work to fill KV past the knee. **Fail-safe by construction**, not by reaction.
   *(Sim: 0 preemptions on the fail-safe path vs 2.9M naive.)*

2. **Bounded, deadline-aware queue + early shedding.** Every request carries a
   deadline. If projected wait would blow it, reject **now** (fast 503 + `Retry-After`
   with jitter); drop already-expired requests at dequeue.
   → Removes the unbounded queue and the retry-storm amplifier. *(Sim: 0 retries
   on the fail-safe path vs 2,011 naive.)*

3. **Prefix-cache-affinity routing.** Hash the prompt prefix (system prompt +
   retrieved-doc set) to a preferred replica; cooperate with Baseten's
   NVIDIA-Dynamo KV-aware router. Fall back to power-of-two-choices when the
   preferred replica is a hotspot.
   → Higher cache hit ⇒ less prefill ⇒ lower TTFT **and** lower KV pressure ⇒ the
   knee moves outward ⇒ more burst absorbed before shedding. *(Sim: +23.5 pts cache
   hit, −24% p50 TTFT vs round-robin. Baseten's own number: 89% hit, −50% TTFT.)*

4. **Keep the hot path warm.** `min_replica ≥ 2`; never scale-to-zero on the
   interactive tier. → Removes minutes-long cold starts from the request path.

5. **Circuit breakers + retry budget + adaptive hedging.** Stop routing to a
   saturated/erroring replica; cap retries to ≤10% of traffic; hedge only when
   there's latency slack (off under load). → No piling onto a wedged node; no
   self-inflicted amplification.

6. **Per-stage budgets across the ensemble.** The conductor gives each specialist
   a deadline and synthesizes from whoever returns in time. → One slow specialist
   degrades the answer, it doesn't hang the request.

7. **Async for non-interactive work.** Batch/precompute jobs go through Baseten
   **async inference** (priority `2`, `max_time_in_queue_seconds`), excluded from
   the interactive concurrency signal. → Batch can't starve clinicians.

## 5. The clinical fail-safe ladder

When primary has no slot within the responsiveness budget, degrade **quality**,
never **safety**:

1. **Full ensemble** — conductor → specialists → synthesis, fully grounded.
2. **Reduced ensemble** — smaller/quantized model. Still grounded, less nuanced.
3. **Retrieval-only** — *"I can't fully synthesize right now; here are the most
   relevant peer-reviewed sources."* No generation GPU, clinically safe, still useful.
4. **Cached prior answer** for a prefix-identical query.
5. **Fast 503 + `Retry-After`** — an honest "busy," never a spinning timeout.

> **Invariant: never emit an ungrounded/hallucinated answer to make an SLO.** The
> degraded states are all either grounded or honestly empty. *In medicine, that is
> what "fail safe" means.*

The simulator shows the ladder absorbing a 1×→14× sweep with **FAILED pinned at
~0.1% and 503-shed at 0%** — load slides down the quality tiers, it does not fail.
Because the retrieval floor is cheap and scalable, hard-rejection (503) is the
theoretical last resort, not the operating mode.

## 6. Why autoscaling is necessary but not sufficient

Baseten's autoscaler (the SLA-based planner) is the right **background** fix: it
grows the primary pool on sustained load and is essential for cost. But it
**re-evaluates only once per `autoscaling_window` (default 60 s)** and then a large
model **cold-starts in minutes**. A clinical burst peaks in seconds. So:

- **Autoscaling** handles the minute-to-hour timescale (and cost via scale-down).
- **Admission control + the fallback ladder** handle the second-to-minute timescale
  — the gap where the naive system collapses.
- `scale_down_delay` (900 s) + removing only ⌈excess/2⌉ damps flapping.

## 7. Goodput, not throughput

Optimize **goodput** = fraction served *safely within the responsiveness budget*.
A saturated server can show 100% throughput while goodput is zero (everything
returns, all too slow). The headline chart is goodput-over-time through the burst
**and its recovery**.

Watch: KV-utilization/replica (alarm > 85%), queue depth + wait, prefix-cache hit
rate, shed rate, circuit state, fallback-tier distribution. Baseten exposes *Time
to first byte*, *Concurrent requests*, and *GPU memory* on the dashboard but **no
first-party KV-utilization metric** — so the model server surfaces it explicitly
(see [`baseten/model.py`](baseten/model.py)) and exports it via the Prometheus `/metrics` endpoint.

## 8. A lesson the simulator surfaced: adaptive concurrency can self-lock

An AIMD controller that **keys off total TTFT** mistakes inherent cold-prefill time
for overload and throttles capacity it shouldn't. Worse, if its floor sits *below*
baseline demand, it shrinks during a burst and **never climbs back** — the queue
never drains, so the "healthy" probe-up never fires. Fixes: (a) signal on **queue
wait**, not TTFT; (b) keep the floor **above** baseline demand. The default policy
here uses a **static cap at the knee** (robust, maps 1:1 to `predict_concurrency`);
AIMD is an opt-in refinement. *This exact failure mode is in the code and is a good
thing to talk through.*

## 9. What's modeled vs. not (honesty)

**Modeled:** continuous-batching decode whose step-time grows with batch (the
throughput↔latency feedback); KV cache as the real concurrency limit; recompute
preemption + the swapping cliff; prefix-cache reuse (hit ⇒ cheap prefill + shared
KV); admission control; deadline-aware bounded queue; prefix-affinity + P2C;
circuit breaker; fallback ladder; client retry storm.

**Abstracted:** a real tokenizer/attention kernel; exact TRT-LLM/vLLM scheduler
internals; network/serialization latency; multi-tenant fairness; the conductor's
*content* routing (we model it as a topology + per-stage budgets, not real model
quality). The simulator argues about **systems dynamics**, not model accuracy. The
constants are plausible, not measured from a specific GPU — the *shapes* (collapse
vs graceful degradation, recovery vs metastability) are the point, and they are
robust to the constants.

## 10. Sources

- Baseten × NVIDIA Dynamo — KV-cache-aware routing (89% hit, −50% TTFT, −34% TPOT, −49% p99): https://www.baseten.co/blog/how-baseten-achieved-2x-faster-inference-with-nvidia-dynamo/
- Baseten concurrency (`predict_concurrency`, `concurrency_target`): https://docs.baseten.co/performance/concurrency
- Baseten autoscaling (`autoscaling_window`, `scale_down_delay`, cold start): https://docs.baseten.co/deployment/autoscaling
- Medical-AI platform — Baseten case study (160 ms, billions of calls/week, MCM): https://www.baseten.co/resources/customers/openevidence-delivers-instant-medical-information-with-baseten/
- Medical-AI platform — engineering writeup (architecture, Fluid Compute): https://vercel.com/blog/how-openevidence-built-a-healthcare-ai-that-physicians-can-trust
- Founder interview — ensemble/conductor architecture (Sequoia *Training Data*): https://www.sequoiacap.com/podcast/training-data-daniel-nadler/
- *Inside vLLM* — preemption, paged KV, continuous batching: https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html
- *Metastable Failures in the Wild* (OSDI '22): https://www.usenix.org/system/files/osdi22-huang-lexiang.pdf
- AWS Builders' Library — timeouts, retries, backoff + jitter: https://aws.amazon.com/builders-library/timeouts-retries-and-backoff-with-jitter/
- *Sarathi-Serve* — chunked prefill: https://arxiv.org/pdf/2403.02310
