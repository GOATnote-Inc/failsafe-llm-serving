#!/usr/bin/env python3
"""Headline experiment: the same 5x burst through a naive path vs the fail-safe path.

Produces:
  plots/goodput_over_time.svg   — safely-served throughput; naive flatlines and stays down
  plots/ttft_over_time.svg      — p95 TTFT vs the 160 ms SLO; naive blows the budget by ~60x
  plots/tier_distribution.svg   — fail-safe quality ladder shifting under load (graceful degradation)
  plots/naive_kv_collapse.svg   — naive KV utilization pinned at 100% + the preemption storm

Run: python3 experiments/run_collapse_vs_failsafe.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim import run, SimConfig, naive_config          # noqa: E402
from sim import plotting as P                          # noqa: E402

PLOTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")


def _clamp(vals, cap):
    return [min(v, cap) for v in vals]


def main():
    os.makedirs(PLOTS, exist_ok=True)
    cfg = SimConfig()
    fs = run(cfg)
    nv = run(naive_config(cfg))
    fss, nvs = fs.summary(), nv.summary()
    fsx, nvx = fs.series(), nv.series()
    shade = (cfg.spike_start_ms / 1000.0, cfg.spike_end_ms / 1000.0, "5x burst")

    # ---- console report ----
    print("\n  metric                  FAIL-SAFE        NAIVE")
    print("  " + "-" * 48)
    rows = [
        ("requests (incl retries)", "total_requests", "{:.0f}"),
        ("safely served", "served_safe_pct", "{:.1f}%"),
        ("goodput (safe & <=160ms)", "goodput_pct", "{:.1f}%"),
        ("full-quality (primary)", "primary_pct", "{:.1f}%"),
        ("FAILED (timeout/dropped)", "failure_pct", "{:.1f}%"),
        ("client retries", "retries", "{:.0f}"),
        ("KV preemptions", "preemptions", "{:.0f}"),
        ("p95 TTFT", "ttft_p95_ms", "{:.0f} ms"),
        ("p99 TTFT", "ttft_p99_ms", "{:.0f} ms"),
    ]
    for label, key, fmt in rows:
        print(f"  {label:24s}{fmt.format(fss[key]):>12s}{fmt.format(nvs[key]):>14s}")

    print("\n  window      FAIL-SAFE (good/safe/prim/p95)      NAIVE (good/safe/prim/p95)")
    print("  " + "-" * 70)
    fw, nw = fs.window_summary(), nv.window_summary()
    for k in ("calm", "spike", "recovery"):
        f, n = fw[k], nw[k]
        print(f"  {k:10s} {f['goodput_pct']:4.0f}% {f['safe_pct']:4.0f}% {f['primary_pct']:4.0f}% "
              f"{f['ttft_p95_ms']:5.0f}ms      {n['goodput_pct']:4.0f}% {n['safe_pct']:4.0f}% "
              f"{n['primary_pct']:4.0f}% {n['ttft_p95_ms']:5.0f}ms")

    # ---- charts ----
    P.line_chart(
        os.path.join(PLOTS, "goodput_over_time.svg"),
        "Safely-served throughput through a 5x burst  (higher = better)",
        fsx["t_s"],
        [("offered load", fsx["offered_rps"], P.PALETTE["offered"]),
         ("fail-safe: safely served", fsx["goodput_safe_rps"], P.PALETTE["failsafe"]),
         ("naive: safely served", nvx["goodput_safe_rps"], P.PALETTE["naive"])],
        "time (s)", "requests / s", shade=shade)

    P.line_chart(
        os.path.join(PLOTS, "ttft_over_time.svg"),
        "p95 time-to-first-token  (naive clipped at 2000 ms; it peaks ~9700 ms)",
        fsx["t_s"],
        [("fail-safe p95 TTFT", _clamp(fsx["ttft_p95_ms"], 2000), P.PALETTE["failsafe"]),
         ("naive p95 TTFT", _clamp(nvx["ttft_p95_ms"], 2000), P.PALETTE["naive"])],
        "time (s)", "p95 TTFT (ms)",
        hlines=[(cfg.slo_ttft_ms, "160 ms SLO", P.PALETTE["slo"])], shade=shade, ymax=2000)

    P.stacked_area(
        os.path.join(PLOTS, "tier_distribution.svg"),
        "Fail-safe quality ladder: how every request was served (graceful degradation)",
        fsx["t_s"],
        [("primary (full ensemble)", fsx["primary_rps"], P.PALETTE["primary"]),
         ("fallback (small model)", fsx["fallback_rps"], P.PALETTE["fallback"]),
         ("retrieval-only (sources)", fsx["retrieval_rps"], P.PALETTE["retrieval"]),
         ("shed (503 + Retry-After)", fsx["shed_rps"], P.PALETTE["shed"]),
         ("FAILED", fsx["failure_rps"], P.PALETTE["failure"])],
        "time (s)", "requests / s", shade=shade)

    P.line_chart(
        os.path.join(PLOTS, "naive_kv_collapse.svg"),
        "Why naive collapses: KV cache pinned at capacity -> recompute-preemption storm",
        nvx["t_s"],
        [("KV utilization (max replica)", [u * 100 for u in nvx["kv_util_max"]], P.PALETTE["kv"]),
         ("preemptions / s (/100)", [p / 100.0 for p in nvx["preemptions"]], P.PALETTE["naive"])],
        "time (s)", "KV util %   |   preemptions/s ÷100",
        hlines=[(cfg.preempt_util * 100, "preempt threshold", P.PALETTE["slo"])], shade=shade)

    print(f"\n  wrote 4 charts to {PLOTS}/\n")


if __name__ == "__main__":
    main()
