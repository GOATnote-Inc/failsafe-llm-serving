#!/usr/bin/env python3
"""Isolate the KV-cache-affinity lever (everything else held fail-safe).

Medical-RAG prompts are long and heavily shared (one system prompt + retrieved
NEJM/JAMA passages per topic). Routing same-prefix requests to the same replica
maximizes KV reuse -> lower prefill -> lower TTFT and lower KV pressure. This is
the app-layer cooperation with Baseten's NVIDIA-Dynamo KV-aware router, which
reported -50% TTFT / 89% cache hit on a long-context stress test.

Produces: plots/prefix_affinity.svg   (p50/p95 TTFT, affinity on vs off)
Run: python3 experiments/run_prefix_affinity.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim import run, SimConfig                          # noqa: E402
from sim import plotting as P                           # noqa: E402

PLOTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")


def main():
    os.makedirs(PLOTS, exist_ok=True)
    # Steady load, no burst: in a queue-dominated burst the prefill win is masked,
    # so we isolate cache locality at a load where TTFT ~= prefill time.
    on = run(SimConfig(spike_multiplier=1.0, use_prefix_affinity=True)).summary()
    off = run(SimConfig(spike_multiplier=1.0, use_prefix_affinity=False)).summary()

    print("\n  metric                     AFFINITY ON     ROUND-ROBIN     delta")
    print("  " + "-" * 60)
    def row(label, key, fmt, better="lower"):
        a, b = on[key], off[key]
        if key.endswith("ms") or "ttft" in key:
            d = f"{(a-b)/b*100:+.0f}%" if b else "-"
        else:
            d = f"{a-b:+.1f}"
        print(f"  {label:26s}{fmt.format(a):>12s}{fmt.format(b):>14s}   {d:>7s}")
    row("prefix cache hit", "prefix_cache_hit_pct", "{:.1f}%")
    row("p50 TTFT", "ttft_p50_ms", "{:.0f} ms")
    row("p95 TTFT", "ttft_p95_ms", "{:.0f} ms")
    row("full-quality (primary)", "primary_pct", "{:.1f}%")
    row("goodput (safe & <=160ms)", "goodput_pct", "{:.1f}%")

    P.grouped_bars(
        os.path.join(PLOTS, "prefix_affinity.svg"),
        "KV-cache-affinity routing vs round-robin  (lower TTFT = better)",
        ["p50 TTFT (ms)", "p95 TTFT (ms)"],
        [("affinity on", [on["ttft_p50_ms"], on["ttft_p95_ms"]], P.PALETTE["a"]),
         ("round-robin", [off["ttft_p50_ms"], off["ttft_p95_ms"]], P.PALETTE["b"])],
        "milliseconds")
    P.grouped_bars(
        os.path.join(PLOTS, "prefix_affinity_quality.svg"),
        "KV-cache-affinity routing vs round-robin  (higher = better)",
        ["prefix cache hit %", "goodput %", "full-quality %"],
        [("affinity on", [on["prefix_cache_hit_pct"], on["goodput_pct"], on["primary_pct"]], P.PALETTE["a"]),
         ("round-robin", [off["prefix_cache_hit_pct"], off["goodput_pct"], off["primary_pct"]], P.PALETTE["b"])],
        "percent")
    print(f"\n  wrote 2 charts to {PLOTS}/\n")


if __name__ == "__main__":
    main()
