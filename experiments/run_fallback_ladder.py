#!/usr/bin/env python3
"""Sweep burst intensity and watch the fail-safe quality ladder absorb it.

The point: as offered load climbs far past primary capacity, requests don't fail
— they slide DOWN the quality ladder (full ensemble -> small model -> sources-only)
while the FAILED and 503 fractions stay ~zero. Because the retrieval-only floor is
cheap and scalable, the system essentially never has to hard-reject a clinician;
503-shedding is the theoretical last resort, not the operating mode.

Produces: plots/fallback_ladder.svg   (tier composition vs burst multiplier)
Run: python3 experiments/run_fallback_ladder.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim import run, SimConfig                          # noqa: E402
from sim import plotting as P                           # noqa: E402
from sim.types import Outcome                           # noqa: E402

PLOTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")
MULTS = [1, 2, 4, 6, 10, 14]


def tier_pct(metrics):
    n = len(metrics.terminal) or 1
    c = {o: 0 for o in Outcome}
    for r in metrics.terminal:
        c[r.outcome] += 1
    return {
        "primary": 100.0 * c[Outcome.SERVED_PRIMARY] / n,
        "fallback": 100.0 * c[Outcome.SERVED_FALLBACK] / n,
        "retrieval": 100.0 * c[Outcome.SERVED_RETRIEVAL] / n,
        "shed": 100.0 * c[Outcome.SHED] / n,
        "failure": 100.0 * (c[Outcome.TIMEOUT] + c[Outcome.DROPPED_EXPIRED]) / n,
    }


def main():
    os.makedirs(PLOTS, exist_ok=True)
    rows = [(m, tier_pct(run(SimConfig(spike_multiplier=float(m))))) for m in MULTS]

    print("\n  burst   primary  fallback  retrieval   shed   FAILED   (% of all requests)")
    print("  " + "-" * 64)
    for m, t in rows:
        print(f"  {m:3d}x   {t['primary']:7.1f}  {t['fallback']:8.1f}  {t['retrieval']:9.1f}  "
              f"{t['shed']:5.1f}  {t['failure']:6.1f}")

    groups = [f"{m}x" for m, _ in rows]
    series = [
        ("primary", [t["primary"] for _, t in rows], P.PALETTE["primary"]),
        ("fallback", [t["fallback"] for _, t in rows], P.PALETTE["fallback"]),
        ("retrieval", [t["retrieval"] for _, t in rows], P.PALETTE["retrieval"]),
        ("shed (503)", [t["shed"] for _, t in rows], P.PALETTE["shed"]),
        ("FAILED", [t["failure"] for _, t in rows], P.PALETTE["failure"]),
    ]
    P.grouped_bars(
        os.path.join(PLOTS, "fallback_ladder.svg"),
        "Quality ladder vs burst intensity: load slides down tiers, it does not fail",
        groups, series, "% of requests", ymax=100)
    print(f"\n  wrote 1 chart to {PLOTS}/\n")


if __name__ == "__main__":
    main()
