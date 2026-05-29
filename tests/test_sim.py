#!/usr/bin/env python3
"""Behavioral tests that encode the claims the artifact makes. Pure stdlib — run
either with `python3 tests/test_sim.py` or `pytest`. Each test pins one assertion
from the writeup so the numbers in README/DESIGN can't silently drift."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim import run, SimConfig, naive_config   # noqa: E402


def test_naive_collapses_and_stays_collapsed():
    nv = run(naive_config(SimConfig()))
    s = nv.summary()
    assert s["failure_pct"] > 80, s["failure_pct"]          # mass timeout under the burst
    assert s["preemptions"] > 100_000, s["preemptions"]     # KV recompute storm
    assert s["retries"] > 500, s["retries"]                 # retry amplification
    rec = nv.window_summary()["recovery"]
    assert rec["goodput_pct"] < 5, rec                      # metastable: never recovers


def test_failsafe_stays_safe_and_recovers():
    fs = run(SimConfig())
    s = fs.summary()
    assert s["served_safe_pct"] > 95, s["served_safe_pct"]  # almost everyone served safely
    assert s["preemptions"] == 0, s["preemptions"]          # capped below the cliff -> no preemption
    assert s["retries"] == 0, s["retries"]                  # nothing times out -> no retry storm
    assert s["failure_pct"] < 2, s["failure_pct"]
    w = fs.window_summary()
    assert w["spike"]["safe_pct"] > 95, w["spike"]          # safe even at the peak
    assert w["recovery"]["primary_pct"] > 85, w["recovery"] # full quality restored after burst
    assert w["recovery"]["ttft_p95_ms"] < 400, w["recovery"]


def test_failsafe_beats_naive_by_a_mile():
    fs, nv = run(SimConfig()).summary(), run(naive_config(SimConfig())).summary()
    assert fs["served_safe_pct"] > 10 * nv["served_safe_pct"]
    assert fs["ttft_p95_ms"] < nv["ttft_p95_ms"] / 5


def test_deterministic():
    a = run(SimConfig()).summary()
    b = run(SimConfig()).summary()
    assert a == b, "same seed + config must reproduce byte-for-byte"


def test_prefix_affinity_raises_cache_hit():
    on = run(SimConfig(spike_multiplier=1.0, use_prefix_affinity=True)).summary()
    off = run(SimConfig(spike_multiplier=1.0, use_prefix_affinity=False)).summary()
    assert on["prefix_cache_hit_pct"] > off["prefix_cache_hit_pct"] + 10
    assert on["ttft_p50_ms"] <= off["ttft_p50_ms"]


def test_shed_path_actually_fires_when_floor_is_capped():
    # Shrink the degraded tiers to fleet scale + extreme burst: the 503 last
    # resort must engage (proves it isn't dead code).
    s = run(SimConfig(spike_multiplier=12.0, fallback_extra_capacity=0.1,
                      retrieval_slots=1)).summary()
    assert s["shed_pct"] > 0, s["shed_pct"]


def test_ladder_degrades_quality_not_safety():
    # As the burst grows, full-quality share drops but failures stay ~zero.
    light = run(SimConfig(spike_multiplier=2.0)).summary()
    heavy = run(SimConfig(spike_multiplier=10.0)).summary()
    assert heavy["primary_pct"] < light["primary_pct"]
    assert heavy["failure_pct"] < 3, heavy["failure_pct"]


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n  {len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
