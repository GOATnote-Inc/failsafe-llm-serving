.PHONY: all test demo collapse affinity ladder clean
PY ?= python3

all: test demo                 ## run the test suite, then regenerate every chart

test:                          ## behavioral tests that pin the claimed numbers
	$(PY) tests/test_sim.py

demo: collapse affinity ladder ## regenerate all SVG charts in plots/

collapse:                      ## headline: naive collapse vs fail-safe degradation
	$(PY) experiments/run_collapse_vs_failsafe.py

affinity:                      ## isolate the KV-cache-affinity routing lever
	$(PY) experiments/run_prefix_affinity.py

ladder:                        ## quality ladder vs burst intensity
	$(PY) experiments/run_fallback_ladder.py

clean:
	rm -rf sim/__pycache__ experiments/__pycache__ tests/__pycache__ .pytest_cache
