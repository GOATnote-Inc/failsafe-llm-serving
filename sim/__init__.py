"""A dependency-free simulator of a fail-safe, low-latency LLM inference path,
modeling a high-traffic medical-RAG workload, mapped to Baseten primitives.
Standard library only."""
from .config import SimConfig, naive_config
from .engine import Simulation
from .metrics import MetricsRecorder, percentile

__all__ = ["SimConfig", "naive_config", "Simulation", "MetricsRecorder", "percentile"]


def run(cfg: SimConfig) -> MetricsRecorder:
    return Simulation(cfg).run()
