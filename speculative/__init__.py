"""
Transformer-based Speculative Decoding Framework

No vLLM dependency — runs on macOS (MPS / CPU) and Linux (CUDA).

Modules:
    proposers/  — KMP, HashTable, Trie n-gram draft proposers
    verifier    — Qwen-0.5B transformer verifier
    engine      — Speculative decoding orchestration loop
    metrics     — Hit rate, acceptance rate, speedup statistics
"""

from .metrics import MetricsTracker

__all__ = ["SpeculativeEngine", "TransformerVerifier", "MetricsTracker"]


def __getattr__(name):
    if name == "TransformerVerifier":
        from .verifier import TransformerVerifier
        return TransformerVerifier
    if name == "SpeculativeEngine":
        from .engine import SpeculativeEngine
        return SpeculativeEngine
    raise AttributeError(f"module 'speculative' has no attribute {name!r}")
