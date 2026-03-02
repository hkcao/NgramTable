"""
Metrics tracker for speculative decoding experiments.

Tracks per-step and aggregate statistics:
  - draft_hit_rate:     fraction of draft proposals that exactly matched the
                        verifier's greedy output (measures proposer quality)
  - token_acceptance_rate: fraction of proposed tokens accepted by verifier
  - mean_accepted_length:  avg number of tokens accepted per speculative step
  - speedup_ratio:      estimated wall-clock speedup vs pure autoregressive
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class StepMetrics:
    """Metrics for a single speculative decoding step."""
    num_proposed: int      # number of draft tokens proposed
    num_accepted: int      # number accepted (before bonus)
    bonus_token: int       # verifier's bonus token
    context_len: int       # context length at this step
    proposer_name: str     # which proposer was used
    propose_time_ms: float = 0.0   # proposer wall time
    verify_time_ms: float = 0.0    # verifier wall time


@dataclass
class RequestMetrics:
    """Aggregate metrics for a single generation request."""
    prompt: str
    total_output_tokens: int = 0
    total_propose_time_ms: float = 0.0
    total_verify_time_ms: float = 0.0
    total_baseline_time_ms: float = 0.0   # autoregressive equivalent time
    steps: List[StepMetrics] = field(default_factory=list)

    # Per-position acceptance (index = position within a speculative block)
    per_position_accepted: List[int] = field(default_factory=list)
    per_position_proposed: List[int] = field(default_factory=list)

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def total_proposed(self) -> int:
        return sum(s.num_proposed for s in self.steps)

    @property
    def total_accepted(self) -> int:
        return sum(s.num_accepted for s in self.steps)

    @property
    def draft_hit_rate(self) -> float:
        """
        Fraction of steps where all proposed tokens were accepted
        (i.e., the proposer perfectly predicted the next sequence).
        """
        if not self.steps:
            return 0.0
        perfect = sum(
            1 for s in self.steps
            if s.num_proposed > 0 and s.num_accepted == s.num_proposed
        )
        return perfect / len(self.steps)

    @property
    def token_acceptance_rate(self) -> float:
        """Fraction of proposed tokens accepted by the verifier."""
        total = self.total_proposed
        return self.total_accepted / total if total > 0 else 0.0

    @property
    def mean_accepted_length(self) -> float:
        """Average accepted tokens per step (including bonus = +1)."""
        if not self.steps:
            return 0.0
        # Each step emits accepted + 1 bonus token
        return sum(s.num_accepted + 1 for s in self.steps) / len(self.steps)

    @property
    def speculative_time_ms(self) -> float:
        return self.total_propose_time_ms + self.total_verify_time_ms

    @property
    def speedup_ratio(self) -> float:
        """
        Estimated speedup = baseline_time / speculative_time.

        baseline_time: time that pure AR generation would take, estimated as
            total_output_tokens * (avg_verify_time_per_token).
        speculative_time: actual propose + verify time.
        """
        if self.speculative_time_ms <= 0:
            return 0.0
        if self.total_baseline_time_ms > 0:
            return self.total_baseline_time_ms / self.speculative_time_ms
        # Fallback estimate when baseline wasn't measured
        if self.total_output_tokens <= 0:
            return 0.0
        # Each verify call processes (1 + num_proposed) tokens in one pass,
        # saving num_accepted extra AR steps.
        ar_steps = self.total_output_tokens
        spec_verify_calls = self.num_steps
        if spec_verify_calls == 0:
            return 1.0
        # rough estimate: speedup ≈ ar_steps / spec_verify_calls
        return ar_steps / spec_verify_calls

    @property
    def per_position_acceptance_rate(self) -> List[float]:
        """Acceptance rate broken down by position within a speculative block."""
        rates = []
        for acc, prop in zip(self.per_position_accepted, self.per_position_proposed):
            rates.append(acc / prop if prop > 0 else 0.0)
        return rates

    def to_dict(self) -> Dict:
        return {
            "num_steps": self.num_steps,
            "total_output_tokens": self.total_output_tokens,
            "total_proposed": self.total_proposed,
            "total_accepted": self.total_accepted,
            "draft_hit_rate": round(self.draft_hit_rate, 4),
            "token_acceptance_rate": round(self.token_acceptance_rate, 4),
            "token_acceptance_rate_pct": round(self.token_acceptance_rate * 100, 2),
            "mean_accepted_length": round(self.mean_accepted_length, 3),
            "speedup_ratio": round(self.speedup_ratio, 3),
            "total_propose_time_ms": round(self.total_propose_time_ms, 2),
            "total_verify_time_ms": round(self.total_verify_time_ms, 2),
            "total_baseline_time_ms": round(self.total_baseline_time_ms, 2),
            "speculative_time_ms": round(self.speculative_time_ms, 2),
            "per_position_acceptance_rate": [
                round(r, 4) for r in self.per_position_acceptance_rate
            ],
        }


class MetricsTracker:
    """
    Accumulates metrics across multiple requests for a benchmark run.
    """

    def __init__(self, proposer_name: str, num_speculative_tokens: int):
        self.proposer_name = proposer_name
        self.num_speculative_tokens = num_speculative_tokens
        self.requests: List[RequestMetrics] = []

    def new_request(self, prompt: str) -> RequestMetrics:
        req = RequestMetrics(prompt=prompt)
        self.requests.append(req)
        return req

    # ------------------------------------------------------------------
    # Aggregate stats across all requests
    # ------------------------------------------------------------------

    @property
    def total_output_tokens(self) -> int:
        return sum(r.total_output_tokens for r in self.requests)

    @property
    def total_proposed(self) -> int:
        return sum(r.total_proposed for r in self.requests)

    @property
    def total_accepted(self) -> int:
        return sum(r.total_accepted for r in self.requests)

    @property
    def token_acceptance_rate(self) -> float:
        p = self.total_proposed
        return self.total_accepted / p if p > 0 else 0.0

    @property
    def mean_accepted_length(self) -> float:
        total_steps = sum(r.num_steps for r in self.requests)
        if total_steps == 0:
            return 0.0
        return sum(
            s.num_accepted + 1
            for r in self.requests
            for s in r.steps
        ) / total_steps

    @property
    def draft_hit_rate(self) -> float:
        total_steps = sum(r.num_steps for r in self.requests)
        if total_steps == 0:
            return 0.0
        perfect = sum(
            1
            for r in self.requests
            for s in r.steps
            if s.num_proposed > 0 and s.num_accepted == s.num_proposed
        )
        return perfect / total_steps

    @property
    def avg_speedup(self) -> float:
        valid = [r.speedup_ratio for r in self.requests if r.speedup_ratio > 0]
        return sum(valid) / len(valid) if valid else 0.0

    @property
    def total_speculative_time_ms(self) -> float:
        return sum(r.speculative_time_ms for r in self.requests)

    @property
    def total_baseline_time_ms(self) -> float:
        return sum(r.total_baseline_time_ms for r in self.requests)

    @property
    def overall_speedup(self) -> float:
        if self.total_speculative_time_ms <= 0:
            return 0.0
        if self.total_baseline_time_ms > 0:
            return self.total_baseline_time_ms / self.total_speculative_time_ms
        # Estimate from step counts
        total_ar_steps = self.total_output_tokens
        total_verify_calls = sum(r.num_steps for r in self.requests)
        if total_verify_calls == 0:
            return 1.0
        return total_ar_steps / total_verify_calls

    def summary(self) -> Dict:
        return {
            "proposer": self.proposer_name,
            "num_speculative_tokens": self.num_speculative_tokens,
            "num_requests": len(self.requests),
            "total_output_tokens": self.total_output_tokens,
            "total_proposed": self.total_proposed,
            "total_accepted": self.total_accepted,
            "draft_hit_rate": round(self.draft_hit_rate, 4),
            "draft_hit_rate_pct": round(self.draft_hit_rate * 100, 2),
            "token_acceptance_rate": round(self.token_acceptance_rate, 4),
            "token_acceptance_rate_pct": round(self.token_acceptance_rate * 100, 2),
            "mean_accepted_length": round(self.mean_accepted_length, 3),
            "overall_speedup": round(self.overall_speedup, 3),
            "avg_speedup_per_request": round(self.avg_speedup, 3),
            "total_speculative_time_ms": round(self.total_speculative_time_ms, 1),
            "total_baseline_time_ms": round(self.total_baseline_time_ms, 1),
        }
