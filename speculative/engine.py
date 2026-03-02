"""
Speculative Decoding Engine.

Orchestrates the proposer → verify → accept/reject loop:

    while not done:
        1. proposer.propose(context, k)  →  draft[0..k-1]
        2. verifier.verify(context, draft)  →  accepted, bonus_token
        3. append accepted + bonus_token to context
        4. update proposer state
        5. record metrics

Terminates when:
    - max_new_tokens is reached, or
    - eos_token_id is generated.
"""

import time
import logging
from typing import List, Optional

from .metrics import MetricsTracker, RequestMetrics, StepMetrics
from .proposers.base import BaseProposer
from .verifier import TransformerVerifier

logger = logging.getLogger(__name__)


class SpeculativeEngine:
    """
    End-to-end speculative decoding engine.

    Args:
        proposer:              One of KMPProposer / HashTableProposer / TrieProposer.
        verifier:              TransformerVerifier wrapping Qwen-0.5B (or similar).
        num_speculative_tokens: Max draft tokens to propose per step (k).
    """

    def __init__(
        self,
        proposer: BaseProposer,
        verifier: TransformerVerifier,
        num_speculative_tokens: int = 5,
    ):
        self.proposer = proposer
        self.verifier = verifier
        self.k = num_speculative_tokens
        self._proposer_name = type(proposer).__name__

    # ------------------------------------------------------------------
    # Single-request generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        metrics: Optional[RequestMetrics] = None,
    ) -> str:
        """
        Generate text from `prompt` using speculative decoding.

        Returns:
            Generated text (excluding the prompt).
        """
        context_ids = self.verifier.encode(prompt)
        eos = self.verifier.eos_token_id
        generated: List[int] = []

        self.proposer.reset()

        while len(generated) < max_new_tokens:
            remaining = max_new_tokens - len(generated)

            # --- 1. Propose ---
            t0 = time.perf_counter()
            draft = self.proposer.propose(context_ids, min(self.k, remaining))
            propose_ms = (time.perf_counter() - t0) * 1000

            # --- 2. Verify ---
            t1 = time.perf_counter()
            accepted, bonus = self.verifier.verify(context_ids, draft)
            verify_ms = (time.perf_counter() - t1) * 1000

            # --- 3. Append accepted tokens + bonus ---
            new_tokens = accepted + [bonus]
            # Truncate if it would exceed max_new_tokens
            can_take = max_new_tokens - len(generated)
            new_tokens = new_tokens[:can_take]

            generated.extend(new_tokens)
            context_ids = context_ids + new_tokens

            # --- 4. Record metrics ---
            if metrics is not None:
                step = StepMetrics(
                    num_proposed=len(draft),
                    num_accepted=min(len(accepted), len(new_tokens) - 1),
                    bonus_token=bonus,
                    context_len=len(context_ids) - len(new_tokens),
                    proposer_name=self._proposer_name,
                    propose_time_ms=propose_ms,
                    verify_time_ms=verify_ms,
                )
                metrics.steps.append(step)
                metrics.total_propose_time_ms += propose_ms
                metrics.total_verify_time_ms += verify_ms

                # Per-position acceptance
                for pos, (d, a) in enumerate(
                    zip(draft, accepted + [None] * (len(draft) - len(accepted)))
                ):
                    while len(metrics.per_position_proposed) <= pos:
                        metrics.per_position_proposed.append(0)
                        metrics.per_position_accepted.append(0)
                    metrics.per_position_proposed[pos] += 1
                    if pos < len(accepted):
                        metrics.per_position_accepted[pos] += 1

            # --- 5. Update proposer state ---
            self.proposer.update(context_ids)

            # --- 6. Check EOS ---
            if eos in new_tokens:
                eos_pos = new_tokens.index(eos)
                # Trim generated to stop at eos
                generated = generated[: len(generated) - len(new_tokens) + eos_pos]
                break

        if metrics is not None:
            metrics.total_output_tokens = len(generated)

        return self.verifier.decode(generated)

    # ------------------------------------------------------------------
    # Baseline (pure autoregressive, no speculation)
    # ------------------------------------------------------------------

    def generate_baseline(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        metrics: Optional[RequestMetrics] = None,
    ) -> str:
        """Pure autoregressive generation for baseline measurement."""
        context_ids = self.verifier.encode(prompt)

        t0 = time.perf_counter()
        generated = self.verifier.generate_baseline(
            context_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.verifier.eos_token_id,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if metrics is not None:
            metrics.total_output_tokens = len(generated)
            metrics.total_baseline_time_ms = elapsed_ms

        return self.verifier.decode(generated)

    # ------------------------------------------------------------------
    # Benchmark helper: run both modes and return metrics
    # ------------------------------------------------------------------

    def benchmark_request(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        tracker: Optional[MetricsTracker] = None,
        run_baseline: bool = True,
    ):
        """
        Run speculative + (optionally) baseline generation for one prompt.

        Returns:
            (spec_output, spec_metrics, baseline_output, baseline_ms)
        """
        req_metrics = tracker.new_request(prompt) if tracker else None

        spec_output = self.generate(prompt, max_new_tokens, metrics=req_metrics)

        baseline_output = None
        baseline_ms = None
        if run_baseline:
            baseline_req = RequestMetrics(prompt=prompt)
            baseline_output = self.generate_baseline(
                prompt, max_new_tokens, metrics=baseline_req
            )
            baseline_ms = baseline_req.total_baseline_time_ms
            if req_metrics is not None:
                req_metrics.total_baseline_time_ms = baseline_ms

        return spec_output, req_metrics, baseline_output, baseline_ms
