"""
Transformer-based Speculative Decoding Benchmark on SWE-bench Lite.

Compares three draft proposers:
  1. KMP      — longest-match n-gram scan (O(n) per query)
  2. HashTable — frequency-driven hash table (O(1) per query)
  3. Trie      — token-level trie structure  (O(max_n) per query)

Each proposer is paired with a Qwen-0.5B transformer verifier.

No vLLM required — runs on macOS (MPS), Linux (CUDA), or CPU.

Usage:
    python benchmark_transformer.py --num-samples 20 --max-new-tokens 256
    python benchmark_transformer.py --proposers kmp hash trie --num-samples 50
"""

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from datasets import load_dataset

from speculative import MetricsTracker, SpeculativeEngine, TransformerVerifier
from speculative.proposers import HashTableProposer, KMPProposer, TrieProposer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def build_swe_prompts(
    num_samples: int = 20,
    max_prompt_chars: int = 4096,
) -> List[str]:
    """Build code-fix prompts from SWE-bench Lite."""
    logger.info("Loading SWE-bench Lite ...")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    logger.info("Dataset size: %d", len(ds))

    prompts: List[str] = []
    for sample in ds:
        if len(prompts) >= num_samples:
            break
        problem = sample.get("problem_statement") or ""
        repo = sample.get("repo") or ""
        hints = sample.get("hints_text") or ""

        text = (
            f"You are a software engineer fixing a bug in `{repo}`.\n\n"
            f"## Problem Description\n\n{problem}\n\n"
        )
        if hints:
            text += f"## Hints\n\n{hints}\n\n"
        text += (
            "## Task\n\n"
            "Analyze the bug and provide a fix as a unified diff patch. "
            "Think step by step:\n"
            "1. Identify the root cause.\n"
            "2. Determine which file(s) to modify.\n"
            "3. Provide the patch.\n\n"
            "```diff\n"
        )
        if len(text) > max_prompt_chars:
            text = text[:max_prompt_chars]
        prompts.append(text)

    logger.info("Built %d prompts", len(prompts))
    return prompts


# ---------------------------------------------------------------------------
# Proposer factory
# ---------------------------------------------------------------------------

_PROPOSER_REGISTRY = {
    "kmp": KMPProposer,
    "hash": HashTableProposer,
    "trie": TrieProposer,
}


def make_proposer(name: str, min_n: int, max_n: int):
    cls = _PROPOSER_REGISTRY.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown proposer '{name}'. Choose from: {list(_PROPOSER_REGISTRY)}")
    return cls(min_n=min_n, max_n=max_n)


# ---------------------------------------------------------------------------
# Single-proposer benchmark
# ---------------------------------------------------------------------------

@dataclass
class ProposerResult:
    proposer_name: str
    num_speculative_tokens: int
    min_n: int
    max_n: int
    num_samples: int
    max_new_tokens: int
    # Timing
    total_speculative_time_s: float = 0.0
    total_baseline_time_s: float = 0.0
    wall_clock_s: float = 0.0
    # Metrics summary
    metrics_summary: Dict = field(default_factory=dict)
    # Per-request details
    per_request: List[Dict] = field(default_factory=list)


def run_proposer_benchmark(
    proposer_name: str,
    prompts: List[str],
    verifier: TransformerVerifier,
    num_speculative_tokens: int,
    min_n: int,
    max_n: int,
    max_new_tokens: int,
    run_baseline: bool,
) -> ProposerResult:
    proposer = make_proposer(proposer_name, min_n=min_n, max_n=max_n)
    engine = SpeculativeEngine(
        proposer=proposer,
        verifier=verifier,
        num_speculative_tokens=num_speculative_tokens,
    )
    tracker = MetricsTracker(
        proposer_name=proposer_name,
        num_speculative_tokens=num_speculative_tokens,
    )

    result = ProposerResult(
        proposer_name=proposer_name,
        num_speculative_tokens=num_speculative_tokens,
        min_n=min_n,
        max_n=max_n,
        num_samples=len(prompts),
        max_new_tokens=max_new_tokens,
    )

    wall_start = time.perf_counter()

    for idx, prompt in enumerate(prompts):
        logger.info(
            "[%s] sample %d/%d ...", proposer_name, idx + 1, len(prompts)
        )
        spec_out, req_metrics, _, _ = engine.benchmark_request(
            prompt,
            max_new_tokens=max_new_tokens,
            tracker=tracker,
            run_baseline=run_baseline,
        )

        if req_metrics:
            req_dict = req_metrics.to_dict()
            req_dict["sample_idx"] = idx
            req_dict["output_preview"] = spec_out[:120]
            result.per_request.append(req_dict)

            logger.info(
                "  accept_rate=%.1f%%  mean_len=%.2f  speedup=%.2fx",
                req_dict["token_acceptance_rate_pct"],
                req_dict["mean_accepted_length"],
                req_dict["speedup_ratio"],
            )

    result.wall_clock_s = time.perf_counter() - wall_start
    result.total_speculative_time_s = tracker.total_speculative_time_ms / 1000
    result.total_baseline_time_s = tracker.total_baseline_time_ms / 1000
    result.metrics_summary = tracker.summary()

    return result


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(
    verifier_model: str,
    proposers: List[str],
    num_samples: int,
    max_new_tokens: int,
    max_prompt_chars: int,
    num_speculative_tokens: int,
    min_n: int,
    max_n: int,
    run_baseline: bool,
    output_path: str,
):
    print("=" * 80)
    print("Transformer-based Speculative Decoding Benchmark")
    print(f"Verifier model: {verifier_model}")
    print(f"Proposers:       {proposers}")
    print(f"Samples:         {num_samples}  |  max_new_tokens: {max_new_tokens}")
    print(f"Spec tokens:     {num_speculative_tokens}  |  n-gram: {min_n}..{max_n}")
    print("=" * 80)

    prompts = build_swe_prompts(num_samples, max_prompt_chars)

    # Load verifier once; share across all proposer runs
    logger.info("Loading verifier: %s", verifier_model)
    verifier = TransformerVerifier(model_name=verifier_model)

    all_results: List[ProposerResult] = []

    for pname in proposers:
        print(f"\n>>> Benchmarking proposer: {pname.upper()} ...")
        res = run_proposer_benchmark(
            proposer_name=pname,
            prompts=prompts,
            verifier=verifier,
            num_speculative_tokens=num_speculative_tokens,
            min_n=min_n,
            max_n=max_n,
            max_new_tokens=max_new_tokens,
            run_baseline=run_baseline,
        )
        all_results.append(res)
        _print_result(res)

    _print_comparison(all_results)
    _save_results(all_results, output_path)


def _print_result(res: ProposerResult) -> None:
    s = res.metrics_summary
    print(f"\n  --- {res.proposer_name.upper()} ---")
    print(f"  draft_hit_rate:         {s.get('draft_hit_rate_pct', 0):.1f}%")
    print(f"  token_acceptance_rate:  {s.get('token_acceptance_rate_pct', 0):.1f}%")
    print(f"  mean_accepted_length:   {s.get('mean_accepted_length', 0):.2f} tokens/step")
    print(f"  overall_speedup:        {s.get('overall_speedup', 0):.2f}x")
    print(f"  wall_clock:             {res.wall_clock_s:.1f}s")


def _print_comparison(results: List[ProposerResult]) -> None:
    if len(results) < 2:
        return
    print("\n" + "=" * 80)
    print("PROPOSER COMPARISON SUMMARY")
    print("=" * 80)
    hdr = (
        f"{'Proposer':<15} {'Hit%':>7} {'Accept%':>8} "
        f"{'MeanLen':>9} {'Speedup':>9} {'Wall(s)':>9}"
    )
    print(hdr)
    print("-" * 60)
    for res in results:
        s = res.metrics_summary
        print(
            f"{res.proposer_name:<15} "
            f"{s.get('draft_hit_rate_pct', 0):>7.1f} "
            f"{s.get('token_acceptance_rate_pct', 0):>8.1f} "
            f"{s.get('mean_accepted_length', 0):>9.2f} "
            f"{s.get('overall_speedup', 0):>9.2f}x "
            f"{res.wall_clock_s:>9.1f}"
        )


def _save_results(results: List[ProposerResult], output_path: str) -> None:
    data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": [asdict(r) for r in results],
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Transformer-based Speculative Decoding Benchmark"
    )
    parser.add_argument(
        "--verifier-model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model name for the verifier (default: Qwen2.5-0.5B-Instruct)",
    )
    parser.add_argument(
        "--proposers",
        nargs="+",
        default=["kmp", "hash", "trie"],
        choices=["kmp", "hash", "trie"],
        help="Proposer(s) to benchmark (default: all three)",
    )
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-prompt-chars", type=int, default=4096)
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=5,
        help="Draft tokens proposed per step (k)",
    )
    parser.add_argument("--min-n", type=int, default=2, help="Min n-gram length")
    parser.add_argument("--max-n", type=int, default=5, help="Max n-gram length")
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip baseline autoregressive generation (faster)",
    )
    parser.add_argument(
        "--output",
        default="results/transformer_benchmark.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    run_benchmark(
        verifier_model=args.verifier_model,
        proposers=args.proposers,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        max_prompt_chars=args.max_prompt_chars,
        num_speculative_tokens=args.num_speculative_tokens,
        min_n=args.min_n,
        max_n=args.max_n,
        run_baseline=not args.no_baseline,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
