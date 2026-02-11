"""
Hash Table vs KMP ngram proposer A/B benchmark on SWE-bench Lite.

Compares:
  1. Baseline (no speculative decoding)
  2. Original KMP ngram (VLLM_NGRAM_USE_HASH_TABLE=0)
  3. Hash table ngram (VLLM_NGRAM_USE_HASH_TABLE=1)

Each configuration runs in a separate subprocess to isolate GPU memory.
"""

import argparse
import json
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass
from typing import Optional

from datasets import load_dataset


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def build_swe_prompts(num_samples: int = 20,
                      max_prompt_len: int = 2048) -> list[str]:
    """Build code-fix prompts from SWE-bench Lite."""
    print("Loading SWE-bench Lite ...")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    print(f"Dataset size: {len(ds)}")

    prompts: list[str] = []
    for sample in ds:
        if len(prompts) >= num_samples:
            break
        problem = sample["problem_statement"] or ""
        repo = sample["repo"] or ""
        hints = sample["hints_text"] or ""

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
        if len(text) > max_prompt_len:
            text = text[:max_prompt_len]
        prompts.append(text)

    print(f"Built {len(prompts)} prompts")
    return prompts


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    mode: str
    num_prompts: int
    total_input_tokens: int
    total_output_tokens: int
    elapsed_time: float
    tokens_per_second: float
    avg_latency_per_prompt: float
    ngram_config: Optional[dict] = None
    acceptance_stats: Optional[dict] = None


# ---------------------------------------------------------------------------
# Worker (runs inside subprocess)
# ---------------------------------------------------------------------------

def _run_single_config(
    prompt_texts: list[str],
    model_name: str,
    gpu_mem: float,
    max_tokens: int,
    spec_config: Optional[dict],
    mode_name: str,
    use_hash_table: bool,
    num_warmup: int,
    result_queue: mp.Queue,
):
    """Run a single configuration inside a subprocess."""
    import gc
    import torch

    # Set env BEFORE importing vllm so it takes effect at init time.
    os.environ["VLLM_NGRAM_USE_HASH_TABLE"] = "1" if use_hash_table else "0"

    from vllm import LLM, SamplingParams
    from vllm.v1.metrics.reader import Counter, Vector, get_metrics_snapshot

    def read_spec_metrics():
        metrics = get_metrics_snapshot()
        out = {}
        for m in metrics:
            if "spec_decode" not in m.name:
                continue
            if isinstance(m, Counter):
                out[m.name] = m.value
            elif isinstance(m, Vector):
                out[m.name] = m.values
        return out

    try:
        print(f"\n{'=' * 70}")
        print(f"Mode: {mode_name}")
        if spec_config:
            print(f"  hash_table={use_hash_table}")
        print(f"{'=' * 70}")

        llm_kwargs = dict(
            model=model_name,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_mem,
            disable_log_stats=False,
        )
        if spec_config:
            llm_kwargs["speculative_config"] = spec_config

        llm = LLM(**llm_kwargs)
        sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        is_spec = spec_config is not None

        # Warmup
        print(f"Warmup ({num_warmup} rounds) ...")
        warmup_prompts = prompt_texts[:min(3, len(prompt_texts))]
        for _ in range(num_warmup):
            llm.generate(warmup_prompts, sampling_params)

        # Metrics snapshot (before)
        metrics_before = read_spec_metrics() if is_spec else {}

        # Inference
        print(f"Running inference on {len(prompt_texts)} prompts ...")
        start = time.perf_counter()
        outputs = llm.generate(prompt_texts, sampling_params)
        elapsed = time.perf_counter() - start

        # Metrics snapshot (after) + acceptance stats
        acceptance_stats = None
        if is_spec:
            metrics_after = read_spec_metrics()

            def delta(name):
                return metrics_after.get(name, 0) - metrics_before.get(name, 0)

            num_drafts = delta("vllm:spec_decode_num_drafts")
            num_draft_tokens = delta("vllm:spec_decode_num_draft_tokens")
            num_accepted = delta("vllm:spec_decode_num_accepted_tokens")
            acc_rate = (
                num_accepted / num_draft_tokens * 100
                if num_draft_tokens > 0 else 0
            )
            mean_len = (
                1 + num_accepted / num_drafts if num_drafts > 0 else 0
            )

            per_pos_before = metrics_before.get(
                "vllm:spec_decode_num_accepted_tokens_per_pos", [])
            per_pos_after = metrics_after.get(
                "vllm:spec_decode_num_accepted_tokens_per_pos", [])
            per_pos = []
            if per_pos_after and num_drafts > 0:
                for i in range(len(per_pos_after)):
                    bv = per_pos_before[i] if i < len(per_pos_before) else 0
                    per_pos.append(
                        round((per_pos_after[i] - bv) / num_drafts, 3))

            acceptance_stats = {
                "num_drafts": int(num_drafts),
                "num_draft_tokens": int(num_draft_tokens),
                "num_accepted_tokens": int(num_accepted),
                "acceptance_rate_pct": round(acc_rate, 2),
                "mean_accepted_length": round(mean_len, 2),
                "per_position_acceptance": per_pos,
            }

        total_in = sum(len(o.prompt_token_ids) for o in outputs)
        total_out = sum(len(o.outputs[0].token_ids) for o in outputs)
        tps = total_out / elapsed if elapsed > 0 else 0

        # Print samples
        for i, o in enumerate(outputs[:2]):
            print(
                f"[Sample {i+1}] in={len(o.prompt_token_ids)} "
                f"out={len(o.outputs[0].token_ids)}"
            )
            print(f"  {o.outputs[0].text[:200]}...\n")

        result = BenchmarkResult(
            mode=mode_name,
            num_prompts=len(prompt_texts),
            total_input_tokens=total_in,
            total_output_tokens=total_out,
            elapsed_time=round(elapsed, 3),
            tokens_per_second=round(tps, 2),
            avg_latency_per_prompt=round(elapsed / len(prompt_texts), 4),
            ngram_config=spec_config,
            acceptance_stats=acceptance_stats,
        )

        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()
        torch.cuda.empty_cache()

        result_queue.put(asdict(result))

    except Exception as e:
        import traceback
        traceback.print_exc()
        result_queue.put({"error": str(e), "mode": mode_name})


def run_config_in_subprocess(prompt_texts, model_name, gpu_mem, max_tokens,
                             spec_config, mode_name, use_hash_table=False,
                             num_warmup=2):
    """Run inference in an isolated subprocess."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(
        target=_run_single_config,
        args=(prompt_texts, model_name, gpu_mem, max_tokens,
              spec_config, mode_name, use_hash_table, num_warmup, q),
    )
    p.start()
    p.join(timeout=600)
    if p.is_alive():
        p.terminate()
        p.join()
        return {"error": "timeout", "mode": mode_name}
    if not q.empty():
        return q.get()
    return {"error": "no result returned", "mode": mode_name}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark(model_name, num_samples, max_tokens, ngram_configs,
                  max_prompt_len, gpu_mem):
    print("=" * 70)
    print("Hash Table vs KMP Ngram Proposer — A/B Benchmark")
    print(f"Model: {model_name}")
    print(f"Samples: {num_samples}  |  max_tokens: {max_tokens}")
    print(f"GPU mem utilization: {gpu_mem}")
    print("=" * 70)

    prompt_texts = build_swe_prompts(num_samples, max_prompt_len)
    all_results = []
    total_runs = 1 + len(ngram_configs) * 2
    run_idx = 0

    # 1. Baseline
    run_idx += 1
    print(f"\n>>> [{run_idx}/{total_runs}] baseline ...")
    r = run_config_in_subprocess(
        prompt_texts, model_name, gpu_mem, max_tokens,
        spec_config=None, mode_name="baseline",
    )
    all_results.append(r)
    time.sleep(2)

    # 2. Each ngram config: KMP then hash table
    for cfg in ngram_configs:
        spec_config = {
            "method": "ngram",
            "num_speculative_tokens": cfg["num_speculative_tokens"],
            "prompt_lookup_max": cfg["prompt_lookup_max"],
            "prompt_lookup_min": cfg["prompt_lookup_min"],
        }
        tag = (f"spec={cfg['num_speculative_tokens']} "
               f"n={cfg['prompt_lookup_min']}-{cfg['prompt_lookup_max']}")

        # KMP (original)
        run_idx += 1
        label_kmp = f"KMP {tag}"
        print(f"\n>>> [{run_idx}/{total_runs}] {label_kmp} ...")
        r = run_config_in_subprocess(
            prompt_texts, model_name, gpu_mem, max_tokens,
            spec_config=spec_config, mode_name=label_kmp,
            use_hash_table=False,
        )
        all_results.append(r)
        time.sleep(2)

        # Hash table
        run_idx += 1
        label_ht = f"HashTable {tag}"
        print(f"\n>>> [{run_idx}/{total_runs}] {label_ht} ...")
        r = run_config_in_subprocess(
            prompt_texts, model_name, gpu_mem, max_tokens,
            spec_config=spec_config, mode_name=label_ht,
            use_hash_table=True,
        )
        all_results.append(r)
        time.sleep(2)

    # 3. Summary
    print("\n\n" + "=" * 110)
    print("RESULTS SUMMARY")
    print("=" * 110)

    valid = [r for r in all_results if "error" not in r]
    errors = [r for r in all_results if "error" in r]
    for e in errors:
        print(f"  [ERROR] {e.get('mode', '?')}: {e.get('error', '?')}")

    if not valid:
        print("All runs failed!")
        return

    base_tps = valid[0]["tokens_per_second"] if valid else 1

    hdr = (f"{'Mode':<35} {'tok/s':>10} {'Time(s)':>8} "
           f"{'Speedup':>8} {'Accept%':>8} {'MeanLen':>8}")
    print(hdr)
    print("-" * 85)

    for r in valid:
        speedup = r["tokens_per_second"] / base_tps if base_tps > 0 else 0
        acc = ""
        mlen = ""
        if r.get("acceptance_stats"):
            s = r["acceptance_stats"]
            acc = f"{s['acceptance_rate_pct']:.1f}%"
            mlen = f"{s['mean_accepted_length']:.2f}"
        print(f"{r['mode']:<35} {r['tokens_per_second']:>10.2f} "
              f"{r['elapsed_time']:>8.3f} {speedup:>8.2f}x "
              f"{acc:>8} {mlen:>8}")

    # Detailed acceptance stats
    for r in valid:
        if r.get("acceptance_stats"):
            s = r["acceptance_stats"]
            print(f"\n--- {r['mode']} ---")
            print(f"  drafts={s['num_drafts']}  "
                  f"draft_tokens={s['num_draft_tokens']}  "
                  f"accepted={s['num_accepted_tokens']}")
            print(f"  acceptance_rate={s['acceptance_rate_pct']:.1f}%  "
                  f"mean_accepted_len={s['mean_accepted_length']:.2f}")
            if s.get("per_position_acceptance"):
                pos = ", ".join(
                    f"p{i}={v:.3f}"
                    for i, v in enumerate(s["per_position_acceptance"]))
                print(f"  per_position: {pos}")

    # Pairwise KMP vs HashTable comparison
    print("\n" + "=" * 70)
    print("KMP vs HashTable Pairwise Comparison")
    print("=" * 70)
    kmp_results = {r["mode"]: r for r in valid if r["mode"].startswith("KMP")}
    ht_results = {r["mode"]: r for r in valid
                  if r["mode"].startswith("HashTable")}
    for kmp_name, kmp_r in kmp_results.items():
        ht_name = kmp_name.replace("KMP", "HashTable")
        ht_r = ht_results.get(ht_name)
        if not ht_r:
            continue
        tag = kmp_name.replace("KMP ", "")
        kmp_tps = kmp_r["tokens_per_second"]
        ht_tps = ht_r["tokens_per_second"]
        ratio = ht_tps / kmp_tps if kmp_tps > 0 else 0
        kmp_acc = (kmp_r.get("acceptance_stats", {})
                   .get("acceptance_rate_pct", 0))
        ht_acc = (ht_r.get("acceptance_stats", {})
                  .get("acceptance_rate_pct", 0))
        print(f"  {tag}:")
        print(f"    KMP:       {kmp_tps:>8.2f} tok/s  accept={kmp_acc:.1f}%")
        print(f"    HashTable: {ht_tps:>8.2f} tok/s  accept={ht_acc:.1f}%")
        print(f"    HT/KMP:   {ratio:.3f}x throughput")

    # Save
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "hash_vs_kmp_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name,
            "num_samples": num_samples,
            "max_tokens": max_tokens,
            "results": all_results,
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Hash Table vs KMP ngram proposer A/B benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-prompt-len", type=int, default=2048)
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    args = parser.parse_args()

    ngram_configs = [
        {"num_speculative_tokens": 3,
         "prompt_lookup_max": 3, "prompt_lookup_min": 2},
        {"num_speculative_tokens": 5,
         "prompt_lookup_max": 5, "prompt_lookup_min": 2},
        {"num_speculative_tokens": 8,
         "prompt_lookup_max": 7, "prompt_lookup_min": 3},
    ]

    run_benchmark(
        model_name=args.model,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        ngram_configs=ngram_configs,
        max_prompt_len=args.max_prompt_len,
        gpu_mem=args.gpu_mem,
    )


if __name__ == "__main__":
    main()
