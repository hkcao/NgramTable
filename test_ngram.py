"""
Benchmark: KMP vs Hash vs Trie vs Suffix Decoding vs Baseline on Qwen model.

Sends single requests sequentially to measure per-request latency.
Uses VLLM_NGRAM_USE_HASH / VLLM_NGRAM_USE_TRIE env vars to switch ngram modes.
Suffix Decoding uses method="suffix" with arctic-inference backend.
"""

import argparse
import gc
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
class BenchResult:
    mode: str
    num_prompts: int
    total_input_tokens: int
    total_output_tokens: int
    total_time: float
    avg_latency: float
    median_latency: float
    p90_latency: float
    output_tokens_per_second: float
    per_request: list
    ngram_config: Optional[dict] = None
    acceptance_stats: Optional[dict] = None


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _run_single_seq(
    prompt_texts: list[str],
    model_name: str,
    gpu_mem: float,
    max_tokens: int,
    spec_config: Optional[dict],
    mode_name: str,
    use_hash: bool,
    num_warmup: int,
    result_queue: mp.Queue,
    use_trie: bool = False,
):
    import torch as _torch

    # Set the mode switch env vars BEFORE importing vllm
    os.environ["VLLM_NGRAM_USE_HASH"] = "1" if use_hash else "0"
    os.environ["VLLM_NGRAM_USE_TRIE"] = "1" if use_trie else "0"

    from vllm import LLM, SamplingParams
    from vllm.v1.metrics.reader import Counter as MCounter
    from vllm.v1.metrics.reader import Vector, get_metrics_snapshot

    def read_spec_metrics():
        metrics = get_metrics_snapshot()
        out = {}
        for m in metrics:
            if "spec_decode" not in m.name:
                continue
            if isinstance(m, MCounter):
                out[m.name] = m.value
            elif isinstance(m, Vector):
                out[m.name] = m.values
        return out

    try:
        print(f"\n{'=' * 70}")
        print(f"Mode: {mode_name}  (HASH={use_hash} TRIE={use_trie})")
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
        for _ in range(num_warmup):
            llm.generate([prompt_texts[0]], sampling_params)

        metrics_before = read_spec_metrics() if is_spec else {}

        # Sequential single-request inference
        print(f"Running {len(prompt_texts)} single requests sequentially ...")
        per_request = []
        total_start = time.perf_counter()

        for idx, prompt in enumerate(prompt_texts):
            t0 = time.perf_counter()
            outputs = llm.generate([prompt], sampling_params)
            t1 = time.perf_counter()

            latency = t1 - t0
            o = outputs[0]
            in_tok = len(o.prompt_token_ids)
            out_tok = len(o.outputs[0].token_ids)
            tps = out_tok / latency if latency > 0 else 0

            per_request.append({
                "idx": idx,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "latency": round(latency, 4),
                "tokens_per_second": round(tps, 2),
            })

            if idx < 3 or (idx + 1) % 10 == 0:
                print(f"  [{idx+1}/{len(prompt_texts)}] "
                      f"in={in_tok} out={out_tok} "
                      f"lat={latency:.3f}s tps={tps:.1f}")

        total_time = time.perf_counter() - total_start

        # Acceptance metrics
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
                for j in range(len(per_pos_after)):
                    bv = per_pos_before[j] if j < len(per_pos_before) else 0
                    per_pos.append(
                        round((per_pos_after[j] - bv) / num_drafts, 3))

            acceptance_stats = {
                "num_drafts": int(num_drafts),
                "num_draft_tokens": int(num_draft_tokens),
                "num_accepted_tokens": int(num_accepted),
                "acceptance_rate_pct": round(acc_rate, 2),
                "mean_accepted_length": round(mean_len, 2),
                "per_position_acceptance": per_pos,
            }

        # Aggregate
        latencies = [r["latency"] for r in per_request]
        latencies_sorted = sorted(latencies)
        total_in = sum(r["input_tokens"] for r in per_request)
        total_out = sum(r["output_tokens"] for r in per_request)
        n = len(latencies)

        result = BenchResult(
            mode=mode_name,
            num_prompts=n,
            total_input_tokens=total_in,
            total_output_tokens=total_out,
            total_time=round(total_time, 3),
            avg_latency=round(sum(latencies) / n, 4),
            median_latency=round(latencies_sorted[n // 2], 4),
            p90_latency=round(latencies_sorted[int(n * 0.9)], 4),
            output_tokens_per_second=round(total_out / total_time, 2),
            per_request=per_request,
            ngram_config=spec_config,
            acceptance_stats=acceptance_stats,
        )

        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()
        _torch.cuda.empty_cache()

        result_queue.put(asdict(result))

    except Exception as e:
        import traceback
        traceback.print_exc()
        result_queue.put({"error": str(e), "mode": mode_name})


def run_config_in_subprocess(prompt_texts, model_name, gpu_mem, max_tokens,
                             spec_config, mode_name, use_hash=False,
                             num_warmup=2, use_trie=False):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(
        target=_run_single_seq,
        args=(prompt_texts, model_name, gpu_mem, max_tokens,
              spec_config, mode_name, use_hash, num_warmup, q, use_trie),
    )
    p.start()
    p.join(timeout=1200)
    if p.is_alive():
        p.terminate()
        p.join()
        return {"error": "timeout", "mode": mode_name}
    if not q.empty():
        return q.get()
    return {"error": "no result returned", "mode": mode_name}


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(model_name, num_samples, max_tokens, ngram_configs,
                  max_prompt_len, gpu_mem):
    print("=" * 70)
    print("KMP vs Hash vs Trie vs Suffix Decoding Benchmark")
    print(f"Model: {model_name}")
    print(f"Samples: {num_samples}  |  max_tokens: {max_tokens}")
    print(f"GPU mem utilization: {gpu_mem}")
    print("=" * 70)

    prompt_texts = build_swe_prompts(num_samples, max_prompt_len)
    all_results = []
    # baseline + 3 ngram modes per config + 1 suffix per config
    total_runs = 1 + len(ngram_configs) * 4
    run_idx = 0

    # 1. Baseline (no speculative decoding)
    run_idx += 1
    print(f"\n>>> [{run_idx}/{total_runs}] baseline ...")
    r = run_config_in_subprocess(
        prompt_texts, model_name, gpu_mem, max_tokens,
        spec_config=None, mode_name="baseline",
    )
    all_results.append(r)
    time.sleep(2)

    # 2. For each ngram config: KMP, Hash, Trie, then Suffix
    for cfg in ngram_configs:
        spec_config = {
            "method": "ngram",
            "num_speculative_tokens": cfg["num_speculative_tokens"],
            "prompt_lookup_max": cfg["prompt_lookup_max"],
            "prompt_lookup_min": cfg["prompt_lookup_min"],
        }
        tag = (f"spec={cfg['num_speculative_tokens']} "
               f"n={cfg['prompt_lookup_min']}-{cfg['prompt_lookup_max']}")

        # KMP mode
        run_idx += 1
        label_kmp = f"KMP {tag}"
        print(f"\n>>> [{run_idx}/{total_runs}] {label_kmp} ...")
        r = run_config_in_subprocess(
            prompt_texts, model_name, gpu_mem, max_tokens,
            spec_config=spec_config, mode_name=label_kmp,
            use_hash=False, use_trie=False,
        )
        all_results.append(r)
        time.sleep(2)

        # Hash mode
        run_idx += 1
        label_ht = f"Hash {tag}"
        print(f"\n>>> [{run_idx}/{total_runs}] {label_ht} ...")
        r = run_config_in_subprocess(
            prompt_texts, model_name, gpu_mem, max_tokens,
            spec_config=spec_config, mode_name=label_ht,
            use_hash=True, use_trie=False,
        )
        all_results.append(r)
        time.sleep(2)

        # Trie mode
        run_idx += 1
        label_trie = f"Trie {tag}"
        print(f"\n>>> [{run_idx}/{total_runs}] {label_trie} ...")
        r = run_config_in_subprocess(
            prompt_texts, model_name, gpu_mem, max_tokens,
            spec_config=spec_config, mode_name=label_trie,
            use_hash=False, use_trie=True,
        )
        all_results.append(r)
        time.sleep(2)

        # Suffix Decoding mode (uses same num_speculative_tokens for
        # fair comparison)
        run_idx += 1
        label_suffix = f"Suffix {tag}"
        print(f"\n>>> [{run_idx}/{total_runs}] {label_suffix} ...")
        suffix_spec_config = {
            "method": "suffix",
            "num_speculative_tokens": cfg["num_speculative_tokens"],
        }
        r = run_config_in_subprocess(
            prompt_texts, model_name, gpu_mem, max_tokens,
            spec_config=suffix_spec_config, mode_name=label_suffix,
            use_hash=False, use_trie=False,
        )
        all_results.append(r)
        time.sleep(2)

    # 3. Summary
    print("\n\n" + "=" * 130)
    print("RESULTS SUMMARY")
    print("=" * 130)

    valid = [r for r in all_results if "error" not in r]
    errors = [r for r in all_results if "error" in r]
    for e in errors:
        print(f"  [ERROR] {e.get('mode', '?')}: {e.get('error', '?')}")

    if not valid:
        print("All runs failed!")
        return

    base_tps = valid[0].get("output_tokens_per_second", 1)
    base_avg = valid[0].get("avg_latency", 1)

    hdr = (f"{'Mode':<35} {'AvgLat':>8} {'MedLat':>8} {'P90Lat':>8} "
           f"{'tok/s':>8} {'Speedup':>8} {'Accept%':>8} {'MeanLen':>8}")
    print(hdr)
    print("-" * 120)

    for r in valid:
        lat_ratio = base_avg / r["avg_latency"] if r["avg_latency"] > 0 else 0
        acc = ""
        mlen = ""
        if r.get("acceptance_stats"):
            s = r["acceptance_stats"]
            acc = f"{s['acceptance_rate_pct']:.1f}%"
            mlen = f"{s['mean_accepted_length']:.2f}"
        print(f"{r['mode']:<35} {r['avg_latency']:>8.3f} "
              f"{r['median_latency']:>8.3f} {r['p90_latency']:>8.3f} "
              f"{r['output_tokens_per_second']:>8.2f} "
              f"{lat_ratio:>8.2f}x "
              f"{acc:>8} {mlen:>8}")

    # Pairwise comparison: KMP vs Hash vs Trie vs Suffix
    print("\n" + "=" * 100)
    print("KMP vs Hash vs Trie vs Suffix Pairwise Comparison")
    print("=" * 100)
    kmp_results = {r["mode"]: r for r in valid if r["mode"].startswith("KMP")}
    ht_results = {r["mode"]: r for r in valid if r["mode"].startswith("Hash")}
    trie_results = {r["mode"]: r for r in valid if r["mode"].startswith("Trie")}
    suffix_results = {r["mode"]: r for r in valid
                      if r["mode"].startswith("Suffix")}

    for kmp_name, kmp_r in kmp_results.items():
        ht_name = kmp_name.replace("KMP", "Hash")
        trie_name = kmp_name.replace("KMP", "Trie")
        suffix_name = kmp_name.replace("KMP", "Suffix")
        ht_r = ht_results.get(ht_name)
        trie_r = trie_results.get(trie_name)
        suffix_r = suffix_results.get(suffix_name)
        tag = kmp_name.replace("KMP ", "")

        print(f"\n  {tag}:")

        entries = [("KMP", kmp_r)]
        if ht_r:
            entries.append(("Hash", ht_r))
        if trie_r:
            entries.append(("Trie", trie_r))
        if suffix_r:
            entries.append(("Suffix", suffix_r))

        for label, r in entries:
            avg = r["avg_latency"]
            tps = r["output_tokens_per_second"]
            acc = (r.get("acceptance_stats", {})
                   .get("acceptance_rate_pct", 0))
            mlen = (r.get("acceptance_stats", {})
                    .get("mean_accepted_length", 0))
            print(f"    {label:<8}: avg_lat={avg:.3f}s  "
                  f"tps={tps:.1f}  accept={acc:.1f}%  "
                  f"mean_len={mlen:.2f}")

        # Ratios vs KMP
        kmp_avg = kmp_r["avg_latency"]
        kmp_tps = kmp_r["output_tokens_per_second"]
        for label, r in entries[1:]:
            r_avg = r["avg_latency"]
            r_tps = r["output_tokens_per_second"]
            if r_avg > 0 and kmp_tps > 0:
                print(f"    {label}/KMP:  latency "
                      f"{kmp_avg / r_avg:.3f}x  throughput "
                      f"{r_tps / kmp_tps:.3f}x")

    # Save results
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "benchmark_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name,
            "num_samples": num_samples,
            "max_tokens": max_tokens,
            "mode": "kmp_vs_hash_vs_trie_vs_suffix",
            "results": all_results,
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="KMP vs Hash vs Trie vs Suffix benchmark on Qwen")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-prompt-len", type=int, default=2048)
    parser.add_argument("--gpu-mem", type=float, default=0.9)
    args = parser.parse_args()

    ngram_configs = [
        {"num_speculative_tokens": 3,
         "prompt_lookup_max": 3, "prompt_lookup_min": 2},
        {"num_speculative_tokens": 5,
         "prompt_lookup_max": 5, "prompt_lookup_min": 2},
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
