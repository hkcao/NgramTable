"""
Batch-mode benchmark: Hash vs KMP — focus on hit rate & acceptance rate.

Runs twice to verify reproducibility (temp=0).
"""

import argparse
import gc
import json
import multiprocessing as mp
import os
import time
from datetime import datetime

from datasets import load_dataset


def build_swe_prompts(num_samples: int = 50,
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


def _run_batch(
    prompt_texts: list[str],
    model_name: str,
    gpu_mem: float,
    max_tokens: int,
    spec_config: dict | None,
    mode_name: str,
    use_hash: bool,
    num_warmup: int,
    result_queue: mp.Queue,
):
    import torch as _torch

    os.environ["VLLM_NGRAM_USE_HASH"] = "1" if use_hash else "0"

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
        print(f"Mode: {mode_name}  (VLLM_NGRAM_USE_HASH={use_hash})")
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

        # Warmup with a small batch
        print(f"Warmup ({num_warmup} rounds) ...")
        for _ in range(num_warmup):
            llm.generate(prompt_texts[:2], sampling_params)

        # Reset metrics after warmup
        metrics_before = read_spec_metrics() if is_spec else {}

        # Batch inference
        print(f"Running batch of {len(prompt_texts)} prompts ...")
        t0 = time.perf_counter()
        outputs = llm.generate(prompt_texts, sampling_params)
        t1 = time.perf_counter()
        batch_time = t1 - t0

        # Collect per-request info
        total_in = 0
        total_out = 0
        per_request = []
        for idx, o in enumerate(outputs):
            in_tok = len(o.prompt_token_ids)
            out_tok = len(o.outputs[0].token_ids)
            total_in += in_tok
            total_out += out_tok
            per_request.append({
                "idx": idx,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
            })

        # Acceptance metrics
        acceptance_stats = None
        if is_spec:
            metrics_after = read_spec_metrics()

            def delta(name):
                return metrics_after.get(name, 0) - metrics_before.get(name, 0)

            num_drafts = delta("vllm:spec_decode_num_drafts")
            num_draft_tokens = delta("vllm:spec_decode_num_draft_tokens")
            num_accepted = delta("vllm:spec_decode_num_accepted_tokens")
            num_emitted = delta("vllm:spec_decode_num_emitted_tokens")

            acc_rate = (
                num_accepted / num_draft_tokens * 100
                if num_draft_tokens > 0 else 0
            )
            mean_len = (
                1 + num_accepted / num_drafts if num_drafts > 0 else 0
            )
            # hit rate = drafts that proposed at least 1 token / total drafts
            # approximation: if num_draft_tokens > 0, hit rate ~ num_draft_tokens / (num_drafts * k)
            # But more accurate: per-position acceptance
            per_pos_before = metrics_before.get(
                "vllm:spec_decode_num_accepted_tokens_per_pos", [])
            per_pos_after = metrics_after.get(
                "vllm:spec_decode_num_accepted_tokens_per_pos", [])
            per_pos = []
            if per_pos_after and num_drafts > 0:
                for j in range(len(per_pos_after)):
                    bv = per_pos_before[j] if j < len(per_pos_before) else 0
                    per_pos.append(
                        round((per_pos_after[j] - bv) / num_drafts, 4))

            # Draft hit rate: fraction of draft rounds that proposed > 0 tokens
            # pos[0] acceptance = fraction where at least 1st draft was accepted
            draft_hit_rate = per_pos[0] if per_pos else 0.0

            acceptance_stats = {
                "num_drafts": int(num_drafts),
                "num_draft_tokens": int(num_draft_tokens),
                "num_accepted_tokens": int(num_accepted),
                "num_emitted_tokens": int(num_emitted),
                "acceptance_rate_pct": round(acc_rate, 2),
                "mean_accepted_length": round(mean_len, 2),
                "draft_hit_rate": round(draft_hit_rate, 4),
                "per_position_acceptance": per_pos,
            }

        result = {
            "mode": mode_name,
            "num_prompts": len(prompt_texts),
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "batch_time": round(batch_time, 3),
            "throughput_tok_per_s": round(total_out / batch_time, 2),
            "per_request": per_request,
            "acceptance_stats": acceptance_stats,
        }

        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()
        _torch.cuda.empty_cache()

        result_queue.put(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        result_queue.put({"error": str(e), "mode": mode_name})


def run_in_subprocess(prompt_texts, model_name, gpu_mem, max_tokens,
                      spec_config, mode_name, use_hash=False, num_warmup=2):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(
        target=_run_batch,
        args=(prompt_texts, model_name, gpu_mem, max_tokens,
              spec_config, mode_name, use_hash, num_warmup, q),
    )
    p.start()
    p.join(timeout=1800)
    if p.is_alive():
        p.terminate()
        p.join()
        return {"error": "timeout", "mode": mode_name}
    if not q.empty():
        return q.get()
    return {"error": "no result returned", "mode": mode_name}


def print_summary(all_results: list[dict], run_label: str):
    print(f"\n{'=' * 120}")
    print(f"  {run_label}")
    print(f"{'=' * 120}")

    valid = [r for r in all_results if "error" not in r]
    errors = [r for r in all_results if "error" in r]
    for e in errors:
        print(f"  [ERROR] {e.get('mode', '?')}: {e.get('error', '?')}")

    if not valid:
        print("  All runs failed!")
        return

    hdr = (f"  {'Mode':<35} {'Time':>7} {'tok/s':>8} "
           f"{'Accept%':>9} {'MeanLen':>9} {'HitRate':>9} "
           f"{'Drafts':>8} {'DraftTok':>10} {'Accepted':>10}")
    print(hdr)
    print("  " + "-" * 115)

    for r in valid:
        s = r.get("acceptance_stats")
        if s:
            print(f"  {r['mode']:<35} {r['batch_time']:>7.1f} "
                  f"{r['throughput_tok_per_s']:>8.1f} "
                  f"{s['acceptance_rate_pct']:>8.1f}% "
                  f"{s['mean_accepted_length']:>9.2f} "
                  f"{s['draft_hit_rate']:>9.4f} "
                  f"{s['num_drafts']:>8} "
                  f"{s['num_draft_tokens']:>10} "
                  f"{s['num_accepted_tokens']:>10}")
            if s.get("per_position_acceptance"):
                pos_str = "  ".join(
                    f"p{j}={v:.4f}"
                    for j, v in enumerate(s["per_position_acceptance"]))
                print(f"  {'':35} Per-pos: {pos_str}")
        else:
            print(f"  {r['mode']:<35} {r['batch_time']:>7.1f} "
                  f"{r['throughput_tok_per_s']:>8.1f} "
                  f"{'N/A':>9} {'N/A':>9} {'N/A':>9}")


def run_benchmark(model_name, num_samples, max_tokens, ngram_configs,
                  max_prompt_len, gpu_mem, num_runs):
    print("=" * 70)
    print("Batch Acceptance Rate Benchmark (Hash vs KMP)")
    print(f"Model: {model_name}")
    print(f"Samples: {num_samples}  |  max_tokens: {max_tokens}")
    print(f"GPU mem: {gpu_mem}  |  Runs: {num_runs}")
    print("=" * 70)

    prompt_texts = build_swe_prompts(num_samples, max_prompt_len)
    all_run_results = []

    for run_id in range(1, num_runs + 1):
        print(f"\n{'#' * 70}")
        print(f"# RUN {run_id}/{num_runs}")
        print(f"{'#' * 70}")

        run_results = []

        for cfg in ngram_configs:
            spec_config = {
                "method": "ngram",
                "num_speculative_tokens": cfg["num_speculative_tokens"],
                "prompt_lookup_max": cfg["prompt_lookup_max"],
                "prompt_lookup_min": cfg["prompt_lookup_min"],
            }
            tag = (f"spec={cfg['num_speculative_tokens']} "
                   f"n={cfg['prompt_lookup_min']}-{cfg['prompt_lookup_max']}")

            # KMP
            label_kmp = f"KMP {tag}"
            print(f"\n>>> {label_kmp} ...")
            r = run_in_subprocess(
                prompt_texts, model_name, gpu_mem, max_tokens,
                spec_config=spec_config, mode_name=label_kmp,
                use_hash=False,
            )
            run_results.append(r)
            time.sleep(3)

            # Hash
            label_ht = f"Hash {tag}"
            print(f"\n>>> {label_ht} ...")
            r = run_in_subprocess(
                prompt_texts, model_name, gpu_mem, max_tokens,
                spec_config=spec_config, mode_name=label_ht,
                use_hash=True,
            )
            run_results.append(r)
            time.sleep(3)

        print_summary(run_results, f"RUN {run_id} RESULTS")
        all_run_results.append(run_results)

    # Cross-run reproducibility comparison
    if num_runs >= 2:
        print(f"\n{'=' * 120}")
        print("  REPRODUCIBILITY CHECK (Run 1 vs Run 2)")
        print(f"{'=' * 120}")
        for idx in range(len(all_run_results[0])):
            r1 = all_run_results[0][idx]
            r2 = all_run_results[1][idx]
            if "error" in r1 or "error" in r2:
                continue
            s1 = r1.get("acceptance_stats", {})
            s2 = r2.get("acceptance_stats", {})
            acc1 = s1.get("acceptance_rate_pct", 0)
            acc2 = s2.get("acceptance_rate_pct", 0)
            ml1 = s1.get("mean_accepted_length", 0)
            ml2 = s2.get("mean_accepted_length", 0)
            hr1 = s1.get("draft_hit_rate", 0)
            hr2 = s2.get("draft_hit_rate", 0)
            print(f"  {r1['mode']:<35}"
                  f"  Accept: {acc1:.2f}% vs {acc2:.2f}% "
                  f"(delta={abs(acc1-acc2):.2f})"
                  f"  MeanLen: {ml1:.2f} vs {ml2:.2f} "
                  f"(delta={abs(ml1-ml2):.2f})"
                  f"  HitRate: {hr1:.4f} vs {hr2:.4f}")

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"batch_acceptance_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name,
            "num_samples": num_samples,
            "max_tokens": max_tokens,
            "gpu_mem": gpu_mem,
            "num_runs": num_runs,
            "runs": all_run_results,
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch acceptance rate benchmark: Hash vs KMP")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-prompt-len", type=int, default=2048)
    parser.add_argument("--gpu-mem", type=float, default=0.8)
    parser.add_argument("--num-runs", type=int, default=2)
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
        num_runs=args.num_runs,
    )


if __name__ == "__main__":
    main()
