"""
Quick comparison: Trie root-only n-gram vs Hash vs Suffix (spec=5).
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
            "3. Write the minimal fix.\n\n"
            "Output your patch inside a ```diff code block.\n"
        )
        if len(text) > max_prompt_len * 4:
            text = text[:max_prompt_len * 4]
        prompts.append(text)
    print(f"Prepared {len(prompts)} prompts")
    return prompts


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


def _run(prompt_texts, model_name, gpu_mem, max_tokens,
         spec_config, mode_name, num_warmup, result_queue,
         use_hash=False, use_trie=False, trie_node_size=1,
         extra_env=None):
    import torch as _torch

    os.environ["VLLM_NGRAM_USE_HASH"] = "1" if use_hash else "0"
    os.environ["VLLM_NGRAM_USE_TRIE"] = "1" if use_trie else "0"
    os.environ["VLLM_TRIE_NODE_SIZE"] = str(trie_node_size)
    os.environ["VLLM_TRIE_INTERNAL_NODE_SIZE"] = "1"
    # Apply extra env vars (for fuzzy, skipgram, etc.)
    if extra_env:
        for k, v in extra_env.items():
            os.environ[k] = str(v)

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
        print(f"\n{'='*70}")
        print(f"Mode: {mode_name}")
        print(f"{'='*70}")

        llm_kwargs = dict(
            model=model_name,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_mem,
            disable_log_stats=False,
        )
        if spec_config:
            llm_kwargs["speculative_config"] = spec_config

        llm = LLM(**llm_kwargs)
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        is_spec = spec_config is not None

        print(f"Warmup ({num_warmup} rounds) ...")
        for _ in range(num_warmup):
            llm.generate([prompt_texts[0]], sp)

        metrics_before = read_spec_metrics() if is_spec else {}

        print(f"Running {len(prompt_texts)} requests ...")
        per_request = []
        total_start = time.perf_counter()

        for idx, prompt in enumerate(prompt_texts):
            t0 = time.perf_counter()
            outputs = llm.generate([prompt], sp)
            t1 = time.perf_counter()
            latency = t1 - t0
            o = outputs[0]
            in_tok = len(o.prompt_token_ids)
            out_tok = len(o.outputs[0].token_ids)
            tps = out_tok / latency if latency > 0 else 0
            per_request.append({
                "idx": idx, "input_tokens": in_tok,
                "output_tokens": out_tok,
                "latency": round(latency, 4),
                "tokens_per_second": round(tps, 2),
            })
            if idx < 3 or (idx + 1) % 10 == 0:
                print(f"  [{idx+1}/{len(prompt_texts)}] "
                      f"in={in_tok} out={out_tok} "
                      f"lat={latency:.3f}s tps={tps:.1f}")

        total_time = time.perf_counter() - total_start

        acceptance_stats = None
        if is_spec:
            metrics_after = read_spec_metrics()
            def delta(name):
                return metrics_after.get(name, 0) - metrics_before.get(name, 0)
            num_drafts = delta("vllm:spec_decode_num_drafts")
            num_draft_tokens = delta("vllm:spec_decode_num_draft_tokens")
            num_accepted = delta("vllm:spec_decode_num_accepted_tokens")
            acc_rate = num_accepted / num_draft_tokens * 100 if num_draft_tokens > 0 else 0
            mean_len = 1 + num_accepted / num_drafts if num_drafts > 0 else 0

            per_pos_before = metrics_before.get(
                "vllm:spec_decode_num_accepted_tokens_per_pos", [])
            per_pos_after = metrics_after.get(
                "vllm:spec_decode_num_accepted_tokens_per_pos", [])
            per_pos = []
            if per_pos_after and num_drafts > 0:
                for j in range(len(per_pos_after)):
                    bv = per_pos_before[j] if j < len(per_pos_before) else 0
                    per_pos.append(round((per_pos_after[j] - bv) / num_drafts, 3))

            acceptance_stats = {
                "num_drafts": int(num_drafts),
                "num_draft_tokens": int(num_draft_tokens),
                "num_accepted_tokens": int(num_accepted),
                "acceptance_rate_pct": round(acc_rate, 2),
                "mean_accepted_length": round(mean_len, 2),
                "per_position_acceptance": per_pos,
            }

        latencies = [r["latency"] for r in per_request]
        latencies_sorted = sorted(latencies)
        total_in = sum(r["input_tokens"] for r in per_request)
        total_out = sum(r["output_tokens"] for r in per_request)
        n = len(latencies)

        result = BenchResult(
            mode=mode_name, num_prompts=n,
            total_input_tokens=total_in, total_output_tokens=total_out,
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


def run_subprocess(prompt_texts, model_name, gpu_mem, max_tokens,
                   spec_config, mode_name, num_warmup=2,
                   use_hash=False, use_trie=False, trie_node_size=1,
                   extra_env=None):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(
        target=_run,
        args=(prompt_texts, model_name, gpu_mem, max_tokens,
              spec_config, mode_name, num_warmup, q,
              use_hash, use_trie, trie_node_size, extra_env),
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


def main():
    parser = argparse.ArgumentParser(
        description="Trie root-only vs Hash vs Suffix comparison")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-prompt-len", type=int, default=2048)
    parser.add_argument("--gpu-mem", type=float, default=0.9)
    args = parser.parse_args()

    prompt_texts = build_swe_prompts(args.num_samples, args.max_prompt_len)
    all_results = []

    ngram_spec = {
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 5,
        "prompt_lookup_min": 2,
    }
    suffix_spec = {
        "method": "suffix",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 5,
        "prompt_lookup_min": 2,
    }

    # (name, spec_config, use_hash, use_trie, trie_node_size, extra_env)
    configs = [
        ("baseline",       None,        False, False, 1, None),
        ("Hash",           ngram_spec,  True,  False, 1, None),
        ("Trie-3g root",   ngram_spec,  False, True,  3, None),
        ("Trie-3g+Fuzzy",  ngram_spec,  False, True,  3,
         {"VLLM_TRIE_FUZZY": "1"}),
        ("SkipGram",       ngram_spec,  False, False, 1,
         {"VLLM_NGRAM_USE_SKIPGRAM": "1", "VLLM_NGRAM_USE_HASH": "0"}),
        ("Suffix",         suffix_spec, False, False, 1, None),
    ]

    for i, (name, sc, uh, ut, ns, ee) in enumerate(configs):
        print(f"\n>>> [{i+1}/{len(configs)}] {name} ...")
        r = run_subprocess(
            prompt_texts, args.model, args.gpu_mem, args.max_tokens,
            spec_config=sc, mode_name=name,
            use_hash=uh, use_trie=ut, trie_node_size=ns,
            extra_env=ee,
        )
        all_results.append(r)
        time.sleep(2)

    # Summary
    print("\n\n" + "=" * 120)
    print("COMPARISON: Trie/Fuzzy/SkipGram vs Hash vs Suffix  (spec=5)")
    print("=" * 120)

    valid = [r for r in all_results if "error" not in r]
    errors = [r for r in all_results if "error" in r]
    for e in errors:
        print(f"  [ERROR] {e.get('mode', '?')}: {e.get('error', '?')}")

    if not valid:
        print("All runs failed!")
        return

    base_avg = next((r["avg_latency"] for r in valid
                     if r["mode"] == "baseline"), valid[0]["avg_latency"])

    hdr = (f"{'Mode':<20} {'AvgLat':>8} {'MedLat':>8} {'P90Lat':>8} "
           f"{'tok/s':>8} {'Speedup':>8} {'Accept%':>8} {'MeanLen':>8}")
    print(hdr)
    print("-" * 100)

    for r in valid:
        lat_ratio = base_avg / r["avg_latency"] if r["avg_latency"] > 0 else 0
        acc = ""
        mlen = ""
        if r.get("acceptance_stats"):
            s = r["acceptance_stats"]
            acc = f"{s['acceptance_rate_pct']:.1f}%"
            mlen = f"{s['mean_accepted_length']:.2f}"
        print(f"{r['mode']:<20} {r['avg_latency']:>8.3f} "
              f"{r['median_latency']:>8.3f} {r['p90_latency']:>8.3f} "
              f"{r['output_tokens_per_second']:>8.2f} "
              f"{lat_ratio:>8.2f}x "
              f"{acc:>8} {mlen:>8}")

    # Save
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "compare_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "num_samples": args.num_samples,
            "max_tokens": args.max_tokens,
            "results": all_results,
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
