"""
Single-request sequential benchmark to verify perfect reproducibility with temp=0.

Sends one prompt at a time — no batch scheduling non-determinism.
Runs twice and compares acceptance stats token-for-token.
"""

import argparse
import gc
import json
import multiprocessing as mp
import os
import time
from datetime import datetime

from datasets import load_dataset


def build_swe_prompts(num_samples: int = 20,
                      max_prompt_len: int = 2048) -> list[str]:
    print("Loading SWE-bench Lite ...")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
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


def _run_single_seq(
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

        # Warmup
        print(f"Warmup ({num_warmup} rounds) ...")
        for _ in range(num_warmup):
            llm.generate([prompt_texts[0]], sampling_params)

        metrics_before = read_spec_metrics() if is_spec else {}

        # Single-request sequential inference
        print(f"Running {len(prompt_texts)} single requests sequentially ...")
        per_request = []
        output_hashes = []
        total_start = time.perf_counter()

        for idx, prompt in enumerate(prompt_texts):
            t0 = time.perf_counter()
            outputs = llm.generate([prompt], sampling_params)
            t1 = time.perf_counter()

            o = outputs[0]
            in_tok = len(o.prompt_token_ids)
            out_tok = len(o.outputs[0].token_ids)
            out_ids = list(o.outputs[0].token_ids)
            latency = t1 - t0

            per_request.append({
                "idx": idx,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "latency": round(latency, 4),
                "output_token_ids": out_ids,
            })
            output_hashes.append(hash(tuple(out_ids)))

            if idx < 3 or (idx + 1) % 10 == 0:
                print(f"  [{idx+1}/{len(prompt_texts)}] "
                      f"in={in_tok} out={out_tok} lat={latency:.3f}s")

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
                        round((per_pos_after[j] - bv) / num_drafts, 4))

            draft_hit_rate = per_pos[0] if per_pos else 0.0

            acceptance_stats = {
                "num_drafts": int(num_drafts),
                "num_draft_tokens": int(num_draft_tokens),
                "num_accepted_tokens": int(num_accepted),
                "acceptance_rate_pct": round(acc_rate, 2),
                "mean_accepted_length": round(mean_len, 2),
                "draft_hit_rate": round(draft_hit_rate, 4),
                "per_position_acceptance": per_pos,
            }

        total_out = sum(r["output_tokens"] for r in per_request)
        result = {
            "mode": mode_name,
            "num_prompts": len(prompt_texts),
            "total_time": round(total_time, 3),
            "throughput_tok_per_s": round(total_out / total_time, 2),
            "acceptance_stats": acceptance_stats,
            "output_hashes": output_hashes,
            "per_request": per_request,
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
        target=_run_single_seq,
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


def main():
    parser = argparse.ArgumentParser(
        description="Single-request reproducibility check")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-prompt-len", type=int, default=2048)
    parser.add_argument("--gpu-mem", type=float, default=0.8)
    args = parser.parse_args()

    prompt_texts = build_swe_prompts(args.num_samples, args.max_prompt_len)

    spec_config = {
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 5,
        "prompt_lookup_min": 2,
    }

    configs = [
        ("KMP_run1", False),
        ("KMP_run2", False),
        ("Hash_run1", True),
        ("Hash_run2", True),
    ]

    all_results = {}
    for mode_name, use_hash in configs:
        print(f"\n>>> {mode_name} ...")
        r = run_in_subprocess(
            prompt_texts, args.model, args.gpu_mem, args.max_tokens,
            spec_config=spec_config, mode_name=mode_name,
            use_hash=use_hash,
        )
        all_results[mode_name] = r
        time.sleep(3)

    # Summary
    print("\n" + "=" * 100)
    print("RESULTS")
    print("=" * 100)
    for name, r in all_results.items():
        if "error" in r:
            print(f"  {name}: ERROR - {r['error']}")
            continue
        s = r.get("acceptance_stats", {})
        print(f"  {name:<15} Accept={s.get('acceptance_rate_pct',0):>6.2f}%  "
              f"MeanLen={s.get('mean_accepted_length',0):>5.2f}  "
              f"HitRate={s.get('draft_hit_rate',0):>.4f}  "
              f"Drafts={s.get('num_drafts',0)}  "
              f"Accepted={s.get('num_accepted_tokens',0)}")

    # Reproducibility: compare output token IDs
    print("\n" + "=" * 100)
    print("OUTPUT REPRODUCIBILITY (token-level)")
    print("=" * 100)
    for pair_label, name_a, name_b in [
        ("KMP run1 vs run2", "KMP_run1", "KMP_run2"),
        ("Hash run1 vs run2", "Hash_run1", "Hash_run2"),
        ("KMP vs Hash (run1)", "KMP_run1", "Hash_run1"),
    ]:
        ra = all_results.get(name_a, {})
        rb = all_results.get(name_b, {})
        if "error" in ra or "error" in rb:
            print(f"  {pair_label}: SKIPPED (error)")
            continue

        ha = ra.get("output_hashes", [])
        hb = rb.get("output_hashes", [])
        n = min(len(ha), len(hb))
        match = sum(1 for i in range(n) if ha[i] == hb[i])
        print(f"  {pair_label}: {match}/{n} identical outputs "
              f"({'PERFECT' if match == n else 'MISMATCH'})")

        # Show first few mismatches
        if match < n:
            mismatches = [i for i in range(n) if ha[i] != hb[i]]
            for idx in mismatches[:3]:
                pa = ra["per_request"][idx]
                pb = rb["per_request"][idx]
                ids_a = pa["output_token_ids"]
                ids_b = pb["output_token_ids"]
                # Find first divergence point
                div = 0
                for j in range(min(len(ids_a), len(ids_b))):
                    if ids_a[j] != ids_b[j]:
                        div = j
                        break
                print(f"    prompt[{idx}]: diverges at token {div}, "
                      f"len_a={len(ids_a)} len_b={len(ids_b)}")

    # Acceptance comparison
    print("\n" + "=" * 100)
    print("ACCEPTANCE REPRODUCIBILITY")
    print("=" * 100)
    for pair_label, name_a, name_b in [
        ("KMP run1 vs run2", "KMP_run1", "KMP_run2"),
        ("Hash run1 vs run2", "Hash_run1", "Hash_run2"),
    ]:
        ra = all_results.get(name_a, {})
        rb = all_results.get(name_b, {})
        if "error" in ra or "error" in rb:
            continue
        sa = ra.get("acceptance_stats", {})
        sb = rb.get("acceptance_stats", {})
        acc_a = sa.get("acceptance_rate_pct", 0)
        acc_b = sb.get("acceptance_rate_pct", 0)
        ml_a = sa.get("mean_accepted_length", 0)
        ml_b = sb.get("mean_accepted_length", 0)
        print(f"  {pair_label}:")
        print(f"    Accept:  {acc_a:.2f}% vs {acc_b:.2f}% "
              f"(delta={abs(acc_a-acc_b):.2f})")
        print(f"    MeanLen: {ml_a:.2f} vs {ml_b:.2f} "
              f"(delta={abs(ml_a-ml_b):.2f})")

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"single_repro_{ts}.json")
    # Strip output_token_ids to keep file size reasonable
    save_results = {}
    for name, r in all_results.items():
        if "error" in r:
            save_results[name] = r
            continue
        save_r = dict(r)
        save_r["per_request"] = [
            {k: v for k, v in pr.items() if k != "output_token_ids"}
            for pr in r["per_request"]
        ]
        save_results[name] = save_r
    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
