"""
Reproducibility test: run C++ Suffix and PySuffix ns=1 twice each,
compare per-request output token counts to verify determinism.
"""
import argparse
import json
import os
import time

from test_pysuffix import build_swe_prompts, run_subprocess


def main():
    parser = argparse.ArgumentParser(description="Reproducibility test")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-prompt-len", type=int, default=2048)
    parser.add_argument("--gpu-mem", type=float, default=0.8)
    args = parser.parse_args()

    prompt_texts = build_swe_prompts(args.num_samples, args.max_prompt_len)

    suffix_spec = {
        "method": "suffix",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 5,
        "prompt_lookup_min": 2,
    }
    ngram_spec = {
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 5,
        "prompt_lookup_min": 2,
    }

    configs = [
        ("C++ Suffix", suffix_spec, {
            "VLLM_SUFFIX_MIN_MATCH_LEN": "0",
        }),
        ("PySuffix ns=1", ngram_spec, {
            "VLLM_NGRAM_USE_HASH": "0",
            "VLLM_NGRAM_USE_TRIE": "0",
            "VLLM_NGRAM_USE_PYSUFFIX": "1",
            "VLLM_PYSUFFIX_ROOT_NS": "1",
            "VLLM_PYSUFFIX_MIN_PROB": "0.1",
            "VLLM_PYSUFFIX_MIN_MATCH_LEN": "0",
        }),
    ]

    # Run each config twice
    all_results = {}  # name -> [run1, run2]
    for name, sc, env in configs:
        all_results[name] = []
        for run_idx in range(2):
            label = f"{name} (run {run_idx+1})"
            print(f"\n>>> {label} ...")
            r = run_subprocess(
                prompt_texts, args.model, args.gpu_mem, args.max_tokens,
                spec_config=sc, mode_name=label,
                env_overrides=env,
            )
            all_results[name].append(r)
            time.sleep(2)

    # Compare
    print("\n" + "=" * 100)
    print("REPRODUCIBILITY COMPARISON")
    print("=" * 100)

    for name in all_results:
        r1, r2 = all_results[name]
        if "error" in r1 or "error" in r2:
            print(f"\n{name}: ERROR in one or both runs")
            continue

        print(f"\n--- {name} ---")
        s1 = r1.get("acceptance_stats", {})
        s2 = r2.get("acceptance_stats", {})
        print(f"  Run 1: accept={s1.get('acceptance_rate_pct')}% "
              f"mean_len={s1.get('mean_accepted_length')} "
              f"drafts={s1.get('num_drafts')} "
              f"draft_tok={s1.get('num_draft_tokens')} "
              f"accepted={s1.get('num_accepted_tokens')}")
        print(f"  Run 2: accept={s2.get('acceptance_rate_pct')}% "
              f"mean_len={s2.get('mean_accepted_length')} "
              f"drafts={s2.get('num_drafts')} "
              f"draft_tok={s2.get('num_draft_tokens')} "
              f"accepted={s2.get('num_accepted_tokens')}")

        # Compare per-request output tokens
        pr1 = r1.get("per_request", [])
        pr2 = r2.get("per_request", [])
        mismatches = []
        for i in range(min(len(pr1), len(pr2))):
            if pr1[i]["output_tokens"] != pr2[i]["output_tokens"]:
                mismatches.append((i, pr1[i]["output_tokens"],
                                   pr2[i]["output_tokens"]))

        if not mismatches:
            print(f"  Per-request output tokens: IDENTICAL ({len(pr1)} requests)")
        else:
            print(f"  Per-request output tokens: {len(mismatches)} MISMATCHES "
                  f"out of {len(pr1)}:")
            for idx, t1, t2 in mismatches[:10]:
                print(f"    req[{idx}]: run1={t1} run2={t2}")

        # Compare per-position acceptance
        pp1 = s1.get("per_position_acceptance", [])
        pp2 = s2.get("per_position_acceptance", [])
        if pp1 == pp2:
            print(f"  Per-position acceptance: IDENTICAL")
        else:
            print(f"  Per-position acceptance: DIFFERENT")
            print(f"    Run 1: {pp1}")
            print(f"    Run 2: {pp2}")

        # Overall stats match?
        stats_match = (
            s1.get("acceptance_rate_pct") == s2.get("acceptance_rate_pct") and
            s1.get("mean_accepted_length") == s2.get("mean_accepted_length") and
            s1.get("num_drafts") == s2.get("num_drafts") and
            s1.get("num_draft_tokens") == s2.get("num_draft_tokens") and
            s1.get("num_accepted_tokens") == s2.get("num_accepted_tokens")
        )
        print(f"  Overall stats: {'IDENTICAL' if stats_match else 'DIFFERENT'}")

    # Save
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "reproducibility_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
