"""
Debug test: run C++ Suffix and PySuffix ns=1 on 20 prompts,
with debug logging enabled to find actual divergence points.
Processes requests sequentially (one at a time) so debug logs
are per-request comparable.
"""
import json
import os
import re
import time

from test_pysuffix import build_swe_prompts, run_subprocess


def parse_log(path):
    """Parse debug log into list of (draft_str, full_line)."""
    entries = []
    if not os.path.exists(path):
        return entries
    with open(path) as f:
        for line in f:
            line = line.strip()
            idx = line.find("draft=")
            if idx >= 0:
                # Extract just "draft=[...]" (before any space-delimited extras)
                draft_part = line[idx:]
                # Find the end of the draft list: first ']' after 'draft=['
                bracket_end = draft_part.find("]")
                if bracket_end >= 0:
                    draft_str = draft_part[:bracket_end + 1]
                else:
                    draft_str = draft_part
                entries.append((draft_str, line))
    return entries


def main():
    prompt_texts = build_swe_prompts(num_samples=20, max_prompt_len=2048)

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

    model = "Qwen/Qwen2.5-3B-Instruct"
    gpu_mem = 0.8
    max_tokens = 512

    # Run C++ Suffix with debug
    print(">>> Running C++ Suffix with debug (20 prompts) ...")
    r1 = run_subprocess(
        prompt_texts, model, gpu_mem, max_tokens,
        spec_config=suffix_spec, mode_name="C++ Suffix",
        env_overrides={
            "VLLM_SUFFIX_MIN_MATCH_LEN": "0",
            "VLLM_SUFFIX_DEBUG": "1",
        },
    )
    time.sleep(2)

    # Run PySuffix ns=1 with debug
    print(">>> Running PySuffix ns=1 with debug (20 prompts) ...")
    r2 = run_subprocess(
        prompt_texts, model, gpu_mem, max_tokens,
        spec_config=ngram_spec, mode_name="PySuffix ns=1",
        env_overrides={
            "VLLM_NGRAM_USE_HASH": "0",
            "VLLM_NGRAM_USE_TRIE": "0",
            "VLLM_NGRAM_USE_PYSUFFIX": "1",
            "VLLM_PYSUFFIX_ROOT_NS": "1",
            "VLLM_PYSUFFIX_MIN_PROB": "0.1",
            "VLLM_PYSUFFIX_MIN_MATCH_LEN": "0",
            "VLLM_PYSUFFIX_DEBUG": "1",
        },
    )

    # Show results
    print("\n" + "=" * 80)
    print("OVERALL RESULTS")
    print("=" * 80)
    for r in [r1, r2]:
        if "error" in r:
            print(f"ERROR: {r}")
            continue
        s = r.get("acceptance_stats", {})
        print(f"{r['mode']}: accept={s.get('acceptance_rate_pct')}% "
              f"mean_len={s.get('mean_accepted_length')} "
              f"drafts={s.get('num_drafts')}")

    # Per-request comparison
    if "error" not in r1 and "error" not in r2:
        pr1 = r1.get("per_request", [])
        pr2 = r2.get("per_request", [])
        print(f"\nPer-request output tokens comparison:")
        for i in range(min(len(pr1), len(pr2))):
            t1 = pr1[i]["output_tokens"]
            t2 = pr2[i]["output_tokens"]
            match = "==" if t1 == t2 else f"DIFF ({t1} vs {t2})"
            print(f"  req[{i}]: C++={t1} Py={t2} {match}")

    # Compare debug logs
    print("\n" + "=" * 80)
    print("DEBUG LOG ANALYSIS")
    print("=" * 80)

    suffix_entries = parse_log("/tmp/suffix_debug.log")
    pysuffix_entries = parse_log("/tmp/pysuffix_debug.log")
    print(f"C++ Suffix: {len(suffix_entries)} draft steps")
    print(f"PySuffix:   {len(pysuffix_entries)} draft steps")

    if not suffix_entries or not pysuffix_entries:
        print("Cannot compare - missing logs")
        return

    # Find first divergence
    n = min(len(suffix_entries), len(pysuffix_entries))
    first_diff = None
    for i in range(n):
        if suffix_entries[i][0] != pysuffix_entries[i][0]:
            first_diff = i
            break

    if first_diff is None:
        print(f"\nFirst {n} steps: ALL IDENTICAL")
    else:
        print(f"\n*** First divergence at step {first_diff + 1} ***")
        print(f"  C++ Suffix:  {suffix_entries[first_diff][1]}")
        print(f"  PySuffix:    {pysuffix_entries[first_diff][1]}")
        # Show 5 steps before for context
        print(f"\n  Context (5 steps before):")
        for j in range(max(0, first_diff - 5), first_diff):
            print(f"    C++[{j+1}]: {suffix_entries[j][1]}")
            print(f"    Py [{j+1}]: {pysuffix_entries[j][1]}")
        # Show 5 steps after
        print(f"\n  After divergence (5 steps):")
        for j in range(first_diff + 1, min(first_diff + 6, n)):
            print(f"    C++[{j+1}]: {suffix_entries[j][1]}")
            print(f"    Py [{j+1}]: {pysuffix_entries[j][1]}")

    # Count divergences in total
    diff_count = 0
    for i in range(n):
        if suffix_entries[i][0] != pysuffix_entries[i][0]:
            diff_count += 1
    print(f"\nTotal divergent steps: {diff_count}/{n}")

    # Tie-breaking stats
    tie_lines = [e[1] for e in pysuffix_entries if "TIES:" in e[1]]
    print(f"PySuffix tie-breaking events: {len(tie_lines)}/{len(pysuffix_entries)}")

    # Check if divergence steps have ties
    if first_diff is not None:
        has_tie = "TIES:" in pysuffix_entries[first_diff][1]
        print(f"First divergence has tie-breaking: {has_tie}")


if __name__ == "__main__":
    main()
