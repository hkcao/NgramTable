"""
Compare C++ SuffixDecodingCache vs Python SuffixCache at the cache level.

Feeds identical tokens to both, then compares speculate() at each step,
including local vs global tree selection.
"""
import numpy as np
from arctic_inference.suffix_decoding import SuffixDecodingCache

import sys
sys.path.insert(0, "/home/hank/Agent/ngram")
from ngram_proposer import SuffixCache


def main():
    # Use first SWE-bench prompt tokens
    from test_pysuffix import build_swe_prompts
    from vllm import LLM, SamplingParams

    prompts = build_swe_prompts(num_samples=1, max_prompt_len=2048)
    llm = LLM(model="Qwen/Qwen2.5-3B-Instruct",
              gpu_memory_utilization=0.8,
              trust_remote_code=True)
    # Tokenize the prompt
    tokenizer = llm.get_tokenizer()
    prompt_token_ids = tokenizer.encode(prompts[0])
    print(f"Prompt tokens: {len(prompt_token_ids)}")

    # Generate output to get response tokens
    sp = SamplingParams(temperature=0.0, max_tokens=512)
    outputs = llm.generate([prompts[0]], sp)
    response_token_ids = list(outputs[0].outputs[0].token_ids)
    print(f"Response tokens: {len(response_token_ids)}")

    # Clean up LLM
    llm.llm_engine.engine_core.shutdown()
    del llm
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()

    # Now simulate speculative decoding step by step
    max_depth = 24
    max_spec_factor = 1.0
    min_prob = 0.1
    max_spec_tokens = 5

    # Init C++ cache
    cpp_cache = SuffixDecodingCache(max_tree_depth=max_depth)
    cpp_cache.start_request("req0", prompt_token_ids)

    # Init Python cache
    py_cache = SuffixCache(root_ns=1, max_depth=max_depth,
                           max_spec_factor=max_spec_factor)
    py_cache.start_request(0, prompt_token_ids)

    # Simulate feeding response tokens one at a time and comparing speculation
    all_tokens = list(prompt_token_ids)
    divergences = []

    for step, tok in enumerate(response_token_ids):
        # Add token to both
        cpp_cache.add_active_response("req0", [tok])
        py_cache.add_tokens(0, [tok])
        all_tokens.append(tok)

        # Build context (last max_depth tokens)
        ctx_len = min(max_depth, len(all_tokens))
        context = all_tokens[-ctx_len:]

        # C++ speculate
        cpp_draft = cpp_cache.speculate(
            "req0", context,
            max_spec_tokens=max_spec_tokens,
            max_spec_factor=max_spec_factor,
            min_token_prob=min_prob)

        # Python speculate - get both local and global results
        msf = py_cache.max_spec_factor
        py_local, py_local_score, _ = py_cache._speculate_from(
            py_cache._locals.get(0, {}), context,
            max_spec_tokens, min_prob, msf)
        py_global, py_global_score, _ = py_cache._speculate_from(
            py_cache._global, context,
            max_spec_tokens, min_prob, msf)
        if py_local_score >= py_global_score:
            py_draft = py_local
            py_score = py_local_score
            py_src = "L"
        else:
            py_draft = py_global
            py_score = py_global_score
            py_src = "G"

        # Also get C++ local and global separately
        cpp_local = cpp_cache._local_trees["req0"].speculate(
            context, max_spec_tokens, max_spec_factor, 0.0,
            min_prob, False)
        cpp_global = cpp_cache._global_tree.speculate(
            context, max_spec_tokens, max_spec_factor, 0.0,
            min_prob, False)

        cpp_local_score = cpp_local.score
        cpp_global_score = cpp_global.score
        cpp_src = "L" if cpp_local_score >= cpp_global_score else "G"

        if cpp_draft.token_ids != py_draft:
            divergences.append(step)
            print(f"\nDIVERGENCE at step {step+1}, ctx_tail={context[-3:]}:")
            print(f"  C++: draft={cpp_draft.token_ids} match_len={cpp_draft.match_len} "
                  f"score={cpp_draft.score:.10f} src={cpp_src}")
            print(f"    C++ local:  draft={cpp_local.token_ids} "
                  f"score={cpp_local_score:.10f} match_len={cpp_local.match_len}")
            print(f"    C++ global: draft={cpp_global.token_ids} "
                  f"score={cpp_global_score:.10f} match_len={cpp_global.match_len}")
            print(f"  Py:  draft={py_draft} "
                  f"score={py_score:.10f} src={py_src}")
            print(f"    Py  local:  draft={py_local} "
                  f"score={py_local_score:.10f}")
            print(f"    Py  global: draft={py_global} "
                  f"score={py_global_score:.10f}")

            # Dump tree details for root key = last context token
            last_tok = context[-1]
            response_so_far = all_tokens[len(prompt_token_ids):]
            tok_count_in_resp = response_so_far.count(last_tok)
            py_global_tree = py_cache._global.get(last_tok)
            py_local_tree = py_cache._locals.get(0, {}).get(last_tok)
            print(f"  --- Root key {last_tok} analysis ---")
            print(f"  Occurrences in response[:{len(response_so_far)}]: {tok_count_in_resp}")
            if py_global_tree:
                children_info = {t: c.count for t, c in py_global_tree.root.children.items()}
                print(f"  Py global trie: root.count={py_global_tree.root.count}, "
                      f"children={children_info}")
            else:
                print(f"  Py global trie: NOT FOUND")
            if py_local_tree:
                children_info = {t: c.count for t, c in py_local_tree.root.children.items()}
                print(f"  Py local trie: root.count={py_local_tree.root.count}, "
                      f"children={children_info}")
            else:
                print(f"  Py local trie: NOT FOUND")

            if len(divergences) >= 5:
                print("\n(stopping after 5 divergences)")
                break
        elif (step + 1) % 50 == 0:
            print(f"  step {step+1}: OK (C++={cpp_draft.token_ids[:3]}... "
                  f"Py={py_draft[:3]}...)")

    print(f"\n{'='*60}")
    print(f"Total steps: {step+1}")
    print(f"Divergences: {len(divergences)}")
    if not divergences:
        print("ALL IDENTICAL!")
    else:
        print(f"First divergence at step {divergences[0]+1}")


if __name__ == "__main__":
    main()
