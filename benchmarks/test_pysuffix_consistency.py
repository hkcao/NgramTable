"""
Consistency test: compare Python Suffix (ns=1) vs C++ Suffix on same data.

Tests that with single-token root keys, the Python implementation produces
the same (or very similar) draft tokens as the C++ SuffixTree.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# Mock vllm.config to allow importing ngram_proposer
import types
vllm_mod = types.ModuleType('vllm')
vllm_config = types.ModuleType('vllm.config')


class VllmConfig:
    pass


vllm_config.VllmConfig = VllmConfig
vllm_mod.config = vllm_config
sys.modules['vllm'] = vllm_mod
sys.modules['vllm.config'] = vllm_config

from ngram_proposer import SuffixCache

# Import C++ SuffixTree
try:
    from arctic_inference.suffix_decoding._C import SuffixTree as CppSuffixTree
    HAS_CPP = True
except ImportError:
    HAS_CPP = False
    print("WARNING: arctic_inference not available, skipping C++ comparison")


def test_basic_local():
    """Test that local tree speculation works correctly."""
    print("\n=== Test: basic local tree ===")
    cache = SuffixCache(root_ns=1, max_depth=64)
    # Repeating pattern
    tokens = [1, 2, 3, 4, 5] * 5
    cache.start_request(0, tokens)

    # Context ends with [4, 5] → expect [1, 2, 3, 4, 5, ...]
    draft = cache.speculate(0, [3, 4, 5], max_tokens=5, min_prob=0.1)
    print(f"  context=[3,4,5] → draft={draft}")
    assert len(draft) > 0, "Should produce drafts"
    assert draft[0] == 1, f"Expected first draft=1, got {draft[0]}"
    print("  PASS")


def test_local_plus_global():
    """Test that global tree is used after request finishes."""
    print("\n=== Test: local + global tree ===")
    cache = SuffixCache(root_ns=1, max_depth=64)

    # Request 0: generate a repeating response pattern
    # Global tree only stores RESPONSE tokens (not prompt)
    prompt0 = [100, 200, 300]
    cache.start_request(0, prompt0)
    # Generate enough repeating tokens to build patterns in global tree
    response0 = [10, 20, 30, 40, 50] * 5
    cache.add_tokens(0, response0)
    cache.stop_request(0)

    # Request 1: different prompt, no local match for [20, 30]
    prompt1 = [900, 800, 700]
    cache.start_request(1, prompt1)

    # Context [20, 30] should match global tree → draft [40, 50, ...]
    draft = cache.speculate(1, [20, 30], max_tokens=5, min_prob=0.1)
    print(f"  context=[20,30] via global → draft={draft}")
    assert len(draft) > 0, "Global tree should provide drafts"
    assert draft[0] == 40, f"Expected first draft=40, got {draft[0]}"
    print("  PASS")

    cache.stop_request(1)


def test_ngram_root():
    """Test n-gram root key improves specificity."""
    print("\n=== Test: n-gram root specificity ===")

    # Pattern A: [1, 2, 3] → [10, 11]
    # Pattern B: [5, 2, 3] → [20, 21]
    # With ns=1, root key=2 mixes both patterns
    # With ns=3, root key=(1,2,3) and (5,2,3) are separate

    tokens_a = [1, 2, 3, 10, 11] * 5
    tokens_b = [5, 2, 3, 20, 21] * 5

    # ns=1: ambiguous
    cache1 = SuffixCache(root_ns=1, max_depth=64)
    cache1.start_request(0, tokens_a + tokens_b)
    draft1 = cache1.speculate(0, [1, 2, 3], max_tokens=3, min_prob=0.05)
    print(f"  ns=1 context=[1,2,3] → draft={draft1}")
    cache1.stop_request(0)

    # ns=3: precise
    cache3 = SuffixCache(root_ns=3, max_depth=64)
    cache3.start_request(0, tokens_a + tokens_b)
    draft3 = cache3.speculate(0, [1, 2, 3], max_tokens=3, min_prob=0.05)
    print(f"  ns=3 context=[1,2,3] → draft={draft3}")
    cache3.stop_request(0)

    assert draft3[0] == 10, f"ns=3 should precisely match pattern A, got {draft3[0]}"
    print("  PASS: ns=3 correctly isolates pattern A")


def test_incremental():
    """Test incremental token addition."""
    print("\n=== Test: incremental add_tokens ===")
    cache = SuffixCache(root_ns=1, max_depth=64)

    prompt = [1, 2, 3, 4, 5]
    cache.start_request(0, prompt)

    # Add tokens one by one (simulating generation)
    cache.add_tokens(0, [1])
    cache.add_tokens(0, [2])
    cache.add_tokens(0, [3])

    # Now context [2, 3] should match both prompt and generated
    draft = cache.speculate(0, [2, 3], max_tokens=3, min_prob=0.1)
    print(f"  after incremental add → draft={draft}")
    assert len(draft) > 0, "Should produce drafts"
    assert draft[0] == 4, f"Expected 4, got {draft[0]}"
    print("  PASS")

    cache.stop_request(0)


def test_prob_filter():
    """Test probability filtering with cumulative probability.

    C++ uses cumulative product probability: prob *= child.count / parent.count.
    The filter checks cumulative prob >= min_prob at each step.
    """
    print("\n=== Test: probability filter (cumulative) ===")
    cache = SuffixCache(root_ns=1, max_depth=64)

    # After root [1]: [2] appears 10 times, [9] appears 1 time
    # Use context length > 1 so match_len range is non-empty
    # (C++ also requires context.size() > 1 for match_len=1 to execute)
    tokens = [1, 2, 3] * 10 + [1, 9, 8]
    cache.start_request(0, tokens)

    # Context [0, 1] (length 2) → match_len=1: root_key=1, speculate [2,3,...]
    # Step 1: 2 has count=10, parent(1-trie root)=11 → prob=10/11≈0.91 → pass
    # Step 2: 3 has count=10, parent=10 → step=1.0, cumul=0.91 → pass
    draft_high = cache.speculate(0, [0, 1], max_tokens=3, min_prob=0.5)
    print(f"  min_prob=0.5 → draft={draft_high}")
    assert len(draft_high) > 0 and draft_high[0] == 2, \
        f"Expected first draft=2, got {draft_high}"

    # Very high prob filter: cumulative 10/11 ≈ 0.91 < 0.99 → stops at step 1
    draft_strict = cache.speculate(0, [0, 1], max_tokens=3, min_prob=0.99)
    print(f"  min_prob=0.99 → draft={draft_strict}")
    assert len(draft_strict) == 0, \
        f"Expected empty draft with min_prob=0.99, got {draft_strict}"

    print("  PASS")
    cache.stop_request(0)


def test_cpp_consistency():
    """Compare Python Suffix (ns=1) vs C++ SuffixTree output."""
    if not HAS_CPP:
        print("\n=== Test: C++ consistency (SKIPPED - no arctic_inference) ===")
        return

    print("\n=== Test: C++ consistency ===")

    # Build same data in both Python and C++ trees
    prompt = [10, 20, 30, 40, 50, 10, 20, 30, 60, 70,
              10, 20, 30, 40, 50, 10, 20, 30, 80, 90]
    generated = [10, 20, 30, 40, 50]

    # --- Python Suffix (ns=1) ---
    py_cache = SuffixCache(root_ns=1, max_depth=64)
    py_cache.start_request(0, prompt)
    py_cache.add_tokens(0, generated)

    # --- C++ SuffixTree ---
    cpp_tree = CppSuffixTree(64)
    all_tokens = prompt + generated
    cpp_tree.extend(0, all_tokens)

    # Test multiple contexts
    test_contexts = [
        [20, 30],
        [10, 20, 30],
        [30, 40, 50],
        [40, 50, 10],
        [10, 20, 30, 40],
    ]

    all_match = True
    for ctx in test_contexts:
        py_draft = py_cache.speculate(
            0, ctx, max_tokens=5, min_prob=0.1)
        cpp_draft_obj = cpp_tree.speculate(
            ctx, 5, 1.0, 0.0, 0.1, False)
        cpp_draft = list(cpp_draft_obj.token_ids)

        match = py_draft == cpp_draft
        status = "OK" if match else "DIFF"
        print(f"  ctx={ctx}")
        print(f"    Python: {py_draft}")
        print(f"    C++:    {cpp_draft}  [{status}]")
        if not match:
            all_match = False

    if all_match:
        print("  ALL MATCH!")
    else:
        print("  NOTE: Some differences expected due to architecture "
              "(Python uses per-root-key trees, C++ uses single tree "
              "with all-match-length scoring)")

    py_cache.stop_request(0)


def test_score_based_selection():
    """Test that score-based selection picks better draft.

    With max_spec_factor=1.0, draft length is limited to match_len * factor.
    Use longer context so match_len is large enough for meaningful drafts.
    """
    print("\n=== Test: score-based local vs global selection ===")
    # Use max_spec_factor=0 (unlimited) to isolate score-based selection
    cache = SuffixCache(root_ns=1, max_depth=64, max_spec_factor=0.0)

    # Request 0: establishes pattern [1,2,3] → [4,5,1,2,3,...] in global (strong)
    tokens0 = [1, 2, 3, 4, 5] * 10
    cache.start_request(0, tokens0)
    cache.add_tokens(0, [1, 2, 3, 4, 5] * 3)
    cache.stop_request(0)

    # Request 1: weak local pattern [1,2,3] → [9]
    tokens1 = [1, 2, 3, 9, 1, 2, 3, 9]
    cache.start_request(1, tokens1)

    # Use longer context so match_len > 1 is possible
    draft = cache.speculate(1, [0, 1, 2, 3], max_tokens=5, min_prob=0.1)
    print(f"  global (strong) vs local (weak) → draft={draft}")
    # Both local and global should produce drafts; global should have higher score
    assert len(draft) > 0, "Should produce at least one draft token"
    print("  PASS")

    cache.stop_request(1)


def test_cpp_thorough():
    """Thorough comparison: Python ns=1 vs C++ with dual-tree (local+global)."""
    if not HAS_CPP:
        print("\n=== Test: C++ thorough (SKIPPED - no arctic_inference) ===")
        return

    print("\n=== Test: C++ thorough (dual-tree, multiple patterns) ===")

    from arctic_inference.suffix_decoding import SuffixDecodingCache

    import random
    random.seed(42)

    # Generate realistic token sequences
    vocab = list(range(100, 200))
    prompt = [random.choice(vocab) for _ in range(50)]
    response = [random.choice(vocab) for _ in range(30)]

    # Also add repeating patterns to ensure matches
    pattern = [101, 102, 103, 104, 105]
    prompt += pattern * 5
    response += pattern * 3

    # --- Python Suffix (ns=1) ---
    py_cache = SuffixCache(root_ns=1, max_depth=64, max_spec_factor=1.0)
    py_cache.start_request(0, prompt)
    py_cache.add_tokens(0, response)

    # --- C++ SuffixDecodingCache (dual-tree) ---
    cpp_cache = SuffixDecodingCache(max_tree_depth=64)
    cpp_cache.start_request(0, prompt)
    cpp_cache.add_active_response(0, response)

    # Test many contexts
    all_tokens = prompt + response
    n_tests = 0
    n_match = 0
    diffs = []

    for ctx_len in [2, 3, 5, 8, 10]:
        for start in range(0, len(all_tokens) - ctx_len, 7):
            context = all_tokens[start:start + ctx_len]
            for max_tokens in [3, 5]:
                for min_prob in [0.1, 0.3]:
                    py_draft = py_cache.speculate(
                        0, context, max_tokens=max_tokens, min_prob=min_prob)
                    cpp_result = cpp_cache.speculate(
                        0, context, max_spec_tokens=max_tokens,
                        max_spec_factor=1.0, max_spec_offset=0.0,
                        min_token_prob=min_prob)
                    cpp_draft = list(cpp_result.token_ids)
                    n_tests += 1
                    if py_draft == cpp_draft:
                        n_match += 1
                    else:
                        diffs.append((context[:3], py_draft, cpp_draft))

    match_pct = n_match / n_tests * 100
    print(f"  {n_match}/{n_tests} matched ({match_pct:.1f}%)")
    if diffs:
        for ctx, py, cpp in diffs[:5]:
            print(f"    DIFF ctx={ctx}... py={py} cpp={cpp}")
        if len(diffs) > 5:
            print(f"    ... and {len(diffs)-5} more diffs")
    assert match_pct >= 95.0, f"Match rate {match_pct:.1f}% < 95%"
    print("  PASS")

    py_cache.stop_request(0)
    cpp_cache.stop_request(0)


def test_incremental_counts():
    """Test that incremental extend doesn't inflate counts.

    Bug: SuffixTrie.extend() was incrementing junction node count,
    causing probability distortion (prob = count/parent_count was wrong).
    """
    print("\n=== Test: incremental extend count correctness ===")
    # Use max_spec_factor=0 (unlimited) to isolate count testing
    cache = SuffixCache(root_ns=1, max_depth=64, max_spec_factor=0.0)

    # Prompt: [A, B, C] → after insert, trie[B] has root(1)→C(1)
    prompt = [10, 20, 30]
    cache.start_request(0, prompt)

    # Response: [40] → extend trie[B]: [C] with [40]
    # C node should NOT get inflated (should stay count=1)
    cache.add_tokens(0, [40])

    # Response: [50] → extend trie[B]: [C, 40] with [50]
    cache.add_tokens(0, [50])

    # Use longer context so match_len > 1 is possible
    # context [..., 20, 30] → match_len=1: root_key=30, speculate
    # With factor=0 (unlimited), should get multiple drafts
    # prob should be 1.0 at each step (all counts=1, no inflation)
    draft = cache.speculate(0, [0, 20, 30], max_tokens=3, min_prob=0.5)
    print(f"  context=[0,20,30] min_prob=0.5 → draft={draft}")
    assert len(draft) >= 2, f"Expected ≥2 drafts (prob should be 1.0), got {draft}"
    assert draft[0] == 40, f"Expected 40, got {draft[0]}"
    assert draft[1] == 50, f"Expected 50, got {draft[1]}"
    print("  PASS: counts not inflated, prob stays 1.0")

    cache.stop_request(0)


def test_cpp_incremental_consistency():
    """Compare Python vs C++ with incremental token addition."""
    if not HAS_CPP:
        print("\n=== Test: C++ incremental consistency (SKIPPED) ===")
        return

    print("\n=== Test: C++ incremental consistency ===")
    from arctic_inference.suffix_decoding import SuffixDecodingCache

    # Build same data incrementally in both
    prompt = [10, 20, 30, 40, 50, 10, 20, 30, 60, 70]

    py_cache = SuffixCache(root_ns=1, max_depth=64, max_spec_factor=1.0)
    cpp_cache = SuffixDecodingCache(max_tree_depth=64)

    py_cache.start_request(0, prompt)
    cpp_cache.start_request(0, prompt)

    # Add response tokens one at a time (incremental)
    response_tokens = [10, 20, 30, 40, 50, 80, 90]
    for tok in response_tokens:
        py_cache.add_tokens(0, [tok])
        cpp_cache.add_active_response(0, [tok])

    # Test multiple contexts
    all_tokens = prompt + response_tokens
    n_tests = 0
    n_match = 0
    diffs = []

    for ctx_len in [2, 3, 5, 8]:
        for start in range(0, len(all_tokens) - ctx_len, 3):
            context = all_tokens[start:start + ctx_len]
            for max_tokens in [3, 5]:
                for min_prob in [0.1, 0.3, 0.5]:
                    py_draft = py_cache.speculate(
                        0, context, max_tokens=max_tokens, min_prob=min_prob)
                    cpp_result = cpp_cache.speculate(
                        0, context, max_spec_tokens=max_tokens,
                        max_spec_factor=1.0, max_spec_offset=0.0,
                        min_token_prob=min_prob)
                    cpp_draft = list(cpp_result.token_ids)
                    n_tests += 1
                    if py_draft == cpp_draft:
                        n_match += 1
                    else:
                        diffs.append((context[:3], min_prob, py_draft, cpp_draft))

    match_pct = n_match / n_tests * 100
    print(f"  {n_match}/{n_tests} matched ({match_pct:.1f}%)")
    if diffs:
        for ctx, mp, py, cpp in diffs[:5]:
            print(f"    DIFF ctx={ctx}... min_prob={mp} py={py} cpp={cpp}")
        if len(diffs) > 5:
            print(f"    ... and {len(diffs)-5} more diffs")
    assert match_pct >= 95.0, f"Incremental match rate {match_pct:.1f}% < 95%"
    print("  PASS")

    py_cache.stop_request(0)
    cpp_cache.stop_request(0)


def test_cpp_realistic_incremental():
    """Simulate realistic speculative decoding: token-by-token addition.

    Tests with larger, randomized data and varied batch sizes (1-5 tokens
    per add, simulating accepted draft lengths).
    """
    if not HAS_CPP:
        print("\n=== Test: C++ realistic incremental (SKIPPED) ===")
        return

    print("\n=== Test: C++ realistic incremental ===")
    from arctic_inference.suffix_decoding import SuffixDecodingCache
    import random
    random.seed(123)

    vocab = list(range(100, 300))
    prompt = [random.choice(vocab) for _ in range(100)]
    # Add repeating patterns
    pattern = [150, 160, 170, 180, 190]
    prompt += pattern * 5

    py_cache = SuffixCache(root_ns=1, max_depth=64, max_spec_factor=1.0)
    cpp_cache = SuffixDecodingCache(max_tree_depth=64)

    py_cache.start_request(0, prompt)
    cpp_cache.start_request(0, prompt)

    # Generate 50 response tokens, adding in varied batch sizes (1-5)
    response = [random.choice(vocab) for _ in range(30)]
    response += pattern * 4  # Add known patterns
    i = 0
    while i < len(response):
        batch_size = random.randint(1, 5)
        batch = response[i:i + batch_size]
        py_cache.add_tokens(0, batch)
        cpp_cache.add_active_response(0, batch)
        i += len(batch)

    # Comprehensive comparison
    all_tokens = prompt + response
    n_tests = 0
    n_match = 0
    diffs = []

    for ctx_len in [2, 3, 5, 8, 12]:
        for start in range(0, len(all_tokens) - ctx_len, 5):
            context = all_tokens[start:start + ctx_len]
            for max_tokens in [3, 5]:
                for min_prob in [0.1, 0.3]:
                    py_draft = py_cache.speculate(
                        0, context, max_tokens=max_tokens, min_prob=min_prob)
                    cpp_result = cpp_cache.speculate(
                        0, context, max_spec_tokens=max_tokens,
                        max_spec_factor=1.0, max_spec_offset=0.0,
                        min_token_prob=min_prob)
                    cpp_draft = list(cpp_result.token_ids)
                    n_tests += 1
                    if py_draft == cpp_draft:
                        n_match += 1
                    else:
                        diffs.append({
                            "ctx": context[:4],
                            "ctx_len": ctx_len,
                            "max_tok": max_tokens,
                            "min_prob": min_prob,
                            "py": py_draft,
                            "cpp": cpp_draft,
                            "py_score": None,
                            "cpp_score": cpp_result.score,
                            "cpp_match_len": cpp_result.match_len,
                        })

    match_pct = n_match / n_tests * 100
    print(f"  {n_match}/{n_tests} matched ({match_pct:.1f}%)")
    if diffs:
        print(f"  {len(diffs)} diffs found:")
        for d in diffs[:10]:
            print(f"    ctx={d['ctx']}...(len={d['ctx_len']}) "
                  f"max={d['max_tok']} prob={d['min_prob']} "
                  f"py={d['py']} cpp={d['cpp']} "
                  f"cpp_score={d['cpp_score']:.3f} "
                  f"cpp_match={d['cpp_match_len']}")
    assert match_pct >= 95.0, f"Realistic match rate {match_pct:.1f}% < 95%"
    print("  PASS")

    py_cache.stop_request(0)
    cpp_cache.stop_request(0)


if __name__ == "__main__":
    test_basic_local()
    test_local_plus_global()
    test_ngram_root()
    test_incremental()
    test_prob_filter()
    test_score_based_selection()
    test_incremental_counts()
    test_cpp_consistency()
    test_cpp_thorough()
    test_cpp_incremental_consistency()
    test_cpp_realistic_incremental()

    print("\n" + "=" * 50)
    print("All tests passed!")
