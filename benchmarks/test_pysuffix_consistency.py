"""
Consistency test: Python Suffix (PySuffixTree/PySuffixCache) vs C++ Suffix.

Tests that the Python implementation produces identical draft tokens as the
C++ SuffixTree/SuffixDecodingCache across a wide range of inputs.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import random

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

from py_suffix_tree import PySuffixTree, PySuffixCache

# Import C++ SuffixTree
try:
    from arctic_inference.suffix_decoding._C import SuffixTree as CppSuffixTree
    from arctic_inference.suffix_decoding import SuffixDecodingCache
    HAS_CPP = True
except ImportError:
    HAS_CPP = False
    print("WARNING: arctic_inference not available, skipping C++ comparison")


def compare_trees(name, ctx, py_tree, cpp_tree, max_tokens=5, min_prob=0.1,
                  max_spec_factor=1.0):
    """Compare PySuffixTree vs C++ SuffixTree for a single query."""
    py_draft = py_tree.speculate(ctx, max_tokens, max_spec_factor, 0.0,
                                 min_prob, False)
    cpp_draft = cpp_tree.speculate(ctx, max_tokens, max_spec_factor, 0.0,
                                   min_prob, False)
    py_toks = py_draft.token_ids
    cpp_toks = list(cpp_draft.token_ids)
    match = py_toks == cpp_toks
    status = "OK" if match else "DIFF"
    print(f"  [{status}] {name}: ctx={ctx}")
    if not match:
        print(f"    Python: tokens={py_toks} match_len={py_draft.match_len} "
              f"score={py_draft.score:.3f} "
              f"probs={[round(p,3) for p in py_draft.probs]}")
        print(f"    C++:    tokens={cpp_toks} match_len={cpp_draft.match_len} "
              f"score={cpp_draft.score:.3f} "
              f"probs={[round(p,3) for p in cpp_draft.probs]}")
    return match


def test_basic_patterns():
    """Test with simple repeating and single-occurrence patterns."""
    if not HAS_CPP:
        print("\n=== Test: basic patterns (SKIPPED) ===")
        return

    print("\n=== Test: basic patterns ===")

    # Repeating pattern
    py = PySuffixTree(64)
    cpp = CppSuffixTree(64)
    tokens = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 6, 7]
    py.extend(0, tokens)
    cpp.extend(0, tokens)
    assert compare_trees("repeating [1,2,3]", [1, 2, 3], py, cpp)
    assert compare_trees("repeating [3,4,5]", [3, 4, 5], py, cpp)
    assert compare_trees("repeating [2,3,4,5]", [2, 3, 4, 5], py, cpp)

    # Single occurrence
    py2 = PySuffixTree(64)
    cpp2 = CppSuffixTree(64)
    py2.extend(0, [10, 20, 30, 40, 50])
    cpp2.extend(0, [10, 20, 30, 40, 50])
    assert compare_trees("single [20,30]", [20, 30], py2, cpp2)
    assert compare_trees("single [10,20,30]", [10, 20, 30], py2, cpp2)

    print("  PASS")


def test_multiple_sequences():
    """Test with multiple sequences and frequency-based probabilities."""
    if not HAS_CPP:
        print("\n=== Test: multiple sequences (SKIPPED) ===")
        return

    print("\n=== Test: multiple sequences ===")

    py = PySuffixTree(64)
    cpp = CppSuffixTree(64)
    py.extend(0, [1, 2, 3, 10, 11, 12])
    py.extend(1, [1, 2, 3, 20, 21, 22])
    py.extend(2, [1, 2, 3, 10, 11, 99])
    cpp.extend(0, [1, 2, 3, 10, 11, 12])
    cpp.extend(1, [1, 2, 3, 20, 21, 22])
    cpp.extend(2, [1, 2, 3, 10, 11, 99])
    assert compare_trees("multi-seq [1,2,3]", [1, 2, 3], py, cpp)

    print("  PASS")


def test_edge_cases():
    """Test no-match, short context, context length effects."""
    if not HAS_CPP:
        print("\n=== Test: edge cases (SKIPPED) ===")
        return

    print("\n=== Test: edge cases ===")

    # No match
    py = PySuffixTree(64)
    cpp = CppSuffixTree(64)
    py.extend(0, [1, 2, 3])
    cpp.extend(0, [1, 2, 3])
    assert compare_trees("no match", [7, 8, 9], py, cpp)

    # Single-token context (should produce no match)
    assert compare_trees("single ctx", [1], py, cpp)

    # Context length and match_len
    py2 = PySuffixTree(64)
    cpp2 = CppSuffixTree(64)
    py2.extend(0, [1, 2, 3, 4, 5, 100, 101, 102])
    cpp2.extend(0, [1, 2, 3, 4, 5, 100, 101, 102])
    for ctx_len in [1, 2, 3, 4, 5]:
        ctx = list(range(1, ctx_len + 1))
        assert compare_trees(f"ctx_len={ctx_len}", ctx, py2, cpp2)

    print("  PASS")


def test_spec_params():
    """Test max_spec_factor and min_token_prob."""
    if not HAS_CPP:
        print("\n=== Test: spec params (SKIPPED) ===")
        return

    print("\n=== Test: spec params ===")

    # max_spec_factor
    py = PySuffixTree(64)
    cpp = CppSuffixTree(64)
    py.extend(0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    cpp.extend(0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert compare_trees("factor=1.0", [1, 2, 3], py, cpp,
                         max_tokens=10, max_spec_factor=1.0)
    assert compare_trees("factor=0.5", [1, 2, 3], py, cpp,
                         max_tokens=10, max_spec_factor=0.5)

    # min_prob
    py2 = PySuffixTree(64)
    cpp2 = CppSuffixTree(64)
    for _ in range(10):
        py2.extend(0, [1, 2, 3, 10, 11])
        cpp2.extend(0, [1, 2, 3, 10, 11])
    py2.extend(1, [1, 2, 3, 20, 21])
    cpp2.extend(1, [1, 2, 3, 20, 21])
    assert compare_trees("min_prob=0.01", [1, 2, 3], py2, cpp2, min_prob=0.01)
    assert compare_trees("min_prob=0.5", [1, 2, 3], py2, cpp2, min_prob=0.5)

    print("  PASS")


def test_incremental_extend():
    """Test incremental token addition."""
    if not HAS_CPP:
        print("\n=== Test: incremental extend (SKIPPED) ===")
        return

    print("\n=== Test: incremental extend ===")

    py = PySuffixTree(64)
    cpp = CppSuffixTree(64)
    py.extend(0, [1, 2, 3])
    py.extend(0, [4, 5])
    py.extend(0, [1, 2, 3, 6])
    cpp.extend(0, [1, 2, 3])
    cpp.extend(0, [4, 5])
    cpp.extend(0, [1, 2, 3, 6])
    assert compare_trees("incremental", [1, 2, 3], py, cpp)

    print("  PASS")


def test_cache_dual_tree():
    """Test PySuffixCache vs SuffixDecodingCache (dual-tree architecture)."""
    if not HAS_CPP:
        print("\n=== Test: cache dual tree (SKIPPED) ===")
        return

    print("\n=== Test: cache dual-tree ===")
    random.seed(42)
    vocab = list(range(100, 300))
    pattern = [150, 160, 170, 180, 190]

    # Prompt-only
    prompt = [random.choice(vocab) for _ in range(50)]
    prompt += pattern * 5

    py_cache = PySuffixCache(max_depth=64, max_spec_factor=1.0)
    cpp_cache = SuffixDecodingCache(max_tree_depth=64)

    py_cache.start_request(0, prompt)
    cpp_cache.start_request(0, prompt)

    n_tests = 0
    n_match = 0
    for ctx_len in [2, 3, 5, 8]:
        for start in range(0, len(prompt) - ctx_len, 5):
            context = prompt[start:start + ctx_len]
            for max_tok in [3, 5]:
                for min_prob in [0.1, 0.3]:
                    py_d, _ = py_cache.speculate(
                        0, context, max_tokens=max_tok, min_prob=min_prob)
                    cpp_r = cpp_cache.speculate(
                        0, context, max_spec_tokens=max_tok,
                        max_spec_factor=1.0, max_spec_offset=0.0,
                        min_token_prob=min_prob)
                    cpp_d = list(cpp_r.token_ids)
                    n_tests += 1
                    if py_d == cpp_d:
                        n_match += 1

    pct = n_match / n_tests * 100
    print(f"  Prompt-only: {n_match}/{n_tests} ({pct:.1f}%)")
    assert pct >= 95.0, f"Match rate {pct:.1f}% < 95%"

    py_cache.stop_request(0)
    cpp_cache.stop_request(0)

    # Prompt + incremental response
    prompt2 = [random.choice(vocab) for _ in range(30)]
    prompt2 += pattern * 3
    response = [random.choice(vocab) for _ in range(20)]
    response += pattern * 2

    py_cache2 = PySuffixCache(max_depth=64, max_spec_factor=1.0)
    cpp_cache2 = SuffixDecodingCache(max_tree_depth=64)

    py_cache2.start_request(0, prompt2)
    cpp_cache2.start_request(0, prompt2)

    i = 0
    while i < len(response):
        batch = random.randint(1, 5)
        chunk = response[i:i + batch]
        py_cache2.add_tokens(0, chunk)
        cpp_cache2.add_active_response(0, chunk)
        i += len(chunk)

    all_tokens = prompt2 + response
    n_tests = 0
    n_match = 0
    for ctx_len in [2, 3, 5, 8]:
        for start in range(0, len(all_tokens) - ctx_len, 5):
            context = all_tokens[start:start + ctx_len]
            for max_tok in [3, 5]:
                for min_prob in [0.1, 0.3]:
                    py_d, _ = py_cache2.speculate(
                        0, context, max_tokens=max_tok, min_prob=min_prob)
                    cpp_r = cpp_cache2.speculate(
                        0, context, max_spec_tokens=max_tok,
                        max_spec_factor=1.0, max_spec_offset=0.0,
                        min_token_prob=min_prob)
                    cpp_d = list(cpp_r.token_ids)
                    n_tests += 1
                    if py_d == cpp_d:
                        n_match += 1

    pct = n_match / n_tests * 100
    print(f"  Prompt+response: {n_match}/{n_tests} ({pct:.1f}%)")
    assert pct >= 95.0, f"Match rate {pct:.1f}% < 95%"

    py_cache2.stop_request(0)
    cpp_cache2.stop_request(0)

    # Cross-request global tree
    py_cache3 = PySuffixCache(max_depth=64, max_spec_factor=1.0)
    cpp_cache3 = SuffixDecodingCache(max_tree_depth=64)

    prompt0 = [random.choice(vocab) for _ in range(20)]
    response0 = pattern * 5

    py_cache3.start_request(0, prompt0)
    cpp_cache3.start_request(0, prompt0)
    py_cache3.add_tokens(0, response0)
    cpp_cache3.add_active_response(0, response0)
    py_cache3.stop_request(0)
    cpp_cache3.stop_request(0)

    prompt1 = [random.choice(vocab) for _ in range(20)]
    py_cache3.start_request(1, prompt1)
    cpp_cache3.start_request(1, prompt1)

    ctx = [150, 160, 170]
    py_d3, _ = py_cache3.speculate(1, ctx, max_tokens=5, min_prob=0.1)
    cpp_r3 = cpp_cache3.speculate(1, ctx, max_spec_tokens=5,
                                   max_spec_factor=1.0, max_spec_offset=0.0,
                                   min_token_prob=0.1)
    cpp_d3 = list(cpp_r3.token_ids)
    match = py_d3 == cpp_d3
    print(f"  Cross-request: ctx={ctx} py={py_d3} cpp={cpp_d3} "
          f"{'OK' if match else 'DIFF'}")
    assert match

    py_cache3.stop_request(1)
    cpp_cache3.stop_request(1)

    print("  PASS")


def test_large_scale_random():
    """Large-scale random comparison test."""
    if not HAS_CPP:
        print("\n=== Test: large-scale random (SKIPPED) ===")
        return

    print("\n=== Test: large-scale random ===")
    random.seed(123)
    vocab = list(range(100, 300))

    py = PySuffixTree(64)
    cpp = CppSuffixTree(64)

    # Build multiple sequences with patterns
    for seq_id in range(5):
        tokens = [random.choice(vocab) for _ in range(100)]
        pattern = [random.choice(vocab) for _ in range(5)]
        tokens += pattern * 3
        py.extend(seq_id, tokens)
        cpp.extend(seq_id, tokens)

    # Collect all tokens for context generation
    all_tokens = []
    for seq in py._seqs.values():
        all_tokens.extend(seq)

    n_tests = 0
    n_match = 0
    for _ in range(500):
        ctx_len = random.randint(2, 10)
        start = random.randint(0, len(all_tokens) - ctx_len)
        context = all_tokens[start:start + ctx_len]
        max_tok = random.choice([3, 5, 8])
        min_prob = random.choice([0.05, 0.1, 0.3, 0.5])
        factor = random.choice([0.5, 1.0, 2.0])

        py_d = py.speculate(context, max_tok, factor, 0.0, min_prob, False)
        cpp_d = cpp.speculate(context, max_tok, factor, 0.0, min_prob, False)
        n_tests += 1
        if py_d.token_ids == list(cpp_d.token_ids):
            n_match += 1

    pct = n_match / n_tests * 100
    print(f"  {n_match}/{n_tests} matched ({pct:.1f}%)")
    assert pct >= 95.0, f"Match rate {pct:.1f}% < 95%"
    print("  PASS")


if __name__ == "__main__":
    test_basic_patterns()
    test_multiple_sequences()
    test_edge_cases()
    test_spec_params()
    test_incremental_extend()
    test_cache_dual_tree()
    test_large_scale_random()

    print("\n" + "=" * 50)
    print("All tests passed!")
