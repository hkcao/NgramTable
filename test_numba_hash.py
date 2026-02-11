"""
Unit test: verify numba hash table ngram proposer produces identical
results to the original Python dict implementation.
"""
import numpy as np
from collections import defaultdict


def _build_hash_tables_python(tokens, min_n, max_n):
    """Original Python dict implementation (reference)."""
    freq_tables = {}
    lookup_tables = {}
    num_tokens = len(tokens)

    for n in range(min_n, max_n + 1):
        freq = defaultdict(lambda: defaultdict(int))
        for i in range(num_tokens - n):
            key = tuple(tokens[i:i + n])
            next_token = int(tokens[i + n])
            freq[key][next_token] += 1

        lookup = {}
        for key, next_tokens in freq.items():
            lookup[key] = max(next_tokens, key=next_tokens.get)

        freq_tables[n] = {k: dict(v) for k, v in freq.items()}
        lookup_tables[n] = lookup

    return freq_tables, lookup_tables


def _propose_python(tokens, lookup_tables, min_n, max_n, k):
    """Original Python proposal (reference)."""
    predictions = []
    window = list(tokens[-max_n:]) if len(tokens) >= max_n else list(tokens)

    for _ in range(k):
        found = False
        for n in range(max_n, min_n - 1, -1):
            if len(window) < n:
                continue
            key = tuple(window[-n:])
            if n in lookup_tables and key in lookup_tables[n]:
                next_token = lookup_tables[n][key]
                predictions.append(next_token)
                window.append(next_token)
                found = True
                break
        if not found:
            break

    return predictions


def _update_python(tokens, old_len, new_len, freq_tables, lookup_tables,
                   min_n, max_n):
    """Original Python incremental update (reference)."""
    for n in range(min_n, max_n + 1):
        freq = freq_tables.get(n, {})
        lookup = lookup_tables.get(n, {})
        if n not in freq_tables:
            freq_tables[n] = freq
            lookup_tables[n] = lookup

        start = max(0, old_len - n)
        end = new_len - n
        for i in range(start, end):
            key = tuple(tokens[i:i + n])
            next_token = int(tokens[i + n])

            if key not in freq:
                freq[key] = {next_token: 1}
                lookup[key] = next_token
            else:
                counts = freq[key]
                counts[next_token] = counts.get(next_token, 0) + 1
                if counts[next_token] >= counts.get(lookup[key], 0):
                    lookup[key] = next_token


def test_basic_correctness():
    """Test that numba hash tables produce identical proposals to Python."""
    from vllm.v1.spec_decode.ngram_proposer import (
        _HashTableState, _build_tables_for_n, _next_power_of_2,
        _query_lookup, _update_tables_for_n,
    )

    min_n, max_n, k = 2, 5, 8
    np.random.seed(42)

    # Simulate a token sequence with lots of repetition to ensure proposals.
    vocab = np.arange(10, dtype=np.int32)
    tokens = np.random.choice(vocab, size=200, replace=True).astype(np.int32)

    # --- Python reference ---
    freq_py, lookup_py = _build_hash_tables_python(tokens, min_n, max_n)
    proposals_py = _propose_python(tokens, lookup_py, min_n, max_n, k)

    # --- Numba implementation ---
    num_tokens = len(tokens)
    table_size = _next_power_of_2(num_tokens * 2)
    tables = {}
    for n in range(min_n, max_n + 1):
        state = _HashTableState.allocate(table_size, n)
        _build_tables_for_n(
            tokens, num_tokens, n, table_size,
            state.freq_keys, state.freq_counts, state.freq_occupied,
            state.lut_keys, state.lut_vals, state.lut_best_counts,
            state.lut_occupied)
        tables[n] = state

    # Query
    num_n = max_n - min_n + 1
    all_lut_keys = tuple(tables[n].lut_keys for n in range(min_n, max_n + 1))
    all_lut_vals = tuple(tables[n].lut_vals for n in range(min_n, max_n + 1))
    all_lut_occupied = tuple(tables[n].lut_occupied
                             for n in range(min_n, max_n + 1))
    all_table_sizes = np.array(
        [tables[n].table_size for n in range(min_n, max_n + 1)],
        dtype=np.int64)
    draft_out = np.empty(k, dtype=np.int32)
    num_drafted = _query_lookup(
        tokens, num_tokens, min_n, max_n, k,
        all_lut_keys, all_lut_vals, all_lut_occupied,
        all_table_sizes, draft_out)
    proposals_numba = draft_out[:num_drafted].tolist()

    print(f"Python proposals:  {proposals_py}")
    print(f"Numba proposals:   {proposals_numba}")
    assert proposals_py == proposals_numba, \
        f"Mismatch!\n  Python: {proposals_py}\n  Numba:  {proposals_numba}"
    print("PASS: basic correctness")


def test_incremental_update():
    """Test that incremental update produces same results as full rebuild."""
    from vllm.v1.spec_decode.ngram_proposer import (
        _HashTableState, _build_tables_for_n, _next_power_of_2,
        _query_lookup, _update_tables_for_n,
    )

    min_n, max_n, k = 2, 4, 5
    np.random.seed(123)
    vocab = np.arange(10, dtype=np.int32)
    tokens_full = np.random.choice(vocab, size=150, replace=True).astype(
        np.int32)

    # Split: build from first 100, then update with the rest.
    old_len = 100
    new_len = 150
    tokens = tokens_full[:new_len]

    table_size = _next_power_of_2(new_len * 2)

    # --- Full rebuild ---
    tables_full = {}
    for n in range(min_n, max_n + 1):
        state = _HashTableState.allocate(table_size, n)
        _build_tables_for_n(
            tokens, new_len, n, table_size,
            state.freq_keys, state.freq_counts, state.freq_occupied,
            state.lut_keys, state.lut_vals, state.lut_best_counts,
            state.lut_occupied)
        tables_full[n] = state

    # --- Build + incremental update ---
    tables_inc = {}
    for n in range(min_n, max_n + 1):
        state = _HashTableState.allocate(table_size, n)
        _build_tables_for_n(
            tokens, old_len, n, table_size,
            state.freq_keys, state.freq_counts, state.freq_occupied,
            state.lut_keys, state.lut_vals, state.lut_best_counts,
            state.lut_occupied)
        _update_tables_for_n(
            tokens, old_len, new_len, n, table_size,
            state.freq_keys, state.freq_counts, state.freq_occupied,
            state.lut_keys, state.lut_vals, state.lut_best_counts,
            state.lut_occupied)
        tables_inc[n] = state

    # Query both and compare.
    def query(tables):
        all_lut_keys = tuple(tables[n].lut_keys
                             for n in range(min_n, max_n + 1))
        all_lut_vals = tuple(tables[n].lut_vals
                             for n in range(min_n, max_n + 1))
        all_lut_occupied = tuple(tables[n].lut_occupied
                                 for n in range(min_n, max_n + 1))
        all_table_sizes = np.array(
            [tables[n].table_size for n in range(min_n, max_n + 1)],
            dtype=np.int64)
        draft_out = np.empty(k, dtype=np.int32)
        num = _query_lookup(
            tokens, new_len, min_n, max_n, k,
            all_lut_keys, all_lut_vals, all_lut_occupied,
            all_table_sizes, draft_out)
        return draft_out[:num].tolist()

    full_proposals = query(tables_full)
    inc_proposals = query(tables_inc)

    print(f"Full rebuild proposals:       {full_proposals}")
    print(f"Incremental update proposals: {inc_proposals}")
    assert full_proposals == inc_proposals, \
        f"Mismatch!\n  Full: {full_proposals}\n  Inc:  {inc_proposals}"
    print("PASS: incremental update")


def test_larger_sequences():
    """Test on larger, more realistic token sequences."""
    from vllm.v1.spec_decode.ngram_proposer import (
        _HashTableState, _build_tables_for_n, _next_power_of_2,
        _query_lookup,
    )

    min_n, max_n, k = 3, 7, 8
    np.random.seed(999)

    # Simulate a 2000-token sequence with a small vocab to get repetitions.
    tokens = np.random.randint(0, 20, size=2000, dtype=np.int32)
    # Also add some explicit repeated patterns to test frequency tracking.
    pattern = tokens[100:110]
    tokens[500:510] = pattern
    tokens[1000:1010] = pattern
    tokens[1500:1510] = pattern

    # Python reference
    freq_py, lookup_py = _build_hash_tables_python(tokens, min_n, max_n)
    proposals_py = _propose_python(tokens, lookup_py, min_n, max_n, k)

    # Numba
    num_tokens = len(tokens)
    table_size = _next_power_of_2(num_tokens * 2)
    tables = {}
    for n in range(min_n, max_n + 1):
        state = _HashTableState.allocate(table_size, n)
        _build_tables_for_n(
            tokens, num_tokens, n, table_size,
            state.freq_keys, state.freq_counts, state.freq_occupied,
            state.lut_keys, state.lut_vals, state.lut_best_counts,
            state.lut_occupied)
        tables[n] = state

    all_lut_keys = tuple(tables[n].lut_keys for n in range(min_n, max_n + 1))
    all_lut_vals = tuple(tables[n].lut_vals for n in range(min_n, max_n + 1))
    all_lut_occupied = tuple(tables[n].lut_occupied
                             for n in range(min_n, max_n + 1))
    all_table_sizes = np.array(
        [tables[n].table_size for n in range(min_n, max_n + 1)],
        dtype=np.int64)
    draft_out = np.empty(k, dtype=np.int32)
    num_drafted = _query_lookup(
        tokens, num_tokens, min_n, max_n, k,
        all_lut_keys, all_lut_vals, all_lut_occupied,
        all_table_sizes, draft_out)
    proposals_numba = draft_out[:num_drafted].tolist()

    print(f"Large seq: Python proposals:  {proposals_py}")
    print(f"Large seq: Numba proposals:   {proposals_numba}")
    assert proposals_py == proposals_numba, \
        f"Mismatch!\n  Python: {proposals_py}\n  Numba:  {proposals_numba}"
    print("PASS: larger sequences")


def test_deterministic_pattern():
    """Test with a sequence that has known repeated patterns at the end."""
    from vllm.v1.spec_decode.ngram_proposer import (
        _HashTableState, _build_tables_for_n, _next_power_of_2,
        _query_lookup,
    )

    min_n, max_n, k = 2, 4, 5
    # Sequence where the last 4 tokens [10,20,30,40] appear earlier
    # followed by [50,60,70,80,90].
    tokens = np.array([
        1, 2, 3, 10, 20, 30, 40, 50, 60, 70, 80, 90,
        4, 5, 6, 10, 20, 30, 40, 50, 60, 70, 80, 90,
        7, 8, 9, 10, 20, 30, 40  # ends with the pattern
    ], dtype=np.int32)

    # Python reference
    freq_py, lookup_py = _build_hash_tables_python(tokens, min_n, max_n)
    proposals_py = _propose_python(tokens, lookup_py, min_n, max_n, k)

    # Numba
    num_tokens = len(tokens)
    table_size = _next_power_of_2(num_tokens * 2)
    tables = {}
    for n in range(min_n, max_n + 1):
        state = _HashTableState.allocate(table_size, n)
        _build_tables_for_n(
            tokens, num_tokens, n, table_size,
            state.freq_keys, state.freq_counts, state.freq_occupied,
            state.lut_keys, state.lut_vals, state.lut_best_counts,
            state.lut_occupied)
        tables[n] = state

    all_lut_keys = tuple(tables[n].lut_keys for n in range(min_n, max_n + 1))
    all_lut_vals = tuple(tables[n].lut_vals for n in range(min_n, max_n + 1))
    all_lut_occupied = tuple(tables[n].lut_occupied
                             for n in range(min_n, max_n + 1))
    all_table_sizes = np.array(
        [tables[n].table_size for n in range(min_n, max_n + 1)],
        dtype=np.int64)
    draft_out = np.empty(k, dtype=np.int32)
    num_drafted = _query_lookup(
        tokens, num_tokens, min_n, max_n, k,
        all_lut_keys, all_lut_vals, all_lut_occupied,
        all_table_sizes, draft_out)
    proposals_numba = draft_out[:num_drafted].tolist()

    print(f"Deterministic: Python proposals:  {proposals_py}")
    print(f"Deterministic: Numba proposals:   {proposals_numba}")
    assert proposals_py == proposals_numba, \
        f"Mismatch!\n  Python: {proposals_py}\n  Numba:  {proposals_numba}"
    assert len(proposals_py) > 0, "Expected non-empty proposals!"
    # We expect [50, 60, 70, 80, 90] since the pattern matches.
    assert proposals_py == [50, 60, 70, 80, 90], \
        f"Expected [50,60,70,80,90] but got {proposals_py}"
    print("PASS: deterministic pattern")


def test_empty_and_short():
    """Test edge cases: empty tokens, fewer tokens than min_n."""
    from vllm.v1.spec_decode.ngram_proposer import (
        _HashTableState, _build_tables_for_n, _next_power_of_2,
        _query_lookup,
    )

    min_n, max_n, k = 2, 5, 3

    for size in [0, 1, 2, 3]:
        tokens = np.arange(size, dtype=np.int32)
        num_tokens = len(tokens)
        table_size = _next_power_of_2(max(num_tokens * 2, 1))

        tables = {}
        for n in range(min_n, max_n + 1):
            state = _HashTableState.allocate(table_size, n)
            if num_tokens > n:
                _build_tables_for_n(
                    tokens, num_tokens, n, table_size,
                    state.freq_keys, state.freq_counts, state.freq_occupied,
                    state.lut_keys, state.lut_vals, state.lut_best_counts,
                    state.lut_occupied)
            tables[n] = state

        all_lut_keys = tuple(tables[n].lut_keys
                             for n in range(min_n, max_n + 1))
        all_lut_vals = tuple(tables[n].lut_vals
                             for n in range(min_n, max_n + 1))
        all_lut_occupied = tuple(tables[n].lut_occupied
                                 for n in range(min_n, max_n + 1))
        all_table_sizes = np.array(
            [tables[n].table_size for n in range(min_n, max_n + 1)],
            dtype=np.int64)
        draft_out = np.empty(k, dtype=np.int32)
        num_drafted = _query_lookup(
            tokens, num_tokens, min_n, max_n, k,
            all_lut_keys, all_lut_vals, all_lut_occupied,
            all_table_sizes, draft_out)
        proposals = draft_out[:num_drafted].tolist()
        print(f"  size={size}: proposals={proposals}")

    print("PASS: empty and short sequences")


if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: Basic correctness")
    print("=" * 60)
    test_basic_correctness()

    print()
    print("=" * 60)
    print("Test 2: Incremental update")
    print("=" * 60)
    test_incremental_update()

    print()
    print("=" * 60)
    print("Test 3: Larger sequences")
    print("=" * 60)
    test_larger_sequences()

    print()
    print("=" * 60)
    print("Test 4: Deterministic pattern")
    print("=" * 60)
    test_deterministic_pattern()

    print()
    print("=" * 60)
    print("Test 5: Edge cases")
    print("=" * 60)
    test_empty_and_short()

    print()
    print("ALL TESTS PASSED")
