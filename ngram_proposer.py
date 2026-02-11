# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from dataclasses import dataclass

import numpy as np
import torch
from numba import get_num_threads, jit, njit, prange, set_num_threads

from vllm.config import VllmConfig

# ---------------------------------------------------------------------------
# Constants for numba hash table proposer
# ---------------------------------------------------------------------------
_HASH_PRIME = np.int64(1000003)
_MIN_TABLE_SIZE = 1024
_MAX_TABLE_SIZE = 131072
_EMPTY_TOKEN = np.int32(-1)
_FP_LEN = 8  # Number of leading tokens for request fingerprint.


# ---------------------------------------------------------------------------
# Numba @njit helper functions for hash table operations
# ---------------------------------------------------------------------------

@njit(cache=True)
def _next_power_of_2(n):
    """Round up to next power of 2, clamped to [_MIN_TABLE_SIZE, _MAX_TABLE_SIZE]."""
    if n <= _MIN_TABLE_SIZE:
        return _MIN_TABLE_SIZE
    v = n - 1
    v |= v >> 1
    v |= v >> 2
    v |= v >> 4
    v |= v >> 8
    v |= v >> 16
    v += 1
    if v > _MAX_TABLE_SIZE:
        return _MAX_TABLE_SIZE
    return v


@njit(cache=True)
def _hash_ngram(tokens, start, length, mask):
    """Polynomial rolling hash of tokens[start:start+length] & mask."""
    h = np.int64(0)
    for j in range(length):
        h = (h * _HASH_PRIME + np.int64(tokens[start + j])) & np.int64(mask)
    return h


@njit(cache=True)
def _keys_equal(key_row, tokens, start, length):
    """Check if key_row[:length] == tokens[start:start+length]."""
    for j in range(length):
        if key_row[j] != tokens[start + j]:
            return False
    return True


@njit(cache=True)
def _build_tables_for_n(tokens, num_tokens, n, table_size,
                        freq_keys, freq_counts, freq_occupied,
                        lut_keys, lut_vals, lut_best_counts,
                        lut_occupied):
    """Build freq and lookup tables for one n-gram size.

    Iterates all positions, inserts (ngram, next_token) into freq table,
    and maintains the best next_token in the lookup table.
    """
    mask = np.int64(table_size - 1)

    for i in range(num_tokens - n):
        next_token = tokens[i + n]

        # Compute base ngram hash once, reuse for both tables.
        h_base = _hash_ngram(tokens, i, n, mask)

        # --- Insert into freq table: key = (tokens[i:i+n], next_token) ---
        h_freq = (h_base * _HASH_PRIME + np.int64(next_token)) & mask
        slot = int(h_freq)
        while True:
            if not freq_occupied[slot]:
                for j in range(n):
                    freq_keys[slot, j] = tokens[i + j]
                freq_keys[slot, n] = next_token
                freq_counts[slot] = 1
                freq_occupied[slot] = True
                break
            if _keys_equal(freq_keys[slot], tokens, i, n) and \
               freq_keys[slot, n] == next_token:
                freq_counts[slot] += 1
                break
            slot = int((slot + 1) & mask)

        count = freq_counts[slot]

        # --- Update lookup table: key = tokens[i:i+n] ---
        lut_slot = int(h_base)
        while True:
            if not lut_occupied[lut_slot]:
                for j in range(n):
                    lut_keys[lut_slot, j] = tokens[i + j]
                lut_vals[lut_slot] = next_token
                lut_best_counts[lut_slot] = count
                lut_occupied[lut_slot] = True
                break
            if _keys_equal(lut_keys[lut_slot], tokens, i, n):
                if count >= lut_best_counts[lut_slot]:
                    lut_vals[lut_slot] = next_token
                    lut_best_counts[lut_slot] = count
                break
            lut_slot = int((lut_slot + 1) & mask)


@njit(cache=True)
def _update_tables_for_n(tokens, old_len, new_len, n, table_size,
                         freq_keys, freq_counts, freq_occupied,
                         lut_keys, lut_vals, lut_best_counts,
                         lut_occupied):
    """Incrementally update tables with new tokens for one n-gram size."""
    mask = np.int64(table_size - 1)
    start = old_len - n
    if start < 0:
        start = 0
    end = new_len - n

    for i in range(start, end):
        next_token = tokens[i + n]

        # Compute base ngram hash once, reuse for both tables.
        h_base = _hash_ngram(tokens, i, n, mask)

        # --- Insert into freq table ---
        h_freq = (h_base * _HASH_PRIME + np.int64(next_token)) & mask
        slot = int(h_freq)
        while True:
            if not freq_occupied[slot]:
                for j in range(n):
                    freq_keys[slot, j] = tokens[i + j]
                freq_keys[slot, n] = next_token
                freq_counts[slot] = 1
                freq_occupied[slot] = True
                break
            if _keys_equal(freq_keys[slot], tokens, i, n) and \
               freq_keys[slot, n] == next_token:
                freq_counts[slot] += 1
                break
            slot = int((slot + 1) & mask)

        count = freq_counts[slot]

        # --- Update lookup table ---
        lut_slot = int(h_base)
        while True:
            if not lut_occupied[lut_slot]:
                for j in range(n):
                    lut_keys[lut_slot, j] = tokens[i + j]
                lut_vals[lut_slot] = next_token
                lut_best_counts[lut_slot] = count
                lut_occupied[lut_slot] = True
                break
            if _keys_equal(lut_keys[lut_slot], tokens, i, n):
                if count >= lut_best_counts[lut_slot]:
                    lut_vals[lut_slot] = next_token
                    lut_best_counts[lut_slot] = count
                break
            lut_slot = int((lut_slot + 1) & mask)


@njit(cache=True)
def _query_single_n(window, window_len, n, table_size,
                    lut_keys, lut_vals, lut_occupied):
    """Query the lookup table for a single n-gram size.

    Returns the best next token, or _EMPTY_TOKEN if not found.
    """
    if window_len < n:
        return _EMPTY_TOKEN
    mask = np.int64(table_size - 1)
    start = window_len - n
    h = _hash_ngram(window, start, n, mask)
    slot = int(h)
    while True:
        if not lut_occupied[slot]:
            return _EMPTY_TOKEN
        if _keys_equal(lut_keys[slot], window, start, n):
            return lut_vals[slot]
        slot = int((slot + 1) & mask)


@njit(cache=True)
def _query_lookup(tokens, num_tokens, min_n, max_n, k,
                  all_lut_keys, all_lut_vals, all_lut_occupied,
                  all_table_sizes, draft_out):
    """Query lookup tables to produce up to k draft tokens.

    Tries longest n-gram match first, falling back to shorter ones.
    all_lut_keys/vals/occupied are tuples of arrays, one per n-gram size
    (index 0 = min_n, index 1 = min_n+1, etc.).

    Returns number of draft tokens written to draft_out.
    """
    # Build a working window of the last max_n tokens.
    window_cap = max_n + k
    window = np.empty(window_cap, dtype=np.int32)
    if num_tokens >= max_n:
        wlen = max_n
        for j in range(max_n):
            window[j] = tokens[num_tokens - max_n + j]
    else:
        wlen = num_tokens
        for j in range(num_tokens):
            window[j] = tokens[j]

    num_drafted = 0
    for _ in range(k):
        found = False
        # Try longest match first.
        for n in range(max_n, min_n - 1, -1):
            idx = n - min_n
            result = _query_single_n(
                window, wlen, n,
                all_table_sizes[idx],
                all_lut_keys[idx], all_lut_vals[idx],
                all_lut_occupied[idx])
            if result != _EMPTY_TOKEN:
                draft_out[num_drafted] = result
                num_drafted += 1
                # Extend window.
                window[wlen] = result
                wlen += 1
                found = True
                break
        if not found:
            break

    return num_drafted


# ---------------------------------------------------------------------------
# Per-request hash table state container
# ---------------------------------------------------------------------------

@dataclass
class _HashTableState:
    """Per-request, per-n hash table arrays."""
    table_size: int
    freq_keys: np.ndarray       # (table_size, n+1) int32
    freq_counts: np.ndarray     # (table_size,) int32
    freq_occupied: np.ndarray   # (table_size,) bool
    lut_keys: np.ndarray        # (table_size, n) int32
    lut_vals: np.ndarray        # (table_size,) int32
    lut_best_counts: np.ndarray # (table_size,) int32
    lut_occupied: np.ndarray    # (table_size,) bool

    @staticmethod
    def allocate(table_size: int, n: int) -> '_HashTableState':
        return _HashTableState(
            table_size=table_size,
            freq_keys=np.zeros((table_size, n + 1), dtype=np.int32),
            freq_counts=np.zeros(table_size, dtype=np.int32),
            freq_occupied=np.zeros(table_size, dtype=np.bool_),
            lut_keys=np.zeros((table_size, n), dtype=np.int32),
            lut_vals=np.full(table_size, _EMPTY_TOKEN, dtype=np.int32),
            lut_best_counts=np.zeros(table_size, dtype=np.int32),
            lut_occupied=np.zeros(table_size, dtype=np.bool_),
        )


class NgramProposer:
    def __init__(self, vllm_config: VllmConfig):
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.prompt_lookup_min is not None
        assert vllm_config.speculative_config.prompt_lookup_max is not None

        # Minimum length of the n-gram to match.
        self.min_n = vllm_config.speculative_config.prompt_lookup_min
        # Maximum length of the n-gram to match.
        self.max_n = vllm_config.speculative_config.prompt_lookup_max
        # Number of tokens follow the match. If there are less than k
        # tokens follow the match, we will return the maximum amount of
        # tokens until the end.
        self.k = vllm_config.speculative_config.num_speculative_tokens
        # Maximum length of the model.
        self.max_model_len = vllm_config.model_config.max_model_len

        # Hash table mode toggle via environment variable.
        self.use_hash_table = os.environ.get(
            "VLLM_NGRAM_USE_HASH_TABLE", "0") == "1"

        if self.use_hash_table:
            # Per-request state: req_idx -> {n -> _HashTableState}
            self._req_tables: dict[int, dict[int, _HashTableState]] = {}
            # Maps req_idx -> last processed token count
            self._req_last_num_tokens: dict[int, int] = {}
            # Fingerprint (first _FP_LEN tokens) to detect request reuse.
            self._req_fingerprints: dict[int, tuple] = {}
            # Pre-allocated query buffers per request.
            self._req_query_cache: dict[int, dict] = {}

            # Trigger Numba JIT compilation for hash table functions.
            _warmup_hash_njit(self.min_n, self.max_n, self.k)
        else:
            # Pre-allocate buffers for numba batch propose.
            max_num_seqs = vllm_config.scheduler_config.max_num_seqs
            self.valid_ngram_draft = np.zeros(
                (max_num_seqs, self.k), dtype=np.int32)
            self.valid_ngram_num_drafts = np.zeros(
                (max_num_seqs), dtype=np.int32)

            # Threshold of total number of tokens in the batch to enable
            # multi-threading in numba batch propose.
            self.num_tokens_threshold = 8192
            tp_size = vllm_config.parallel_config.tensor_parallel_size
            cpu_count = os.cpu_count()
            # Max number of threads for numba parallel processing.
            if cpu_count:
                # Divide by 2 to use physical cores
                # and not logical cores (hyper-threading).
                # Cap the number of threads to 8 to avoid using too many
                # threads since other components like frontend (incl
                # tokenization) and Structured Outputs also use multiple
                # threads.
                # TODO(ekagra-ranjan): bump up the cap from 1 to 8
                # when TP parallelization for ngram is implemented.
                self.num_numba_thread_available = min(1, (cpu_count // 2))
                # Divide by tp_size to ensure each tensor parallel rank
                # has some threads since all ranks will run this.
                self.num_numba_thread_available //= tp_size
            else:
                self.num_numba_thread_available = 1

            # Trigger Numba JIT compilation for N-gram proposer.
            # This usually takes less than 1 second.
            self.propose(
                [[]] * 1024,
                np.zeros(1024, dtype=np.int32),
                np.zeros((1024, self.max_model_len), dtype=np.int32),
            )

    def _build_hash_tables(
        self, tokens: np.ndarray
    ) -> dict[int, _HashTableState]:
        """Build frequency and lookup tables from a token sequence.

        Args:
            tokens: 1D int32 numpy array of token IDs.

        Returns:
            dict mapping n -> _HashTableState with populated tables.
        """
        num_tokens = len(tokens)
        table_size = _next_power_of_2(num_tokens * 2)
        tables: dict[int, _HashTableState] = {}

        for n in range(self.min_n, self.max_n + 1):
            state = _HashTableState.allocate(table_size, n)
            _build_tables_for_n(
                tokens, num_tokens, n, table_size,
                state.freq_keys, state.freq_counts, state.freq_occupied,
                state.lut_keys, state.lut_vals, state.lut_best_counts,
                state.lut_occupied)
            tables[n] = state

        return tables

    def _update_hash_tables(
        self, tokens: np.ndarray, old_len: int, new_len: int,
        tables: dict[int, _HashTableState],
    ) -> dict[int, _HashTableState]:
        """Incrementally update hash tables with newly added tokens.

        If the load factor exceeds 0.6 for any n, rebuilds all tables
        with a larger size. Returns the (possibly new) tables dict.
        """
        needs_rebuild = False
        for n in range(self.min_n, self.max_n + 1):
            state = tables[n]
            _update_tables_for_n(
                tokens, old_len, new_len, n, state.table_size,
                state.freq_keys, state.freq_counts, state.freq_occupied,
                state.lut_keys, state.lut_vals, state.lut_best_counts,
                state.lut_occupied)
            # Check load factor on the lookup table.
            occupied = int(np.sum(state.lut_occupied))
            if occupied > state.table_size * 0.6:
                needs_rebuild = True

        if needs_rebuild:
            return self._build_hash_tables(tokens)
        return tables

    def _build_query_cache(
        self, tables: dict[int, _HashTableState],
    ) -> dict:
        """Pre-pack lookup arrays for fast repeated queries."""
        return {
            'lut_keys': tuple(tables[n].lut_keys
                              for n in range(self.min_n, self.max_n + 1)),
            'lut_vals': tuple(tables[n].lut_vals
                              for n in range(self.min_n, self.max_n + 1)),
            'lut_occupied': tuple(
                tables[n].lut_occupied
                for n in range(self.min_n, self.max_n + 1)),
            'table_sizes': np.array(
                [tables[n].table_size
                 for n in range(self.min_n, self.max_n + 1)],
                dtype=np.int64),
            'draft_out': np.empty(self.k, dtype=np.int32),
        }

    def _propose_hash(
        self,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
    ) -> list[list[int]]:
        """Hash-table-based ngram proposal for all requests in the batch.

        Args:
            sampled_token_ids: Sampled token IDs per request.
            num_tokens_no_spec: Number of non-speculative tokens per request.
            token_ids_cpu: Token ID buffer (batch_size x max_model_len).

        Returns:
            List of draft token ID lists, one per request.
        """
        draft_token_ids: list[list[int]] = []
        active_req_indices: set[int] = set()

        for i, sampled_ids in enumerate(sampled_token_ids):
            if not len(sampled_ids):
                draft_token_ids.append([])
                continue

            num_tokens = int(num_tokens_no_spec[i])
            if num_tokens >= self.max_model_len:
                draft_token_ids.append([])
                continue

            active_req_indices.add(i)
            tokens = token_ids_cpu[i, :num_tokens]

            # Fingerprint: first _FP_LEN tokens to detect request reuse.
            fp_len = min(_FP_LEN, num_tokens)
            fp = tuple(tokens[:fp_len].tolist())

            # Detect new request vs continuation.
            is_new = (i not in self._req_last_num_tokens
                      or num_tokens < self._req_last_num_tokens[i]
                      or self._req_fingerprints.get(i) != fp)

            if is_new:
                self._req_tables[i] = self._build_hash_tables(tokens)
                self._req_fingerprints[i] = fp
                self._req_query_cache[i] = \
                    self._build_query_cache(self._req_tables[i])
            else:
                old_len = self._req_last_num_tokens[i]
                if num_tokens > old_len:
                    self._req_tables[i] = self._update_hash_tables(
                        tokens, old_len, num_tokens,
                        self._req_tables[i])
                    self._req_query_cache[i] = \
                        self._build_query_cache(self._req_tables[i])

            self._req_last_num_tokens[i] = num_tokens

            # Query for draft tokens.
            k = min(self.k, self.max_model_len - num_tokens)
            if k <= 0:
                draft_token_ids.append([])
                continue

            qc = self._req_query_cache[i]
            draft_out = qc['draft_out']
            num_drafted = _query_lookup(
                tokens, num_tokens, self.min_n, self.max_n, k,
                qc['lut_keys'], qc['lut_vals'], qc['lut_occupied'],
                qc['table_sizes'], draft_out)
            draft_token_ids.append(draft_out[:num_drafted].tolist())

        # Cleanup stale requests no longer in the batch.
        stale = set(self._req_last_num_tokens.keys()) - active_req_indices
        for idx in stale:
            del self._req_tables[idx]
            del self._req_last_num_tokens[idx]
            self._req_fingerprints.pop(idx, None)
            self._req_query_cache.pop(idx, None)

        return draft_token_ids

    def batch_propose(
        self,
        num_requests: int,
        valid_ngram_requests: list,
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
    ) -> list[list[int]]:
        """Batch version of ngram proposer using numba for acceleration.

        Args:
            valid_ngram_requests:
                Set of indices of requests that need ngram proposals.
            num_tokens_no_spec:
                Numpy array of shape (batch_size,) representing the number
                of tokens without speculative tokens for each request.
            token_ids_cpu:
                Numpy array of shape (batch_size, max_model_len)
                representing the token IDs for each request.

        Returns:
            list[list[int]]:
                A list where each element is a list of proposed
                token IDs for the corresponding request.
        """
        draft_token_ids: list[list[int]] = []

        # Only run batch propose if there are requests needing ngram proposals.
        # avoid calling numba function with empty list which causes error
        # ValueError: cannot compute fingerprint of empty list
        if num_ngram_requests := len(valid_ngram_requests):
            original_num_numba_threads = get_num_threads()
            # Ensure we use at least one thread.
            # If total tokens is small, using multiple threads
            # may slow down due to overhead.
            total_tokens = np.sum(num_tokens_no_spec)
            if total_tokens >= self.num_tokens_threshold:
                final_num_threads = max(
                    1, min(self.num_numba_thread_available, num_ngram_requests)
                )
                set_num_threads(final_num_threads)
            else:
                set_num_threads(1)

            batch_propose_numba(
                valid_ngram_requests,
                num_tokens_no_spec,
                token_ids_cpu,
                self.min_n,
                self.max_n,
                self.max_model_len,
                self.k,
                self.valid_ngram_draft,
                self.valid_ngram_num_drafts,
            )

            # Restore original number of threads.
            set_num_threads(original_num_numba_threads)

        for i in range(num_requests):
            if i in valid_ngram_requests and self.valid_ngram_num_drafts[i] > 0:
                draft_token_ids.append(
                    self.valid_ngram_draft[i, : self.valid_ngram_num_drafts[i]].tolist()
                )
            else:
                draft_token_ids.append([])

        return draft_token_ids

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,  # unused
    ) -> list[list[int]]:
        if self.use_hash_table:
            return self._propose_hash(
                sampled_token_ids, num_tokens_no_spec, token_ids_cpu)

        # Original KMP-based path.
        valid_ngram_requests = []
        for i, sampled_ids in enumerate(sampled_token_ids):
            num_sampled_ids = len(sampled_ids)
            if not num_sampled_ids:
                # Skip speculative decoding.
                continue

            num_tokens = num_tokens_no_spec[i]
            if num_tokens >= self.max_model_len:
                # Skip requests that have already reached the max model length.
                continue

            valid_ngram_requests.append(i)

        draft_token_ids = self.batch_propose(
            len(sampled_token_ids),
            valid_ngram_requests,
            num_tokens_no_spec,
            token_ids_cpu,
        )

        return draft_token_ids

    def load_model(self, *args, **kwargs):
        # No model to load.
        pass


@njit(parallel=True)
def batch_propose_numba(
    valid_ngram_requests: list,
    num_tokens_no_spec: np.ndarray,
    token_ids_cpu: np.ndarray,
    min_n: int,
    max_n: int,
    max_model_len: int,
    k: int,
    valid_ngram_draft: np.ndarray,
    valid_ngram_num_drafts: np.ndarray,
):
    for i in prange(len(valid_ngram_requests)):
        idx = valid_ngram_requests[i]
        num_tokens = num_tokens_no_spec[idx]
        context_token_ids = token_ids_cpu[idx, :num_tokens]
        drafter_output = _find_longest_matched_ngram_and_propose_tokens(
            origin_tokens=context_token_ids,
            min_ngram=min_n,
            max_ngram=max_n,
            max_model_len=max_model_len,
            k=k,
        )

        valid_ngram_num_drafts[idx] = drafter_output.shape[0]
        if len(drafter_output):
            valid_ngram_draft[idx, : drafter_output.shape[0]] = drafter_output


@jit(nopython=True)
def _find_longest_matched_ngram_and_propose_tokens(
    origin_tokens: np.ndarray,
    min_ngram: int,
    max_ngram: int,
    max_model_len: int,
    k: int,
) -> np.ndarray:
    """
    Find the longest n-gram which matches the suffix of the given tokens
    whose length is within [min_ngram, max_ngram] (inclusive).

    If found, we will extract k right after the matched ngram.
    """
    # Do not generate draft tokens is context is shorter than minimum n-gram
    total_token = origin_tokens.shape[0]
    if total_token < min_ngram:
        return np.empty((0,), dtype=origin_tokens.dtype)

    # Do not generate draft tokens beyond the max model length.
    k = min(k, max_model_len - total_token)
    if k <= 0:
        return np.empty((0,), dtype=origin_tokens.dtype)

    # Flip tokens, and the goal become to find longest ngram
    # on the rightmost position which matches the prefix with
    # length [min_n, max_n] (inclusive).
    tokens = origin_tokens[::-1]

    # Longest prefix (not including itself) which is a suffix of
    # the current position.
    #   lps[i] = max{v, where tokens[0:v] == tokens[i+1-v:i+1]}
    #
    # As ngram is capped by max_ngram to save memory, we only need to
    # store lps for the first max_ngram prefix.
    lps = np.zeros(max_ngram, dtype=np.int32)

    longest_ngram = 0
    position = 0

    # lps[0] always equal to 0, we start with index 1
    prev_lps = 0
    i = 1
    while i < total_token:
        # tokens[:prev_lps] is the longest prefix as a suffix of tokens[:i]
        if tokens[prev_lps] == tokens[i]:
            # Token match: tokens[:prev_lps+1] is the longest prefix as
            # a suffix of tokens[:i+1]
            prev_lps += 1
            # Check if we found a longer valid ngram.
            #
            # Update position when longest_ngram matched prev_lps,
            # as we want to get the target n-gram of the earliest position
            # in the original tokens (i.e.
            # latest position in the reversed tokens)
            if prev_lps >= longest_ngram:
                longest_ngram = prev_lps
                position = i
            if i < max_ngram:
                # Store LPS for the first max_ngram prefix
                lps[i] = prev_lps
            if prev_lps == max_ngram:
                # When prev_lps reached max_ngram, update prev_lps
                # to lps[max_ngram-1] to avoid matching ngram
                # longer than max_ngram
                prev_lps = lps[max_ngram - 1]
            i += 1
        elif prev_lps != 0:
            # Token mismatch: try the second-longest prefix
            # among all suffix of tokens[:i],
            # which is the longest prefix of tokens[:prev_lps]
            prev_lps = lps[prev_lps - 1]
        else:
            # Token mismatch, and no more prefix (except empty string)
            # as a suffix of tokens[:i]
            i += 1

    if longest_ngram < min_ngram:
        # No valid ngram is found
        return np.empty((0,), dtype=origin_tokens.dtype)

    # Flip the position back, so in origin_tokens,
    # origin_tokens[total_token-1-position:total_token-1-position+longest_ngram]
    # is the matched ngram, so we should start drafting tokens from
    # total_token-1-position+longest_ngram
    start_position = total_token - 1 - position + longest_ngram
    k = min(k, total_token - start_position)
    return origin_tokens[start_position : start_position + k]


def _warmup_hash_njit(min_n: int, max_n: int, k: int):
    """Trigger JIT compilation of all hash table @njit functions."""
    dummy_tokens = np.arange(32, dtype=np.int32)
    num_tokens = len(dummy_tokens)
    table_size = _next_power_of_2(num_tokens * 2)
    num_n = max_n - min_n + 1

    all_lut_keys = []
    all_lut_vals = []
    all_lut_occupied = []
    all_table_sizes = np.empty(num_n, dtype=np.int64)

    for idx, n in enumerate(range(min_n, max_n + 1)):
        state = _HashTableState.allocate(table_size, n)
        _build_tables_for_n(
            dummy_tokens, num_tokens, n, table_size,
            state.freq_keys, state.freq_counts, state.freq_occupied,
            state.lut_keys, state.lut_vals, state.lut_best_counts,
            state.lut_occupied)
        # Also exercise update path.
        _update_tables_for_n(
            dummy_tokens, num_tokens - 2, num_tokens, n, table_size,
            state.freq_keys, state.freq_counts, state.freq_occupied,
            state.lut_keys, state.lut_vals, state.lut_best_counts,
            state.lut_occupied)
        all_lut_keys.append(state.lut_keys)
        all_lut_vals.append(state.lut_vals)
        all_lut_occupied.append(state.lut_occupied)
        all_table_sizes[idx] = table_size

    draft_out = np.empty(k, dtype=np.int32)
    _query_lookup(
        dummy_tokens, num_tokens, min_n, max_n, k,
        tuple(all_lut_keys), tuple(all_lut_vals), tuple(all_lut_occupied),
        all_table_sizes, draft_out)
