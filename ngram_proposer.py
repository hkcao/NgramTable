# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import atexit
import logging
import os
import pickle
import threading
from collections import Counter
from typing import Optional

import numpy as np
import torch
from numba import get_num_threads, jit, njit, prange, set_num_threads

from vllm.config import VllmConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable to switch between hash-table and original KMP modes.
#   VLLM_NGRAM_USE_HASH="1"  -> hash table mode (default)
#   VLLM_NGRAM_USE_HASH="0"  -> original KMP mode
# ---------------------------------------------------------------------------
_USE_HASH_ENV = "VLLM_NGRAM_USE_HASH"

# Persistence flush threshold: flush ngramTable every N update steps
_FLUSH_EVERY = 100


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

        # Pre-allocate buffers for numba batch propose (KMP mode).
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
            self.num_numba_thread_available = min(1, (cpu_count // 2))
            self.num_numba_thread_available //= tp_size
        else:
            self.num_numba_thread_available = 1

        # ---- Hash table mode initialization ----
        # ngram_table_path: configurable via env VLLM_NGRAM_TABLE_PATH
        self.ngram_table_path: Optional[str] = os.environ.get(
            "VLLM_NGRAM_TABLE_PATH", None)

        # FreqTable: dict[tuple[int, ...], Counter[int]]
        self._freq_table: dict[tuple, Counter] = {}
        # HashTable: dict[int, int]  (hash of context -> best next token)
        self._hash_table: dict[int, int] = {}
        # Per-request tracking for incremental updates
        self._req_last_num_tokens: dict[int, int] = {}
        # Update counter for periodic persistence
        self._update_count: int = 0
        # Lock for async persistence
        self._persist_lock = threading.Lock()

        # Try to load persisted tables
        self._load_or_init_tables()

        # Register atexit for final persistence
        if self.ngram_table_path:
            atexit.register(self._persist_tables_sync)

        # Trigger Numba JIT compilation for N-gram proposer (KMP warmup).
        self.propose(
            [[]] * 1024,
            np.zeros(1024, dtype=np.int32),
            np.zeros((1024, self.max_model_len), dtype=np.int32),
        )

    # ------------------------------------------------------------------
    # Private: table I/O
    # ------------------------------------------------------------------

    def _load_or_init_tables(self) -> None:
        """Load FreqTable and HashTable from disk if available."""
        if self.ngram_table_path and os.path.exists(self.ngram_table_path):
            try:
                with open(self.ngram_table_path, "rb") as f:
                    data = pickle.load(f)
                self._freq_table = data.get("freq_table", {})
                self._hash_table = data.get("hash_table", {})
                logger.info(
                    "Loaded ngramTable from %s: freq_entries=%d, "
                    "hash_entries=%d",
                    self.ngram_table_path,
                    len(self._freq_table),
                    len(self._hash_table))
            except Exception:
                logger.warning(
                    "Failed to load ngramTable from %s, starting fresh",
                    self.ngram_table_path, exc_info=True)
                self._freq_table = {}
                self._hash_table = {}

    def _persist_tables_sync(self) -> None:
        """Synchronously persist FreqTable and HashTable to disk."""
        if not self.ngram_table_path:
            return
        try:
            with self._persist_lock:
                with open(self.ngram_table_path, "wb") as f:
                    pickle.dump({
                        "freq_table": self._freq_table,
                        "hash_table": self._hash_table,
                    }, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.debug("Persisted ngramTable to %s", self.ngram_table_path)
        except Exception:
            logger.warning("Failed to persist ngramTable", exc_info=True)

    def _persist_tables_async(self) -> None:
        """Asynchronously persist tables in a background thread."""
        if not self.ngram_table_path:
            return
        t = threading.Thread(target=self._persist_tables_sync, daemon=True)
        t.start()

    # ------------------------------------------------------------------
    # Private: FreqTable / HashTable build & update
    # ------------------------------------------------------------------

    def _build_tables_from_tokens(self, token_ids: list[int]) -> None:
        """Build FreqTable from a full token sequence, then derive HashTable.

        Uses sliding window of size n+1 for each n in [min_n, max_n].
        """
        for n in range(self.min_n, self.max_n + 1):
            for i in range(len(token_ids) - n):
                context = tuple(token_ids[i:i + n])
                next_token = token_ids[i + n]
                if context not in self._freq_table:
                    self._freq_table[context] = Counter()
                self._freq_table[context][next_token] += 1

        # Derive HashTable from FreqTable
        self._rebuild_hash_table()

    def _rebuild_hash_table(self) -> None:
        """Rebuild the entire HashTable from FreqTable."""
        self._hash_table.clear()
        for context, counter in self._freq_table.items():
            h = hash(context)
            best_token = counter.most_common(1)[0][0]
            self._hash_table[h] = best_token

    def _update_freq_table(self, token_ids: list[int], n_new: int) -> None:
        """Incremental update: add n-gram observations from the last n_new
        tokens appended to the sequence.

        For each n in [min_n, max_n], we only need to process windows that
        include at least one of the new tokens.
        """
        total = len(token_ids)
        for n in range(self.min_n, self.max_n + 1):
            # The earliest start position that includes a new token
            start = max(0, total - n_new - n)
            for i in range(start, total - n):
                context = tuple(token_ids[i:i + n])
                next_token = token_ids[i + n]
                if context not in self._freq_table:
                    self._freq_table[context] = Counter()
                old_best = self._freq_table[context].most_common(1)[0][0] \
                    if self._freq_table[context] else None
                self._freq_table[context][next_token] += 1
                new_best = self._freq_table[context].most_common(1)[0][0]
                # Only update HashTable if the argmax changed
                if new_best != old_best:
                    h = hash(context)
                    self._hash_table[h] = new_best

        self._update_count += 1
        if self._update_count % _FLUSH_EVERY == 0:
            self._persist_tables_async()

    # ------------------------------------------------------------------
    # Private: hash-table-based draft token proposal
    # ------------------------------------------------------------------

    def _propose_tokens_hash(
        self,
        input_ids: list[int],
        k: int,
    ) -> list[int]:
        """Generate k draft tokens by chained hash-table lookups.

        Starting from the last n tokens of input_ids, look up the hash table
        to get the next predicted token. Then shift the window by 1 (drop
        the oldest, append the predicted token) and repeat k times.

        Example: input_ids ends with [A, B, C], n=3
          hash((A,B,C)) -> E
          hash((B,C,E)) -> G
          hash((C,E,G)) -> F
          result: [E, G, F]
        """
        drafts: list[int] = []
        # Use the largest n that fits
        n = min(self.max_n, len(input_ids))
        if n < self.min_n:
            return drafts

        # Current context window (mutable list for shifting)
        window = list(input_ids[-n:])

        for _ in range(k):
            h = hash(tuple(window))
            next_token = self._hash_table.get(h)
            if next_token is None:
                # Try smaller n-grams as fallback
                found = False
                for fallback_n in range(n - 1, self.min_n - 1, -1):
                    fallback_window = tuple(window[-fallback_n:])
                    fh = hash(fallback_window)
                    next_token = self._hash_table.get(fh)
                    if next_token is not None:
                        found = True
                        break
                if not found:
                    break
            drafts.append(next_token)
            # Shift window: drop oldest, append predicted token
            window.pop(0)
            window.append(next_token)

        return drafts

    # ------------------------------------------------------------------
    # Core interface: _find_longest_matched_ngram_and_propose_tokens
    # (signature unchanged — dispatches to hash or KMP internally)
    # ------------------------------------------------------------------

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
                    1, min(self.num_numba_thread_available,
                           num_ngram_requests)
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
            if i in valid_ngram_requests \
                    and self.valid_ngram_num_drafts[i] > 0:
                draft_token_ids.append(
                    self.valid_ngram_draft[
                        i, :self.valid_ngram_num_drafts[i]].tolist()
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
        use_hash = os.environ.get(_USE_HASH_ENV, "1") == "1"

        if use_hash:
            return self._propose_hash_mode(
                sampled_token_ids, num_tokens_no_spec, token_ids_cpu)
        else:
            # ---- Original KMP path (unchanged) ----
            valid_ngram_requests = []
            for i, sampled_ids in enumerate(sampled_token_ids):
                num_sampled_ids = len(sampled_ids)
                if not num_sampled_ids:
                    continue
                num_tokens = num_tokens_no_spec[i]
                if num_tokens >= self.max_model_len:
                    continue
                valid_ngram_requests.append(i)

            draft_token_ids = self.batch_propose(
                len(sampled_token_ids),
                valid_ngram_requests,
                num_tokens_no_spec,
                token_ids_cpu,
            )
            return draft_token_ids

    def _propose_hash_mode(
        self,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
    ) -> list[list[int]]:
        """Hash-table mode: build/update tables and propose draft tokens."""
        draft_token_ids: list[list[int]] = []

        for i, sampled_ids in enumerate(sampled_token_ids):
            if not len(sampled_ids):
                draft_token_ids.append([])
                continue

            num_tokens = int(num_tokens_no_spec[i])
            if num_tokens >= self.max_model_len:
                draft_token_ids.append([])
                continue

            tokens = token_ids_cpu[i, :num_tokens].tolist()

            # Detect new request vs continuation
            prev_len = self._req_last_num_tokens.get(i)
            if prev_len is None or num_tokens < prev_len:
                # New request: build tables from prompt if tables are empty
                if not self._freq_table:
                    self._build_tables_from_tokens(tokens)
                else:
                    # Tables exist (loaded from disk or prior request),
                    # still update with this prompt's n-grams
                    self._update_freq_table(tokens, num_tokens)
            else:
                # Continuation: incremental update with new tokens
                n_new = num_tokens - prev_len
                if n_new > 0:
                    self._update_freq_table(tokens, n_new)

            self._req_last_num_tokens[i] = num_tokens

            # Query draft tokens
            k = min(self.k, self.max_model_len - num_tokens)
            if k <= 0:
                draft_token_ids.append([])
                continue

            drafts = self._propose_tokens_hash(tokens, k)
            draft_token_ids.append(drafts)

        return draft_token_ids

    def load_model(self, *args, **kwargs):
        # No model to load.
        pass


# ======================================================================
# Original KMP-based functions (unchanged, for VLLM_NGRAM_USE_HASH=0)
# ======================================================================

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
