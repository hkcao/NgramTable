"""
Hash-Table N-gram Proposer.

Builds a per-request frequency table mapping n-gram → {next_token: count}.
The best (most frequent) next token for each n-gram is cached in a lookup
dict for O(1) proposal queries.

Compared to KMP:
  - Proposal:     O(max_n)  vs O(context_len * max_n) for KMP
  - Update cost:  O(context_len * (max_n - min_n + 1)) for initial build
  - Memory:       proportional to unique n-grams in context

Design is intentionally pure-Python (no Numba / vLLM) so it runs on macOS.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .base import BaseProposer

# Type aliases
_NGram = Tuple[int, ...]
_FreqTable = Dict[_NGram, Dict[int, int]]   # ngram -> {next_tok: count}
_LUTTable = Dict[_NGram, int]               # ngram -> best_next_tok


class HashTableProposer(BaseProposer):
    """
    Frequency-driven n-gram proposer backed by Python dicts.

    The internal tables are rebuilt from scratch whenever ``update()`` is
    called, which keeps the implementation simple while still supporting
    incremental usage (re-build is cheap for typical context lengths).
    """

    def __init__(self, min_n: int = 2, max_n: int = 5):
        super().__init__(min_n, max_n)
        # freq[n][ngram][next_token] = count
        self._freq: Dict[int, _FreqTable] = {
            n: defaultdict(lambda: defaultdict(int))
            for n in range(min_n, max_n + 1)
        }
        # lut[n][ngram] = best_next_token (argmax of freq)
        self._lut: Dict[int, _LUTTable] = {
            n: {} for n in range(min_n, max_n + 1)
        }
        self._built_up_to: int = 0  # last context length when tables were built

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def propose(self, context_ids: List[int], num_tokens: int) -> List[int]:
        if not context_ids:
            return []
        # Incrementally update tables if context has grown
        self._maybe_update(context_ids)

        draft: List[int] = []
        current = list(context_ids)

        for _ in range(num_tokens):
            token = self._lookup_best(current)
            if token is None:
                break
            draft.append(token)
            current.append(token)

        return draft

    def update(self, context_ids: List[int]) -> None:
        """Rebuild tables from the full context."""
        self._rebuild(context_ids)

    def reset(self) -> None:
        for n in range(self.min_n, self.max_n + 1):
            self._freq[n] = defaultdict(lambda: defaultdict(int))
            self._lut[n] = {}
        self._built_up_to = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_update(self, context_ids: List[int]) -> None:
        """Only rebuild when context has grown since last update."""
        if len(context_ids) > self._built_up_to:
            self._rebuild(context_ids)

    def _rebuild(self, context_ids: List[int]) -> None:
        """Full rebuild of freq + lut from context_ids."""
        n_len = len(context_ids)
        for n in range(self.min_n, self.max_n + 1):
            freq_n = self._freq[n]
            lut_n = self._lut[n]
            # Only process new positions (incremental)
            start = max(0, self._built_up_to - n)
            for i in range(start, n_len - n):
                ngram = tuple(context_ids[i : i + n])
                next_tok = context_ids[i + n]
                freq_n[ngram][next_tok] += 1
                # Update LUT: keep argmax
                if ngram not in lut_n or (
                    freq_n[ngram][next_tok] > freq_n[ngram].get(lut_n[ngram], 0)
                ):
                    lut_n[ngram] = next_tok
        self._built_up_to = n_len

    def _lookup_best(self, context_ids: List[int]) -> Optional[int]:
        """Return the most frequent next token for the longest matching n-gram."""
        ctx_len = len(context_ids)
        for n in range(min(self.max_n, ctx_len), self.min_n - 1, -1):
            ngram = tuple(context_ids[ctx_len - n :])
            lut_n = self._lut.get(n)
            if lut_n and ngram in lut_n:
                return lut_n[ngram]
        return None
