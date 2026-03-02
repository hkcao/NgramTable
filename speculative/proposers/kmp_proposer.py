"""
KMP-style N-gram Proposer.

Mirrors the vLLM prompt-lookup speculative decoding algorithm:
  For each n-gram length from max_n down to min_n, scan the context for the
  longest suffix that matches an earlier position, then propose the tokens
  that follow that match.

This is sometimes called "prompt lookup" because it can copy tokens verbatim
from the prompt / context rather than requiring a model forward pass.

Complexity:
    propose():  O(context_len * max_n)  — linear scan per n-gram length
    update():   O(1)  — stateless, no index to maintain
"""

from typing import List

from .base import BaseProposer


class KMPProposer(BaseProposer):
    """
    Longest-match n-gram proposer using a simple sliding-window scan.

    Strategy:
        1. Take the last `n` tokens of the context as the query n-gram
           (n decreasing from max_n to min_n).
        2. Scan the context (excluding the last n tokens) for that n-gram.
        3. On a match, collect the tokens immediately following it as draft.
        4. Stop at the first (longest) match found.
    """

    def propose(self, context_ids: List[int], num_tokens: int) -> List[int]:
        if len(context_ids) < self.min_n + 1:
            return []

        ctx = context_ids
        ctx_len = len(ctx)

        for n in range(min(self.max_n, ctx_len - 1), self.min_n - 1, -1):
            query = ctx[ctx_len - n:]          # last n tokens
            # Scan positions where the n-gram could start (not the tail itself)
            search_end = ctx_len - n           # exclusive upper bound
            for i in range(search_end):
                if ctx[i : i + n] == query:
                    match_start = i + n        # tokens after the matched n-gram
                    available = ctx_len - match_start
                    take = min(num_tokens, available)
                    if take > 0:
                        return list(ctx[match_start : match_start + take])
        return []

    def update(self, context_ids: List[int]) -> None:
        pass  # stateless
