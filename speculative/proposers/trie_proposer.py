"""
Trie-based N-gram Proposer.

Builds a token-level Trie over the context sequence and traverses it to find
the most likely continuation for the current context suffix.

Advantages over hash table:
  - Naturally handles variable-length n-gram matching in a single traversal.
  - Memory-efficient for sparse contexts (shared prefix compression).

Complexity:
    propose():  O(max_n)   — traverse at most max_n trie edges
    update():   O(context_len * max_n)  — insert all n-gram prefixes
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .base import BaseProposer


class _TrieNode:
    """A single node in the token Trie."""
    __slots__ = ("children", "next_freq", "best_next")

    def __init__(self):
        self.children: Dict[int, "_TrieNode"] = {}
        # Frequency of each token that was observed immediately after a path
        # leading to this node ends here.
        self.next_freq: Dict[int, int] = defaultdict(int)
        self.best_next: Optional[int] = None  # argmax(next_freq)


class TrieProposer(BaseProposer):
    """
    Trie-based N-gram proposer.

    The Trie stores all n-gram prefixes of the context (for n up to max_n).
    To propose the next token, we navigate from the root along the last
    min(max_n, ctx_len) tokens of the context and pick the most frequent
    continuation at the deepest node we can reach.
    """

    def __init__(self, min_n: int = 2, max_n: int = 5):
        super().__init__(min_n, max_n)
        self._root = _TrieNode()
        self._built_up_to: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def propose(self, context_ids: List[int], num_tokens: int) -> List[int]:
        if not context_ids:
            return []
        self._maybe_update(context_ids)

        draft: List[int] = []
        current = list(context_ids)

        for _ in range(num_tokens):
            token = self._query(current)
            if token is None:
                break
            draft.append(token)
            current.append(token)

        return draft

    def update(self, context_ids: List[int]) -> None:
        self._rebuild(context_ids)

    def reset(self) -> None:
        self._root = _TrieNode()
        self._built_up_to = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_update(self, context_ids: List[int]) -> None:
        if len(context_ids) > self._built_up_to:
            self._rebuild(context_ids)

    def _rebuild(self, context_ids: List[int]) -> None:
        """Insert new n-grams introduced since last rebuild."""
        n_len = len(context_ids)
        # For each position, insert paths of length min_n..max_n
        start = max(0, self._built_up_to - self.max_n)
        for i in range(start, n_len - self.min_n + 1):
            # Insert path for context_ids[i .. i+max_n-1], recording next token
            node = self._root
            for depth, tok in enumerate(context_ids[i : i + self.max_n], start=1):
                if depth > self.max_n:
                    break
                if tok not in node.children:
                    node.children[tok] = _TrieNode()
                node = node.children[tok]
                # Record the token that follows this prefix (if available)
                if i + depth < n_len and depth >= self.min_n:
                    next_tok = context_ids[i + depth]
                    node.next_freq[next_tok] += 1
                    # Update argmax
                    if (node.best_next is None or
                            node.next_freq[next_tok] > node.next_freq[node.best_next]):
                        node.best_next = next_tok
        self._built_up_to = n_len

    def _query(self, context_ids: List[int]) -> Optional[int]:
        """
        Find the longest n-gram (n = max_n..min_n) whose tokens match
        context_ids[-n:] exactly, and return its best_next.

        Critically, we try each n-gram length *independently* starting from
        the newest tokens (context[-n:]), not from the oldest.  This ensures
        every candidate n-gram ends at context[-1], so the returned token is
        always the prediction for what comes *after* the current context tail.

        (The naive single-pass "traverse from oldest token, keep deepest best"
        approach is wrong: if traversal stops at depth d, best_next reflects
        what followed context[-n : -n+d] — a prefix that may not include
        context[-1], so the "next token" prediction is actually a token that
        is *already in* the context.)
        """
        ctx_len = len(context_ids)
        for n in range(min(self.max_n, ctx_len), self.min_n - 1, -1):
            suffix = context_ids[ctx_len - n:]
            node = self._root
            for tok in suffix:
                if tok not in node.children:
                    break
                node = node.children[tok]
            else:
                # Full path matched — return prediction if available
                if node.best_next is not None:
                    return node.best_next
        return None
