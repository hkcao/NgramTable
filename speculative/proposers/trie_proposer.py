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
        Traverse the trie using the last min(max_n, ctx_len) tokens of context.
        Return the best next token at the deepest node reached with next_freq.
        """
        ctx_len = len(context_ids)
        suffix_len = min(self.max_n, ctx_len)
        suffix = context_ids[ctx_len - suffix_len :]

        node = self._root
        best: Optional[int] = None

        for tok in suffix:
            if tok not in node.children:
                break
            node = node.children[tok]
            if node.best_next is not None:
                best = node.best_next  # keep deepest valid best

        return best
