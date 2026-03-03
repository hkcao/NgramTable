"""
Trie-based N-gram Proposer.

Design mirrors the LookaheadCache + Tree structure from:
    alipay/PainlessInferenceAcceleration / lookahead / common / lookahead_cache.py

Core data model
---------------
  mem[t]  — for each context token t, a dict {next_token: _TrieNode} that
             represents all sequences seen immediately AFTER t.
             Equivalent to LookaheadCache.mem[t] = Tree(t), where Tree.nodes
             is the root children dict.

  _TrieNode.freq
            — how many times this path was traversed during insertion;
             equivalent to Node.freqs[-1] in the reference implementation
             (output-mode frequency).

  _TrieNode.children
            — {token_id: _TrieNode}, same as Node.children.

Insertion (_put / _rebuild)
---------------------------
  For every position i in the context:
      mem[context[i]].put(context[i+1 : i+max_n+1])

  Walking the path increments freq at each node along the way.
  Equivalent to Tree._put called from LookaheadCache.put / stream_put.

Lookup (_match / _find_nodes / propose)
----------------------------------------
  To propose after context_ids, try n = max_n … min_n:
      root   = context[-n]           → look up mem[root]
      suffix = context[-n+1:]        → match this path in mem[root]
      If the full suffix matches, the children of the last matched node are
      the candidate next tokens; pick the one with highest freq.

  After the first token is found, follow the greedy chain inside the same
  sub-trie (children → children) for subsequent draft tokens, exactly as
  Tree.get_one_branch does.

  Equivalent to LookaheadCache.hier_get + Tree.get_one_branch.
"""

from typing import Dict, List, Optional

from .base import BaseProposer


class _TrieNode:
    """Single node in the token trie.

    Mirrors Node(children, freqs) from lookahead_cache.py, simplified to a
    single scalar freq (output-mode only, no input-mode tracking needed).
    """
    __slots__ = ("children", "freq")

    def __init__(self):
        self.children: Dict[int, "_TrieNode"] = {}
        self.freq: float = 0.0   # ≡ Node.freqs[-1]


class TrieProposer(BaseProposer):
    """
    Token-trie n-gram proposer aligned with PainlessInferenceAcceleration.

    Attribute:
        mem  — {token_id: root_children_dict}
               equivalent to LookaheadCache.mem where mem[t] is Tree(t)
               and root_children_dict ≡ Tree.nodes.
    """

    def __init__(self, min_n: int = 2, max_n: int = 5):
        super().__init__(min_n, max_n)
        # mem[t] = root children dict of the sub-trie keyed by token t
        self.mem: Dict[int, Dict[int, _TrieNode]] = {}
        self._built_up_to: int = 0   # len(context_ids) at last _rebuild call

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def propose(self, context_ids: List[int], num_tokens: int) -> List[int]:
        if not context_ids:
            return []
        self._maybe_update(context_ids)

        draft: List[int] = []
        current = list(context_ids)

        # Initial lookup via hier_get-style search
        nodes = self._find_nodes(current)

        for _ in range(num_tokens):
            if not nodes:
                # Chain exhausted — re-lookup from extended current context
                # (equivalent to hier_get retrying with a fresh token_ids window)
                nodes = self._find_nodes(current)
                if not nodes:
                    break

            # Greedy pick: argmax over children freqs (≡ get_one_branch argmax)
            best_tok = max(nodes, key=lambda k: nodes[k].freq)
            if nodes[best_tok].freq <= 0:
                nodes = {}
                continue

            draft.append(best_tok)
            current.append(best_tok)
            # Follow greedy chain in same sub-trie (≡ get_one_branch nodes = max_node.children)
            nodes = nodes[best_tok].children

        return draft

    def update(self, context_ids: List[int]) -> None:
        self._rebuild(context_ids)

    def reset(self) -> None:
        self.mem = {}
        self._built_up_to = 0

    # ------------------------------------------------------------------
    # Insertion  (≡ Tree._put called from LookaheadCache.put)
    # ------------------------------------------------------------------

    def _put(self, nodes: Dict[int, _TrieNode], token_ids: List[int]) -> None:
        """Insert token_ids as a path under nodes, incrementing freq at each node.

        Iterative equivalent of Tree._put:
            while token_ids:
                t = token_ids[0]
                node = nodes.get(t) or create
                node.freqs[-1] += freq
                nodes = node.children
                token_ids = token_ids[1:]
        """
        for tok in token_ids:
            if tok not in nodes:
                nodes[tok] = _TrieNode()
            node = nodes[tok]
            node.freq += 1.0
            nodes = node.children

    def _maybe_update(self, context_ids: List[int]) -> None:
        if len(context_ids) > self._built_up_to:
            self._rebuild(context_ids)

    def _rebuild(self, context_ids: List[int]) -> None:
        """Insert all n-gram windows introduced since the last rebuild.

        For each position i, inserts context[i+1 : i+max_n+1] into mem[context[i]].
        Overlaps by 1 position (re-processes the last previously-seen position) to
        capture tail growth when new tokens extend a window that was previously short.

        Equivalent to the per-position loop in LookaheadCache.put:
            for i in range(ts - 1):
                token_id = token_ids[i]
                tup = token_ids[i+1 : i+branch_length+1]
                mem[token_id].put(tup)
        """
        n_len = len(context_ids)
        # Overlap by 1 so that the last position's tail is updated when context grows
        start = max(0, self._built_up_to - 1)
        for i in range(start, n_len - 1):
            t = context_ids[i]
            if t not in self.mem:
                self.mem[t] = {}
            tup = context_ids[i + 1 : i + self.max_n + 1]
            self._put(self.mem[t], tup)
        self._built_up_to = n_len

    # ------------------------------------------------------------------
    # Lookup  (≡ LookaheadCache.hier_get + Tree._match + get_one_branch)
    # ------------------------------------------------------------------

    def _find_nodes(
        self, context_ids: List[int]
    ) -> Dict[int, _TrieNode]:
        """Return candidate children dict for what follows context_ids.

        Tries n-gram lengths from max_n down to min_n:
            root   = context[-n]           (≡ LookaheadCache.hier_get iterates over t)
            suffix = context[-n+1:]        (≡ ids = token_ids[i+1:] passed to Tree.get)
            _match(mem[root], suffix)      (≡ Tree._match)

        Returns the children dict of the deepest matched node, i.e., the set
        of candidate tokens that followed the matched n-gram in training data.
        """
        ctx_len = len(context_ids)
        for n in range(min(self.max_n, ctx_len), self.min_n - 1, -1):
            root_tok = context_ids[ctx_len - n]
            root_children = self.mem.get(root_tok)
            if root_children is None:
                continue
            suffix = context_ids[ctx_len - n + 1:]   # last n-1 tokens
            matched = self._match(root_children, suffix)
            if matched:
                return matched
        return {}

    def _match(
        self,
        nodes: Dict[int, _TrieNode],
        token_ids: List[int],
    ) -> Dict[int, _TrieNode]:
        """Follow token_ids path through nodes.

        Returns children dict of the last matched node, or {} if any token
        is not found (path broken).

        Equivalent to Tree._match (output mode):
            for token_id in token_ids:
                node = nodes.get(token_id)
                nodes = {}
                if node is None: break
                if node.freqs.get(-1, 0) > 0:
                    nodes = node.children
            return token_id, nodes
        """
        for tok in token_ids:
            node = nodes.get(tok)
            if node is None or node.freq <= 0:
                return {}
            nodes = node.children
        return nodes
