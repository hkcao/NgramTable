# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import atexit
import logging
import os
import pickle
import threading
from collections import Counter, defaultdict
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

# ---------------------------------------------------------------------------
# Environment variable to switch to Trie-based prediction mode.
#   VLLM_NGRAM_USE_TRIE="1"  -> Trie mode (overrides hash mode)
#   VLLM_NGRAM_USE_TRIE="0"  -> disabled (default)
# ---------------------------------------------------------------------------
_USE_TRIE_ENV = "VLLM_NGRAM_USE_TRIE"
_USE_TRIE_FUZZY_ENV = "VLLM_TRIE_FUZZY"
_USE_SKIPGRAM_ENV = "VLLM_NGRAM_USE_SKIPGRAM"
_USE_TRIE_SKIPGRAM_ENV = "VLLM_TRIE_SKIPGRAM"
_SKIPGRAM_SENTINEL = -999
_USE_TRIE_EDITDIST_ENV = "VLLM_TRIE_EDIT_DIST"
_USE_TRIE_EDITDIST_BUILD_ENV = "VLLM_TRIE_EDIT_DIST_BUILD"
_MAX_EDIT_VARIANTS = 3  # max similar tokens per position during build

# Persistence flush threshold: flush ngramTable every N update steps
_FLUSH_EVERY = 100


# ======================================================================
# Edit distance utilities (BK-tree for token-string fuzzy matching)
# ======================================================================

def _edit_distance(s1: str, s2: str) -> int:
    """Levenshtein edit distance between two strings."""
    m, n = len(s1), len(s2)
    if m < n:
        s1, s2 = s2, s1
        m, n = n, m
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(m):
        curr = [i + 1]
        for j in range(n):
            curr.append(min(
                prev[j + 1] + 1,
                curr[j] + 1,
                prev[j] + (0 if s1[i] == s2[j] else 1)
            ))
        prev = curr
    return prev[n]


class BKTree:
    """BK-tree for efficient nearest-neighbor search by edit distance."""

    def __init__(self):
        self.root = None

    def insert(self, word: str, token_id: int):
        if self.root is None:
            self.root = (word, token_id, {})
            return
        current = self.root
        while True:
            d = _edit_distance(current[0], word)
            if d == 0:
                return
            if d in current[2]:
                current = current[2][d]
            else:
                current[2][d] = (word, token_id, {})
                return

    def search(self, word: str, max_dist: int) -> list[tuple[int, int]]:
        """Find all tokens within max_dist.
        Returns [(token_id, dist), ...], excluding exact matches (d=0)."""
        if self.root is None:
            return []
        results = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            d = _edit_distance(node[0], word)
            if 0 < d <= max_dist:
                results.append((node[1], d))
            for dist, child in node[2].items():
                if d - max_dist <= dist <= d + max_dist:
                    stack.append(child)
        return results


# ======================================================================
# Trie-based draft token prediction (adapted from LookaheadCache)
# ======================================================================


class TrieNode:
    """A single node in the Trie, corresponding to LookaheadCache.Node."""
    __slots__ = ['freqs', 'children']

    def __init__(self, children=None, freqs=None):
        self.children: dict[int, 'TrieNode'] = children if children is not None else {}
        self.freqs: dict[int, float] = freqs if freqs is not None else {}

    def __repr__(self):
        return f'{list(self.children.keys())}:{self.freqs}'


class TrieTree:
    """A per-root-token Trie managing successor sequences.

    Corresponds to LookaheadCache.Tree.
    """

    def __init__(self, token_id: int, max_node: int = 65536,
                 max_output_node: int = 512):
        self.token_id = token_id
        self.max_node = max_node
        self.max_output_node = max_output_node
        self.n_node = 0
        self.n_output_node = 0
        self.nodes: dict = {}

    # ---- write path ----

    def put(self, token_ids: list[int], mode: str = 'output', idx: int = 0,
            freq: float = 1.0, skip_gram: bool = False):
        assert mode in ('input', 'output')
        if mode == 'output':
            idx = -1
        self._put(token_ids, self.nodes, mode=mode, idx=idx, freq=freq)
        # Insert skip-gram variants: for each position, replace that token
        # with sentinel and insert the variant sequence.  The sentinel node
        # aggregates frequencies across different tokens at that position.
        if skip_gram and len(token_ids) > 1:
            for skip_pos in range(len(token_ids)):
                variant = list(token_ids)
                variant[skip_pos] = _SKIPGRAM_SENTINEL
                self._put(variant, self.nodes, mode=mode, idx=idx,
                          freq=freq)

    def _make_key(self, token_ids: list[int], pos: int) -> tuple:
        """Make a node key from pos.  Returns (key, advance_count).
        Key is always a single token (int), advance is always 1."""
        return token_ids[pos], 1

    def _put(self, token_ids: list[int], nodes: dict, mode: str = 'output',
             freq: float = 1.0, idx: int = -1):
        pos = 0
        while pos < len(token_ids):
            key, width = self._make_key(token_ids, pos)
            node = nodes.get(key, None)
            if node is None:
                n = self._pack(token_ids, pos, idx, freq=freq)
                nodes.update(n)
                # count remaining chunks
                p = pos
                cnt = 0
                while p < len(token_ids):
                    _, w = self._make_key(token_ids, p)
                    p += w
                    cnt += 1
                self.n_node += cnt
                if mode == 'output':
                    self.n_output_node += cnt
                break
            node.freqs[idx] = node.freqs.get(idx, 0.0) + freq
            nodes = node.children
            pos += width

    def _pack(self, token_ids: list[int], start: int, idx: int,
              freq: float = 1.0) -> dict:
        """Build a chain of new nodes for token_ids[start:]."""
        # Collect keys from start to end
        keys = []
        p = start
        while p < len(token_ids):
            key, width = self._make_key(token_ids, p)
            keys.append(key)
            p += width
        # Build chain backwards
        ps: dict = {}
        for key in keys[::-1]:
            freqs = {idx: freq}
            p = TrieNode(ps, freqs)
            ps = {key: p}
        return ps

    # ---- read path ----

    def _match(self, token_ids: list[int], mode: str = 'mix',
               idx: int = 0, skip_gram: bool = False) -> tuple:
        """Walk token_ids through the tree.  Returns (last_token, children).
        last_token is the last individual token of the deepest matched chunk.
        When skip_gram=True, if exact child not found, try sentinel child.
        """
        nodes = self.nodes
        token_id = None
        if len(token_ids) == 0:
            return token_id, nodes

        pos = 0
        while pos < len(token_ids):
            key, width = self._make_key(token_ids, pos)
            node = nodes.get(key, None)
            # Skip-gram fallback: try sentinel child
            if node is None and skip_gram:
                node = nodes.get(_SKIPGRAM_SENTINEL, None)
            nodes = {}
            if node is None:
                token_id = token_ids[pos + width - 1]
                break
            token_id = token_ids[pos + width - 1]
            if mode == 'input':
                if node.freqs.get(idx, 0.0) > 0:
                    nodes = node.children
            elif mode == 'output':
                if node.freqs.get(-1, 0.0) > 0:
                    nodes = node.children
            else:
                if node.freqs.get(idx, 0.0) > 0 or node.freqs.get(-1, 0.0) > 0:
                    nodes = node.children
            pos += width

        return token_id, nodes

    def _match_fuzzy(self, token_ids: list[int], mode: str = 'mix',
                     idx: int = 0, wild_budget: int = 1) -> tuple:
        """Fuzzy _match: allow up to wild_budget token mismatches.
        Returns (last_token, children) of the deepest-reaching path."""
        if len(token_ids) == 0:
            return None, self.nodes

        best_depth = [0]
        best_children = [{}]

        def _check_freq(node):
            if mode == 'input':
                return node.freqs.get(idx, 0.0) > 0
            elif mode == 'output':
                return node.freqs.get(-1, 0.0) > 0
            return (node.freqs.get(idx, 0.0) > 0
                    or node.freqs.get(-1, 0.0) > 0)

        def _dfs(nodes, pos, budget, depth):
            if pos >= len(token_ids):
                if depth > best_depth[0]:
                    best_depth[0] = depth
                    best_children[0] = nodes
                return
            if not nodes:
                return
            key, width = self._make_key(token_ids, pos)
            for child_key, child_node in nodes.items():
                is_exact = (child_key == key)
                if not is_exact and budget <= 0:
                    continue
                if not _check_freq(child_node):
                    continue
                new_budget = budget if is_exact else budget - 1
                _dfs(child_node.children, pos + width, new_budget,
                     depth + 1)

        _dfs(self.nodes, 0, wild_budget, 0)
        last_token = token_ids[-1] if token_ids else None
        return last_token, best_children[0]

    def _match_editdist(self, token_ids: list[int], mode: str = 'mix',
                        idx: int = 0, id2str: dict | None = None,
                        max_dist: int = 1) -> tuple:
        """Walk token_ids through tree with edit distance fallback.
        When exact match fails, try the child with smallest edit distance
        (within max_dist) to the query token string."""
        nodes = self.nodes
        token_id = None
        if len(token_ids) == 0:
            return token_id, nodes

        pos = 0
        while pos < len(token_ids):
            key, width = self._make_key(token_ids, pos)
            node = nodes.get(key, None)
            # Edit distance fallback: find closest child
            if node is None and id2str:
                query_str = id2str.get(key, '')
                if query_str:
                    best_node = None
                    best_dist = max_dist + 1
                    for child_key, child_node in nodes.items():
                        if child_key == _SKIPGRAM_SENTINEL:
                            continue
                        if isinstance(child_key, tuple):
                            continue
                        child_str = id2str.get(child_key, '')
                        if child_str:
                            d = _edit_distance(query_str, child_str)
                            if d <= max_dist and d < best_dist:
                                best_dist = d
                                best_node = child_node
                    node = best_node
            nodes = {}
            if node is None:
                token_id = token_ids[pos + width - 1]
                break
            token_id = token_ids[pos + width - 1]
            if mode == 'input':
                if node.freqs.get(idx, 0.0) > 0:
                    nodes = node.children
            elif mode == 'output':
                if node.freqs.get(-1, 0.0) > 0:
                    nodes = node.children
            else:
                if node.freqs.get(idx, 0.0) > 0 \
                        or node.freqs.get(-1, 0.0) > 0:
                    nodes = node.children
            pos += width

        return token_id, nodes

    def get(self, token_ids: list[int], max_size: int = 64,
            max_length: int = 8, min_input_size: int = 0,
            min_output_size: int = 0, output_weight: float = 0.5,
            mode: str = 'mix', idx: int = 0, wild_budget: int = 0,
            skip_gram: bool = False, edit_dist: int = 0,
            id2str: dict | None = None):
        """Multi-branch query — faithful reproduction of Tree.get."""
        assert mode in ('input', 'output', 'mix')

        match_token_id, nodes = self._match(token_ids, mode=mode, idx=idx)
        if len(nodes) == 0 and skip_gram:
            match_token_id, nodes = self._match(
                token_ids, mode=mode, idx=idx, skip_gram=True)
        if len(nodes) == 0 and edit_dist > 0 and id2str is not None:
            match_token_id, nodes = self._match_editdist(
                token_ids, mode=mode, idx=idx,
                id2str=id2str, max_dist=edit_dist)
        if len(nodes) == 0 and wild_budget > 0:
            match_token_id, nodes = self._match_fuzzy(
                token_ids, mode=mode, idx=idx, wild_budget=wild_budget)
        if len(nodes) == 0:
            token_id = token_ids[-1] if len(token_ids) > 0 \
                else self.token_id
            return [token_id], np.ones((1, 1), dtype=np.int64), [0, 0]

        freqs: list = []
        self._dfs_get_freqs(nodes, freqs, idx, output_weight)

        min_mix_freq = 1e9
        min_input_freq = 1e9
        min_output_freq = 1e9
        if mode == 'input':
            output_weight = 0.0
            size = len([x for x in freqs if x[1] > 0])
            if size > max_size:
                input_freqs = sorted(
                    freqs, key=lambda x: x[1], reverse=True)
                min_input_freq = input_freqs[min_input_size - 1][1]
            else:
                min_input_freq = 0.0
        elif mode == 'output':
            output_weight = 1.0
            size = len([x for x in freqs if x[2] > 0])
            if size > max_size:
                output_freqs = sorted(
                    freqs, key=lambda x: x[2], reverse=True)
                min_output_freq = output_freqs[min_output_size - 1][2]
            else:
                min_output_freq = 0.0
        else:
            size = len([x for x in freqs if x[1] > 0 or x[2] > 0])
            if size > max_size:
                indices = set()
                if min_input_size > 0:
                    input_freqs = sorted(
                        freqs, key=lambda x: x[1], reverse=True)
                    min_input_freq = input_freqs[min_input_size - 1][1]
                    indices.update(
                        [x[0] for x in input_freqs[:min_input_size]])
                if min_output_size > 0:
                    output_freqs = sorted(
                        freqs, key=lambda x: x[2], reverse=True)
                    min_output_freq = \
                        output_freqs[min_output_size - 1][2]
                    indices.update(
                        [x[0] for x in output_freqs[:min_output_size]])
                if len(indices) < max_size:
                    mix_freqs = sorted(
                        freqs, key=lambda x: x[3], reverse=True)
                    rest_size = max_size - len(indices)
                    indices.update(
                        [x[0] for x in mix_freqs[:rest_size]])
                    cur_size = len(indices)
                    for ii in range(rest_size,
                                    min(rest_size + max_size, size)):
                        if mix_freqs[ii][0] in indices:
                            continue
                        cur_size += 1
                        if cur_size >= max_size:
                            x = mix_freqs[ii]
                            min_mix_freq = x[3]
                            break
            else:
                min_mix_freq = 0.0

        mask = np.zeros((max_size, max_size), dtype=np.int64)
        mask[:, 0] = 1
        ids = [match_token_id or self.token_id]
        sizes = [0, 0]
        self._ravel(nodes, ids, mask, -1,
                    max_size=max_size,
                    max_length=max_length,
                    min_output_freq=min_output_freq,
                    min_input_freq=min_input_freq,
                    min_mix_freq=min_mix_freq,
                    sizes=sizes,
                    output_weight=output_weight,
                    mode=mode,
                    idx=idx)
        size = len(ids)
        mask = mask[:size, :size]
        return ids, mask, sizes

    def _dfs_get_freqs(self, nodes: dict, freqs: list, idx: int,
                       output_weight: float):
        for node in nodes.values():
            fo = node.freqs.get(-1, 0.0)
            fi = node.freqs.get(idx, 0.0)
            if fo > 0 or fi > 0:
                fm = (1.0 - output_weight) * fi + output_weight * fo
                freqs.append([None, fi, fo, fm])
                if len(node.children) > 0:
                    self._dfs_get_freqs(node.children, freqs, idx,
                                        output_weight)

    def _ravel(self, nodes: dict, ids: list, mask, pid: int,
               max_size: int = 64, max_length: int = 8,
               min_output_freq: float = 1.0, min_input_freq: float = 1.0,
               min_mix_freq: float = 1.0, output_weight: float = 0.5,
               sizes: list | None = None, mode: str = 'mix',
               idx: int = 0):
        if len(ids) >= max_size or max_length <= 0:
            return

        sorts = [
            (k, v,
             (1.0 - output_weight) * v.freqs.get(idx, 0.0)
             + output_weight * v.freqs.get(-1, 0.0))
            for k, v in nodes.items()
        ]
        sorts = sorted(sorts, key=lambda x: x[2], reverse=True)
        for tid, node, fm in sorts:
            if len(ids) >= max_size:
                return
            fi = node.freqs.get(idx, 0.0)
            fo = node.freqs.get(-1, 0.0)
            if mode == 'mix':
                if fi < min_input_freq and fo < min_output_freq \
                        and fm < min_mix_freq:
                    continue
            elif mode == 'input':
                if fi < min_input_freq:
                    continue
            else:
                if fo < min_output_freq:
                    continue
            if fi > 0.0:
                sizes[0] += 1
            if fo > 0.0:
                sizes[1] += 1

            # Expand n-gram tuple keys to individual tokens
            if isinstance(tid, tuple):
                prev = pid
                for t in tid:
                    if len(ids) >= max_size:
                        return
                    ids.append(t)
                    rid = len(ids) - 1
                    if prev > -1:
                        mask[rid] = mask[prev]
                    mask[rid, rid] = 1
                    prev = rid
                rid = prev
            else:
                ids.append(tid)
                rid = len(ids) - 1
                if pid > -1:
                    mask[rid] = mask[pid]
                mask[rid, rid] = 1

            if len(node.children) > 0:
                self._ravel(node.children, ids, mask, rid,
                            max_size=max_size,
                            max_length=max_length - 1,
                            min_output_freq=min_output_freq,
                            min_input_freq=min_input_freq,
                            min_mix_freq=min_mix_freq,
                            output_weight=output_weight,
                            sizes=sizes,
                            mode=mode,
                            idx=idx)

    def get_one_branch(self, token_ids: list[int], max_length: int = 8,
                       mode: str = 'mix', idx: int = 0,
                       wild_budget: int = 0, skip_gram: bool = False,
                       edit_dist: int = 0,
                       id2str: dict | None = None):
        """Single-branch query — faithful reproduction of
        Tree.get_one_branch."""
        assert mode in ('input', 'output', 'mix')

        match_token_id, nodes = self._match(token_ids, mode=mode, idx=idx)
        if len(nodes) == 0 and skip_gram:
            match_token_id, nodes = self._match(
                token_ids, mode=mode, idx=idx, skip_gram=True)
        if len(nodes) == 0 and edit_dist > 0 and id2str is not None:
            match_token_id, nodes = self._match_editdist(
                token_ids, mode=mode, idx=idx,
                id2str=id2str, max_dist=edit_dist)
        if len(nodes) == 0 and wild_budget > 0:
            match_token_id, nodes = self._match_fuzzy(
                token_ids, mode=mode, idx=idx, wild_budget=wild_budget)
        if len(nodes) == 0:
            token_id = token_ids[-1] if len(token_ids) > 0 \
                else self.token_id
            return [token_id], np.ones((1, 1), dtype=np.int64), [0, 0]

        ids = [match_token_id or self.token_id]
        length = 0
        while True:
            if len(nodes) == 0 or length >= max_length:
                break
            max_freq = 0.0
            max_node = None
            max_id = None
            if mode == 'mix':
                for t, node in nodes.items():
                    fo = node.freqs.get(idx, 0.0)
                    fi = node.freqs.get(-1, 0.0)
                    if fo > 0 or fi > 0:
                        freq = fi + fo
                        if freq > max_freq:
                            max_freq = freq
                            max_node = node
                            max_id = t
            elif mode == 'input':
                for t, node in nodes.items():
                    freq = node.freqs.get(idx, 0.0)
                    if freq > 0:
                        if freq > max_freq:
                            max_freq = freq
                            max_node = node
                            max_id = t
            else:
                for t, node in nodes.items():
                    freq = node.freqs.get(-1, 0.0)
                    if freq > 0:
                        if freq > max_freq:
                            max_freq = freq
                            max_node = node
                            max_id = t
            if max_node is None:
                break
            # Expand n-gram tuple keys to individual tokens
            if isinstance(max_id, tuple):
                ids.extend(max_id)
                length += len(max_id)
            else:
                ids.append(max_id)
                length += 1
            nodes = max_node.children

        return (ids,
                np.tril(np.ones((length + 1, length + 1), dtype=np.int64),
                        0),
                [length])

    # ---- pruning ----

    def squeeze(self):
        if self.n_node > self.max_node or \
                self.n_output_node > self.max_output_node:
            self._squeeze(self.nodes)
            sizes = [0]
            self._count_node(self.nodes, sizes)
            self.n_node = sizes[0]
            self.n_output_node = sizes[0]

    def _squeeze(self, nodes: dict):
        for t, p in list(nodes.items()):
            fo = p.freqs.get(-1, 0.0)
            if fo > 1.0:
                p.freqs[-1] *= 0.5
                if len(p.children) > 0:
                    self._squeeze(p.children)
            else:
                nodes.pop(t)

    def _count_node(self, nodes: dict, sizes: list):
        sizes[0] += len(nodes)
        for t, n in nodes.items():
            if len(n.children) > 0:
                self._count_node(n.children, sizes)

    # ---- input freq reset ----

    def reset_input_freq(self, idx: int):
        if len(self.nodes) == 0:
            return
        self._reset_input_freq(self.nodes, idx)

    def _reset_input_freq(self, nodes: dict, idx: int):
        for t, node in nodes.items():
            f = node.freqs.get(idx, 0.0)
            if f == 0.0:
                continue
            node.freqs[idx] = 0.0
            if len(node.children) > 0:
                self._reset_input_freq(node.children, idx)


class TrieCache:
    """Top-level Trie cache, corresponding to LookaheadCache.

    ``mem`` maps each root key to a TrieTree that stores all
    successor sequences starting with that key.

    When ``node_size`` > 1, root keys are tuples of ``node_size``
    tokens.
    """

    def __init__(self, max_node: int = 65536, max_output_node: int = 512,
                 node_size: int = 1, skip_gram: bool = False,
                 edit_dist: int = 0, edit_dist_build: bool = False,
                 id2str: dict | None = None,
                 bk_tree: BKTree | None = None):
        self.max_node = max_node
        self.max_output_node = max_output_node
        self.node_size = node_size
        self.skip_gram = skip_gram  # insert/lookup skip-variant root keys
        self.edit_dist = edit_dist  # edit distance threshold (0=disabled)
        self.edit_dist_build = edit_dist_build  # insert variants during build
        self.id2str = id2str or {}  # token_id -> token string
        self.bk_tree = bk_tree  # BK-tree for similar token search
        self._similar_cache: dict[int, list[int]] = {}  # lazy cache
        self.mem: dict = {}  # key: int (node_size=1) or tuple
        self._output_ids: defaultdict[int, list] = defaultdict(list)
        self._update_trees: set[TrieTree] = set()
        self._update_input_trees: set[TrieTree] = set()

    # ---- skip-gram root key helpers ----

    def _skip_root_variants(self, root_key) -> list[tuple]:
        """Generate skip-gram variant root keys (replace one position with
        sentinel). Only for node_size > 1."""
        if not self.skip_gram or self.node_size <= 1:
            return []
        rk = root_key if isinstance(root_key, tuple) else (root_key,)
        variants = []
        for skip_pos in range(len(rk)):
            v = list(rk)
            v[skip_pos] = _SKIPGRAM_SENTINEL
            variants.append(tuple(v))
        return variants

    # ---- edit-distance helpers ----

    def _get_similar(self, token_id: int) -> list[int]:
        """Get similar token IDs by edit distance, with lazy caching."""
        if token_id in self._similar_cache:
            return self._similar_cache[token_id]
        if not self.bk_tree or token_id not in self.id2str:
            self._similar_cache[token_id] = []
            return []
        word = self.id2str[token_id]
        results = self.bk_tree.search(word, self.edit_dist)
        results.sort(key=lambda x: x[1])  # sort by distance
        similar = [r[0] for r in results[:_MAX_EDIT_VARIANTS]]
        self._similar_cache[token_id] = similar
        return similar

    def _edit_root_variants(self, root_key) -> list:
        """Generate edit-distance variant root keys."""
        if self.edit_dist <= 0 or not self.bk_tree:
            return []
        if self.node_size == 1:
            return self._get_similar(root_key)
        # For node_size > 1: replace one position at a time
        rk = root_key if isinstance(root_key, tuple) else (root_key,)
        variants = []
        for pos in range(len(rk)):
            for sim_id in self._get_similar(rk[pos]):
                v = list(rk)
                v[pos] = sim_id
                variants.append(tuple(v))
        return variants[:_MAX_EDIT_VARIANTS * self.node_size]

    def _put_to_tree(self, root_key, tup: list[int], mode: str, idx: int,
                     tid: int, freq: float = 1.0,
                     skip_gram: bool | None = None):
        """Insert tup into the TrieTree for root_key, creating if needed."""
        sg = self.skip_gram if skip_gram is None else skip_gram
        tree = self.mem.get(root_key, None)
        if tree is not None:
            tree.put(tup, mode=mode, idx=idx, freq=freq, skip_gram=sg)
            self._update_trees.add(tree)
        else:
            tree = TrieTree(tid, max_node=self.max_node,
                            max_output_node=self.max_output_node)
            tree.put(tup, mode=mode, idx=idx, freq=freq, skip_gram=sg)
            self.mem[root_key] = tree
        if mode == 'input':
            self._update_input_trees.add(tree)
        return tree

    # ---- bulk write (prompt / input) ----

    def put(self, token_ids: list[int], branch_length: int = 8,
            mode: str = 'output', idx: int = 0):
        ns = self.node_size
        if len(token_ids) < ns + 1:
            return
        ts = len(token_ids)
        for i in range(ts - ns + 1):
            root_key = token_ids[i] if ns == 1 else tuple(token_ids[i:i + ns])
            tup = token_ids[i + ns:i + ns + branch_length]
            # token_id = last token of root n-gram (anchor for ids[0])
            tid = root_key if ns == 1 else root_key[-1]
            self._put_to_tree(root_key, tup, mode, idx, tid)
            # Insert skip-gram root variants
            for skip_key in self._skip_root_variants(root_key):
                self._put_to_tree(skip_key, tup, mode, idx, tid)
            # Insert edit-distance variants (only if build mode enabled)
            if self.edit_dist_build:
                # Root variants
                for ed_key in self._edit_root_variants(root_key):
                    ed_tid = ed_key if ns == 1 else ed_key[-1]
                    self._put_to_tree(ed_key, tup, mode, idx, ed_tid,
                                      freq=0.5, skip_gram=False)
                # Internal variants
                if len(tup) > 0:
                    tree = self.mem.get(root_key)
                    if tree:
                        for pos in range(len(tup)):
                            for sim_id in self._get_similar(
                                    tup[pos])[:_MAX_EDIT_VARIANTS]:
                                variant = list(tup)
                                variant[pos] = sim_id
                                tree.put(variant, mode=mode, idx=idx,
                                         freq=0.5, skip_gram=False)

    # ---- streaming write (generated output) ----

    def stream_put(self, token_ids: list[int], branch_length: int = 8,
                   final: bool = False, idx: int = 0):
        ns = self.node_size
        self._output_ids[idx].extend(token_ids)
        output_ids = self._output_ids[idx]
        ts = len(output_ids)
        min_branch_length = 1 if final else branch_length
        if ts > min_branch_length:
            for i in range(ts - min_branch_length):
                if i + ns > ts:
                    break
                root_key = output_ids[i] if ns == 1 \
                    else tuple(output_ids[i:i + ns])
                tup = output_ids[i + ns:i + ns + branch_length]
                tid = root_key if ns == 1 else root_key[-1]
                self._put_to_tree(root_key, tup, 'output', idx, tid)
                # Insert skip-gram root variants
                for skip_key in self._skip_root_variants(root_key):
                    self._put_to_tree(skip_key, tup, 'output', idx, tid)
                # Insert edit-distance variants (only if build mode enabled)
                if self.edit_dist_build:
                    for ed_key in self._edit_root_variants(root_key):
                        ed_tid = ed_key if ns == 1 else ed_key[-1]
                        self._put_to_tree(ed_key, tup, 'output', idx,
                                          ed_tid, freq=0.5, skip_gram=False)
                    if len(tup) > 0:
                        tree = self.mem.get(root_key)
                        if tree:
                            for pos in range(len(tup)):
                                for sim_id in self._get_similar(
                                        tup[pos])[:_MAX_EDIT_VARIANTS]:
                                    variant = list(tup)
                                    variant[pos] = sim_id
                                    tree.put(variant, mode='output',
                                             idx=idx, freq=0.5,
                                             skip_gram=False)
            if not final:
                self._output_ids[idx] = output_ids[ts - branch_length:]
        if final:
            self._output_ids[idx] = []
            self.reset_input_freqs(idx)
            self.squeeze_branch_counts()

    # ---- query (faithful reproduction of LookaheadCache) ----

    def hier_get(self, token_ids: list[int], decoding_length: int = 64,
                 branch_length: int = 8, min_input_size: int = 0,
                 min_output_size: int = 0, mode: str = 'mix',
                 idx: int = 0, wild_budget: int = 0):
        """Multi-branch sliding-window query — faithful reproduction of
        LookaheadCache.hier_get."""
        default_mask = np.ones((1, 1), dtype=np.int64)

        if decoding_length <= 1 or branch_length == 0:
            return token_ids[-1:], default_mask, []

        ns = self.node_size
        decoding_ids = None
        decoding_masks = default_mask
        sizes = [0, 0]
        # First pass: exact match only (original behavior)
        for i in range(len(token_ids) - ns + 1):
            root_key = token_ids[i] if ns == 1 \
                else tuple(token_ids[i:i + ns])
            tree = self.mem.get(root_key, None)
            # Skip-gram fallback: try skip-variant root keys
            if tree is None and self.skip_gram and ns > 1:
                for skip_key in self._skip_root_variants(root_key):
                    tree = self.mem.get(skip_key, None)
                    if tree is not None:
                        break
            if tree is not None:
                ids = token_ids[i + ns:]
                decoding_ids, decoding_masks, sizes = tree.get(
                    ids,
                    max_size=decoding_length,
                    max_length=branch_length,
                    min_input_size=min_input_size,
                    min_output_size=min_output_size,
                    mode=mode,
                    idx=idx,
                    wild_budget=wild_budget,
                    skip_gram=self.skip_gram)
                s = len(decoding_ids)
                if s >= branch_length:
                    break

        # Second pass: edit-distance fallback (only if exact match failed)
        if (decoding_ids is None or len(decoding_ids) <= 1) \
                and self.edit_dist > 0:
            for i in range(len(token_ids) - ns + 1):
                root_key = token_ids[i] if ns == 1 \
                    else tuple(token_ids[i:i + ns])
                # Try similar root keys
                tree = None
                for ed_key in self._edit_root_variants(root_key):
                    tree = self.mem.get(ed_key, None)
                    if tree is not None:
                        break
                # Also try exact root with internal edit-dist matching
                if tree is None:
                    tree = self.mem.get(root_key, None)
                if tree is not None:
                    ids = token_ids[i + ns:]
                    decoding_ids, decoding_masks, sizes = tree.get(
                        ids,
                        max_size=decoding_length,
                        max_length=branch_length,
                        min_input_size=min_input_size,
                        min_output_size=min_output_size,
                        mode=mode,
                        idx=idx,
                        wild_budget=wild_budget,
                        skip_gram=self.skip_gram,
                        edit_dist=self.edit_dist,
                        id2str=self.id2str)
                    s = len(decoding_ids)
                    if s >= branch_length:
                        break

        if decoding_ids is None:
            decoding_ids = token_ids[-1:]

        return decoding_ids, decoding_masks, sizes

    def one_get(self, token_ids: list[int], decoding_length: int = 64,
                branch_length: int = 8, min_input_size: int = 0,
                min_output_size: int = 0, mode: str = 'mix',
                idx: int = 0, wild_budget: int = 0):
        """Single-branch sliding-window query — faithful reproduction of
        LookaheadCache.one_get."""
        default_mask = np.ones((1, 1), dtype=np.int64)

        if decoding_length <= 1 or branch_length == 0:
            return token_ids[-1:], default_mask, []

        ns = self.node_size
        decoding_ids = None
        decoding_masks = default_mask
        sizes = [0, 0]
        # First pass: exact match only
        for i in range(len(token_ids) - ns + 1):
            root_key = token_ids[i] if ns == 1 \
                else tuple(token_ids[i:i + ns])
            tree = self.mem.get(root_key, None)
            # Skip-gram fallback: try skip-variant root keys
            if tree is None and self.skip_gram and ns > 1:
                for skip_key in self._skip_root_variants(root_key):
                    tree = self.mem.get(skip_key, None)
                    if tree is not None:
                        break
            if tree is not None:
                ids = token_ids[i + ns:]
                decoding_ids, decoding_masks, sizes = \
                    tree.get_one_branch(
                        ids,
                        max_length=branch_length,
                        mode=mode,
                        idx=idx,
                        wild_budget=wild_budget,
                        skip_gram=self.skip_gram)
                s = len(decoding_ids)
                if s >= branch_length // 2:
                    break

        # Second pass: edit-distance fallback (only if exact match failed)
        if (decoding_ids is None or len(decoding_ids) <= 1) \
                and self.edit_dist > 0:
            for i in range(len(token_ids) - ns + 1):
                root_key = token_ids[i] if ns == 1 \
                    else tuple(token_ids[i:i + ns])
                tree = None
                for ed_key in self._edit_root_variants(root_key):
                    tree = self.mem.get(ed_key, None)
                    if tree is not None:
                        break
                if tree is None:
                    tree = self.mem.get(root_key, None)
                if tree is not None:
                    ids = token_ids[i + ns:]
                    decoding_ids, decoding_masks, sizes = \
                        tree.get_one_branch(
                            ids,
                            max_length=branch_length,
                            mode=mode,
                            idx=idx,
                            wild_budget=wild_budget,
                            skip_gram=self.skip_gram,
                            edit_dist=self.edit_dist,
                            id2str=self.id2str)
                    s = len(decoding_ids)
                    if s >= branch_length // 2:
                        break

        if decoding_ids is None:
            decoding_ids = token_ids[-1:]

        return decoding_ids, decoding_masks, sizes

    def get(self, token_ids: list[int], branch_length: int = 8,
            decoding_length: int = 64, min_output_size: int = 0,
            mode: str = 'output', idx: int = 0,
            wild_budget: int = 0) -> list[int]:
        """vLLM adapter: calls hier_get (faithful to original), then
        extracts the greedy best single branch from the multi-branch
        tree result.

        In _ravel, children are visited sorted by frequency descending
        (DFS).  The first consecutive chain from root where each node
        is an ancestor of the next IS the greedy best path.
        """
        decoding_ids, decoding_masks, sizes = self.hier_get(
            token_ids,
            decoding_length=decoding_length,
            branch_length=branch_length,
            min_output_size=min_output_size,
            mode=mode,
            idx=idx,
            wild_budget=wild_budget)

        if len(decoding_ids) <= 1:
            return []

        # Extract greedy best branch from multi-branch DFS result.
        # ids[0] is the matched/root token (context, not a draft).
        # Walk DFS order: mask[j, current]==1 means j is in current's
        # subtree, so the first such j is the best direct child.
        branch: list[int] = []
        current = 0
        n = len(decoding_ids)
        for j in range(1, n):
            if decoding_masks[j, current] == 1:
                branch.append(decoding_ids[j])
                current = j
            else:
                break

        return branch

    # ---- maintenance ----

    def reset_input_freqs(self, idx: int):
        if len(self._update_input_trees) > 0:
            for t in self._update_input_trees:
                t.reset_input_freq(idx)
            self._update_input_trees.clear()

    def squeeze_branch_counts(self):
        if len(self._update_trees) >= 1024:
            for t in self._update_trees:
                t.squeeze()
            self._update_trees.clear()


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
        # Model name (for tokenizer loading in edit-distance mode).
        self._model_name = vllm_config.model_config.model

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
        # Per-request local frequency table (request-specific patterns)
        self._req_local_freq: dict[int, dict[tuple, Counter]] = {}
        # Weight for local freq when merging with global freq
        self._local_freq_weight: float = float(os.environ.get(
            "VLLM_NGRAM_LOCAL_WEIGHT", "3.0"))
        # Minimum confidence threshold for drafting
        self._min_confidence: float = float(os.environ.get(
            "VLLM_NGRAM_MIN_CONFIDENCE", "0.3"))
        # Skip-gram frequency table (context with 1 gap -> Counter)
        self._skipgram_table: dict[tuple, Counter] = {}
        # Update counter for periodic persistence
        self._update_count: int = 0
        # Lock for async persistence
        self._persist_lock = threading.Lock()

        # Try to load persisted tables
        self._load_or_init_tables()

        # Register atexit for final persistence
        if self.ngram_table_path:
            atexit.register(self._persist_tables_sync)

        # ---- Trie mode initialization ----
        self.trie_table_path: Optional[str] = os.environ.get(
            "VLLM_TRIE_TABLE_PATH", None)
        self._trie_branch_length: int = int(os.environ.get(
            "VLLM_TRIE_BRANCH_LENGTH", "8"))
        self._trie_decoding_length: int = int(os.environ.get(
            "VLLM_TRIE_DECODING_LENGTH", "64"))
        self._trie_node_size: int = int(os.environ.get(
            "VLLM_TRIE_NODE_SIZE", "1"))
        self._trie_cache: Optional[TrieCache] = None
        self._trie_req_last_num_tokens: dict[int, int] = {}
        self._trie_persist_lock = threading.Lock()
        self._trie_fuzzy_budget: int = int(os.environ.get(
            _USE_TRIE_FUZZY_ENV, "0"))
        self._trie_skipgram: bool = os.environ.get(
            _USE_TRIE_SKIPGRAM_ENV, "0") == "1"
        self._trie_edit_dist: int = int(os.environ.get(
            _USE_TRIE_EDITDIST_ENV, "0"))
        self._trie_edit_dist_build: bool = os.environ.get(
            _USE_TRIE_EDITDIST_BUILD_ENV, "0") == "1"

        if os.environ.get(_USE_TRIE_ENV, "0") == "1":
            self._trie_init()
            # Register atexit for trie persistence
            if self.trie_table_path:
                atexit.register(self._trie_persist_sync)

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

    def _get_merged_counter(
        self,
        context: tuple,
        req_idx: int,
    ) -> Counter | None:
        """Get merged frequency counter: global + local (weighted).

        Returns None if context not found in either table.
        """
        global_cnt = self._freq_table.get(context)
        local_table = self._req_local_freq.get(req_idx)
        local_cnt = local_table.get(context) if local_table else None

        if global_cnt is None and local_cnt is None:
            return None
        if local_cnt is None:
            return global_cnt
        if global_cnt is None:
            return local_cnt

        # Merge: global + local * weight
        merged = Counter(global_cnt)
        w = self._local_freq_weight
        for tok, cnt in local_cnt.items():
            merged[tok] += int(cnt * w)
        return merged

    def _check_confidence(self, counter: Counter) -> tuple[int | None, float]:
        """Return (best_token, confidence) from a frequency counter.

        confidence = top_count / total_count.
        """
        if not counter:
            return None, 0.0
        best_token, top_count = counter.most_common(1)[0]
        total = sum(counter.values())
        confidence = top_count / total if total > 0 else 0.0
        return best_token, confidence

    def _vote_ngrams(
        self,
        input_ids: list[int],
        candidate: int,
        req_idx: int,
    ) -> bool:
        """Multi n-gram voting: check if multiple n-gram windows agree on
        the candidate token. Returns True if majority agrees."""
        n_max = min(self.max_n, len(input_ids))
        if n_max < self.min_n:
            return True  # Can't vote, trust the candidate

        votes_for = 0
        votes_total = 0
        for n in range(self.min_n, n_max + 1):
            context = tuple(input_ids[-n:])
            counter = self._get_merged_counter(context, req_idx)
            if counter:
                votes_total += 1
                best, _ = self._check_confidence(counter)
                if best == candidate:
                    votes_for += 1

        if votes_total == 0:
            return True  # No info, trust it
        return votes_for >= (votes_total + 1) // 2  # Majority

    def _propose_tokens_hash(
        self,
        input_ids: list[int],
        k: int,
        req_idx: int = 0,
    ) -> list[int]:
        """Generate k draft tokens with confidence filtering, multi n-gram
        voting, and request-local frequency overlay.

        Optimizations over plain chain lookup:
        1. Confidence filtering: stop when confidence < threshold
        2. Multi n-gram voting: only continue when majority of n-gram
           windows agree on the candidate
        3. Request-local freq overlay: merge global + local (weighted)
           for better position-specific predictions
        """
        drafts: list[int] = []
        n = min(self.max_n, len(input_ids))
        if n < self.min_n:
            return drafts

        # Build extended input for voting context
        extended = list(input_ids)

        for step in range(k):
            best_token = None
            best_confidence = 0.0

            # Try from longest to shortest n-gram
            for try_n in range(n, self.min_n - 1, -1):
                if try_n > len(extended):
                    continue
                context = tuple(extended[-try_n:])
                counter = self._get_merged_counter(context, req_idx)
                if counter:
                    token, conf = self._check_confidence(counter)
                    if token is not None and conf > best_confidence:
                        best_token = token
                        best_confidence = conf
                        if conf >= self._min_confidence:
                            break  # Good enough, use longest confident match

            if best_token is None:
                break  # No match at all

            # Confidence filter: stop if too uncertain
            if best_confidence < self._min_confidence:
                break

            # Multi n-gram voting: check consensus
            if not self._vote_ngrams(extended, best_token, req_idx):
                break  # Disagreement among n-gram windows

            drafts.append(best_token)
            extended.append(best_token)

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
        use_trie = os.environ.get(_USE_TRIE_ENV, "0") == "1"
        use_hash = os.environ.get(_USE_HASH_ENV, "1") == "1"
        use_skipgram = os.environ.get(_USE_SKIPGRAM_ENV, "0") == "1"

        if use_trie:
            return self._propose_trie_mode(
                sampled_token_ids, num_tokens_no_spec, token_ids_cpu)
        elif use_skipgram:
            return self._propose_skipgram_mode(
                sampled_token_ids, num_tokens_no_spec, token_ids_cpu)
        elif use_hash:
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

    def _update_local_freq_table(
        self,
        token_ids: list[int],
        n_new: int,
        req_idx: int,
    ) -> None:
        """Incremental update of per-request local frequency table."""
        if req_idx not in self._req_local_freq:
            self._req_local_freq[req_idx] = {}
        local = self._req_local_freq[req_idx]

        total = len(token_ids)
        for n in range(self.min_n, self.max_n + 1):
            start = max(0, total - n_new - n)
            for i in range(start, total - n):
                context = tuple(token_ids[i:i + n])
                next_token = token_ids[i + n]
                if context not in local:
                    local[context] = Counter()
                local[context][next_token] += 1

    # ------------------------------------------------------------------
    # Private: skip-gram hash table build & update & propose
    # ------------------------------------------------------------------

    def _build_skipgram_from_tokens(self, token_ids: list[int]) -> None:
        """Build skip-gram frequency table from full token sequence."""
        for n in range(self.min_n, self.max_n + 1):
            for i in range(len(token_ids) - n):
                context = token_ids[i:i + n]
                next_token = token_ids[i + n]
                for skip_pos in range(n):
                    pattern = list(context)
                    pattern[skip_pos] = _SKIPGRAM_SENTINEL
                    key = tuple(pattern)
                    if key not in self._skipgram_table:
                        self._skipgram_table[key] = Counter()
                    self._skipgram_table[key][next_token] += 1

    def _update_skipgram_table(self, token_ids: list[int],
                               n_new: int) -> None:
        """Incremental update of skip-gram table."""
        total = len(token_ids)
        for n in range(self.min_n, self.max_n + 1):
            start = max(0, total - n_new - n)
            for i in range(start, total - n):
                context = token_ids[i:i + n]
                next_token = token_ids[i + n]
                for skip_pos in range(n):
                    pattern = list(context)
                    pattern[skip_pos] = _SKIPGRAM_SENTINEL
                    key = tuple(pattern)
                    if key not in self._skipgram_table:
                        self._skipgram_table[key] = Counter()
                    self._skipgram_table[key][next_token] += 1

    def _propose_tokens_skipgram(
        self,
        input_ids: list[int],
        k: int,
        req_idx: int = 0,
    ) -> list[int]:
        """Draft tokens: exact hash lookup first, skip-gram fallback."""
        drafts: list[int] = []
        n = min(self.max_n, len(input_ids))
        if n < self.min_n:
            return drafts

        extended = list(input_ids)
        min_conf = self._min_confidence

        for step in range(k):
            best_token = None
            best_confidence = 0.0

            # 1. Exact match (same as hash mode)
            for try_n in range(n, self.min_n - 1, -1):
                if try_n > len(extended):
                    continue
                context = tuple(extended[-try_n:])
                counter = self._get_merged_counter(context, req_idx)
                if counter:
                    token, conf = self._check_confidence(counter)
                    if token is not None and conf > best_confidence:
                        best_token = token
                        best_confidence = conf
                        if conf >= min_conf:
                            break

            # 2. Skip-gram fallback (when exact fails or low confidence)
            if best_token is None or best_confidence < min_conf:
                for try_n in range(n, self.min_n - 1, -1):
                    if try_n > len(extended):
                        continue
                    context = list(extended[-try_n:])
                    for skip_pos in range(try_n):
                        pattern = list(context)
                        pattern[skip_pos] = _SKIPGRAM_SENTINEL
                        key = tuple(pattern)
                        counter = self._skipgram_table.get(key)
                        if counter:
                            token, conf = self._check_confidence(counter)
                            # Discount skip-gram confidence
                            conf *= 0.7
                            if token is not None and conf > best_confidence:
                                best_token = token
                                best_confidence = conf
                    if best_confidence >= min_conf:
                        break

            if best_token is None:
                break
            if best_confidence < min_conf * 0.5:
                break

            drafts.append(best_token)
            extended.append(best_token)

        return drafts

    def _propose_skipgram_mode(
        self,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
    ) -> list[list[int]]:
        """Skip-gram hash mode: exact hash + skip-gram fallback."""
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
                self._req_local_freq[i] = {}
                if not self._freq_table:
                    self._build_tables_from_tokens(tokens)
                else:
                    self._update_freq_table(tokens, num_tokens)
                self._update_local_freq_table(tokens, num_tokens, i)
                if not self._skipgram_table:
                    self._build_skipgram_from_tokens(tokens)
                else:
                    self._update_skipgram_table(tokens, num_tokens)
            else:
                n_new = num_tokens - prev_len
                if n_new > 0:
                    self._update_freq_table(tokens, n_new)
                    self._update_local_freq_table(tokens, n_new, i)
                    self._update_skipgram_table(tokens, n_new)

            self._req_last_num_tokens[i] = num_tokens

            k = min(self.k, self.max_model_len - num_tokens)
            if k <= 0:
                draft_token_ids.append([])
                continue

            drafts = self._propose_tokens_skipgram(tokens, k, req_idx=i)
            draft_token_ids.append(drafts)

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
                # New request: reset local freq table
                self._req_local_freq[i] = {}
                # Build/update global tables
                if not self._freq_table:
                    self._build_tables_from_tokens(tokens)
                else:
                    self._update_freq_table(tokens, num_tokens)
                # Build local freq from full prompt
                self._update_local_freq_table(tokens, num_tokens, i)
            else:
                # Continuation: incremental update with new tokens
                n_new = num_tokens - prev_len
                if n_new > 0:
                    self._update_freq_table(tokens, n_new)
                    self._update_local_freq_table(tokens, n_new, i)

            self._req_last_num_tokens[i] = num_tokens

            # Query draft tokens
            k = min(self.k, self.max_model_len - num_tokens)
            if k <= 0:
                draft_token_ids.append([])
                continue

            drafts = self._propose_tokens_hash(tokens, k, req_idx=i)
            draft_token_ids.append(drafts)

        return draft_token_ids

    # ------------------------------------------------------------------
    # Private: Trie mode — init / persist / propose
    # ------------------------------------------------------------------

    def _trie_init(self) -> None:
        """Initialize TrieCache, loading from disk if available."""
        ns = self._trie_node_size
        sg = self._trie_skipgram
        ed = self._trie_edit_dist
        # Build BK-tree + id2str if edit distance is enabled
        id2str, bk_tree = None, None
        if ed > 0:
            id2str, bk_tree = self._build_bk_tree()
        if self.trie_table_path and os.path.exists(self.trie_table_path):
            try:
                with open(self.trie_table_path, "rb") as f:
                    mem = pickle.load(f)
                self._trie_cache = TrieCache(
                    max_node=65536, max_output_node=512,
                    node_size=ns, skip_gram=sg,
                    edit_dist=ed, edit_dist_build=self._trie_edit_dist_build,
                    id2str=id2str, bk_tree=bk_tree)
                self._trie_cache.mem = mem
                logger.info(
                    "Loaded trieTable from %s: %d root entries",
                    self.trie_table_path, len(mem))
                return
            except Exception:
                logger.warning(
                    "Failed to load trieTable from %s, starting fresh",
                    self.trie_table_path, exc_info=True)
        self._trie_cache = TrieCache(
            max_node=65536, max_output_node=512,
            node_size=ns, skip_gram=sg,
            edit_dist=ed, edit_dist_build=self._trie_edit_dist_build,
                    id2str=id2str, bk_tree=bk_tree)
        if ns > 1:
            logger.info("Trie root_node_size=%d%s", ns,
                        " +skipgram" if sg else "")
        if ed > 0:
            logger.info("Trie edit_dist=%d (BK-tree with %d vocab entries)",
                        ed, len(id2str) if id2str else 0)

    def _trie_persist_sync(self) -> None:
        """Synchronously persist TrieCache.mem to disk."""
        if not self.trie_table_path or self._trie_cache is None:
            return
        try:
            with self._trie_persist_lock:
                with open(self.trie_table_path, "wb") as f:
                    pickle.dump(self._trie_cache.mem, f,
                                protocol=pickle.HIGHEST_PROTOCOL)
            logger.debug("Persisted trieTable to %s", self.trie_table_path)
        except Exception:
            logger.warning("Failed to persist trieTable", exc_info=True)

    def _build_bk_tree(self) -> tuple[dict, BKTree]:
        """Build BK-tree and id2str map from tokenizer vocabulary."""
        import time as _time
        from transformers import AutoTokenizer
        t0 = _time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, trust_remote_code=True)
        vocab = tokenizer.get_vocab()  # {str: int}
        id2str = {v: k for k, v in vocab.items()}
        bk = BKTree()
        for word, tid in vocab.items():
            bk.insert(word, tid)
        elapsed = _time.perf_counter() - t0
        logger.info("Built BK-tree from %d vocab entries in %.1fs",
                    len(vocab), elapsed)
        return id2str, bk

    def _trie_persist_async(self) -> None:
        """Asynchronously persist trie in a background thread."""
        if not self.trie_table_path or self._trie_cache is None:
            return
        t = threading.Thread(target=self._trie_persist_sync, daemon=True)
        t.start()

    def _propose_trie_mode(
        self,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
    ) -> list[list[int]]:
        """Trie-based mode: build/update trie and propose draft tokens."""
        if self._trie_cache is None:
            self._trie_init()
        assert self._trie_cache is not None

        draft_token_ids: list[list[int]] = []
        branch_length = self._trie_branch_length

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
            prev_len = self._trie_req_last_num_tokens.get(i)
            if prev_len is None or num_tokens < prev_len:
                # New request: write prompt tokens into trie as input
                self._trie_cache.put(tokens, branch_length=branch_length,
                                     mode='input', idx=i)
            else:
                # Continuation: stream new generated tokens into trie
                n_new = num_tokens - prev_len
                if n_new > 0:
                    new_tokens = tokens[prev_len:]
                    self._trie_cache.stream_put(
                        new_tokens, branch_length=branch_length,
                        final=False, idx=i)

            self._trie_req_last_num_tokens[i] = num_tokens

            # Query draft tokens
            k = min(self.k, self.max_model_len - num_tokens)
            if k <= 0:
                draft_token_ids.append([])
                continue

            # Original LookaheadCache callers pass only 2 tokens near
            # the decoding cursor (token before cursor + cursor token),
            # NOT the full sequence.  Match that convention.
            # With node_size > 1, we need more context: ns (root) + ns
            # (at least one internal match step).
            ns = self._trie_node_size
            ctx_len = max(2, ns + 2)
            context = tokens[-ctx_len:] if len(tokens) >= ctx_len else tokens
            bl = min(k, branch_length)
            dl = self._trie_decoding_length
            min_output_size = max(dl // 2, 1)
            drafts = self._trie_cache.get(
                context, branch_length=bl,
                decoding_length=dl,
                min_output_size=min_output_size,
                mode='mix', idx=i,
                wild_budget=self._trie_fuzzy_budget)
            # Truncate to at most k drafts
            draft_token_ids.append(drafts[:k])

        return draft_token_ids

    def _trie_finalize_request(self, idx: int) -> None:
        """Called when a request finishes: flush remaining output buffer,
        reset input freqs, and trigger pruning."""
        if self._trie_cache is None:
            return
        # Flush remaining output tokens with final=True
        self._trie_cache.stream_put(
            [], branch_length=self._trie_branch_length,
            final=True, idx=idx)
        # Clean up tracking
        self._trie_req_last_num_tokens.pop(idx, None)
        # Async persist
        self._trie_persist_async()

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
