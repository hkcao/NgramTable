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

# Persistence flush threshold: flush ngramTable every N update steps
_FLUSH_EVERY = 100


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
        self.nodes: dict[int, TrieNode] = {}

    # ---- write path ----

    def put(self, token_ids: list[int], mode: str = 'output', idx: int = 0,
            freq: float = 1.0):
        assert mode in ('input', 'output')
        if mode == 'output':
            idx = -1
        self._put(token_ids, self.nodes, mode=mode, idx=idx, freq=freq)

    def _put(self, token_ids: list[int], nodes: dict, mode: str = 'output',
             freq: float = 1.0, idx: int = -1):
        while True:
            if len(token_ids) == 0:
                break
            t = token_ids[0]
            node = nodes.get(t, None)
            if node is None:
                n = self._pack(token_ids, idx, freq=freq)
                nodes.update(n)
                self.n_node += len(token_ids)
                if mode == 'output':
                    self.n_output_node += len(token_ids)
                break
            node.freqs[idx] = node.freqs.get(idx, 0.0) + freq
            nodes = node.children
            token_ids = token_ids[1:]

    def _pack(self, token_ids: list[int], idx: int,
              freq: float = 1.0) -> dict:
        ps: dict = {}
        for token in token_ids[::-1]:
            freqs = {idx: freq}
            p = TrieNode(ps, freqs)
            ps = {token: p}
        return ps

    # ---- read path ----

    def _match(self, token_ids: list[int], mode: str = 'mix',
               idx: int = 0) -> tuple:
        nodes = self.nodes
        token_id = None
        if len(token_ids) == 0:
            return token_id, nodes

        for token_id in token_ids:
            node = nodes.get(token_id, None)
            nodes = {}
            if node is None:
                break
            if mode == 'input':
                if node.freqs.get(idx, 0.0) > 0:
                    nodes = node.children
            elif mode == 'output':
                if node.freqs.get(-1, 0.0) > 0:
                    nodes = node.children
            else:
                if node.freqs.get(idx, 0.0) > 0 or node.freqs.get(-1, 0.0) > 0:
                    nodes = node.children

        return token_id, nodes

    def get(self, token_ids: list[int], max_size: int = 64,
            max_length: int = 8, min_input_size: int = 0,
            min_output_size: int = 0, output_weight: float = 1e-4,
            mode: str = 'mix', idx: int = 0):
        """Multi-branch query — faithful reproduction of Tree.get."""
        assert mode in ('input', 'output', 'mix')

        match_token_id, nodes = self._match(token_ids, mode=mode, idx=idx)
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
               min_mix_freq: float = 1.0, output_weight: float = 1e-4,
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
                       mode: str = 'mix', idx: int = 0):
        """Single-branch query — faithful reproduction of
        Tree.get_one_branch."""
        assert mode in ('input', 'output', 'mix')

        match_token_id, nodes = self._match(token_ids, mode=mode, idx=idx)
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
                        freq = 10000 * fi + fo
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
            ids.append(max_id)
            nodes = max_node.children
            length += 1

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

    ``mem`` maps each root token_id to a TrieTree that stores all
    successor sequences starting with that token.
    """

    def __init__(self, max_node: int = 65536, max_output_node: int = 512):
        self.max_node = max_node
        self.max_output_node = max_output_node
        self.mem: dict[int, TrieTree] = {}
        self._output_ids: defaultdict[int, list] = defaultdict(list)
        self._update_trees: set[TrieTree] = set()
        self._update_input_trees: set[TrieTree] = set()

    # ---- bulk write (prompt / input) ----

    def put(self, token_ids: list[int], branch_length: int = 8,
            mode: str = 'output', idx: int = 0):
        if len(token_ids) < 2:
            return
        ts = len(token_ids)
        for i in range(ts - 1):
            token_id = token_ids[i]
            tup = token_ids[i + 1:i + branch_length + 1]
            tree = self.mem.get(token_id, None)
            if tree is not None:
                tree.put(tup, mode=mode, idx=idx)
                self._update_trees.add(tree)
            else:
                tree = TrieTree(token_id, max_node=self.max_node,
                                max_output_node=self.max_output_node)
                tree.put(tup, mode=mode, idx=idx)
                self.mem[token_id] = tree
            if mode == 'input':
                self._update_input_trees.add(tree)

    # ---- streaming write (generated output) ----

    def stream_put(self, token_ids: list[int], branch_length: int = 8,
                   final: bool = False, idx: int = 0):
        self._output_ids[idx].extend(token_ids)
        output_ids = self._output_ids[idx]
        ts = len(output_ids)
        min_branch_length = 1 if final else branch_length
        if ts > min_branch_length:
            for i in range(ts - min_branch_length):
                token_id = output_ids[i]
                tup = output_ids[i + 1:i + branch_length + 1]
                tree = self.mem.get(token_id, None)
                if tree is not None:
                    tree.put(tup, mode='output', idx=idx)
                else:
                    tree = TrieTree(token_id, max_node=self.max_node,
                                    max_output_node=self.max_output_node)
                    tree.put(tup, mode='output', idx=idx)
                    self.mem[token_id] = tree
                self._update_trees.add(tree)
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
                 idx: int = 0):
        """Multi-branch sliding-window query — faithful reproduction of
        LookaheadCache.hier_get."""
        default_mask = np.ones((1, 1), dtype=np.int64)

        if decoding_length <= 1 or branch_length == 0:
            return token_ids[-1:], default_mask, []

        decoding_ids = None
        decoding_masks = default_mask
        sizes = [0, 0]
        for i, t in enumerate(token_ids):
            tree = self.mem.get(t, None)
            if tree is not None:
                ids = token_ids[i + 1:]
                decoding_ids, decoding_masks, sizes = tree.get(
                    ids,
                    max_size=decoding_length,
                    max_length=branch_length,
                    min_input_size=min_input_size,
                    min_output_size=min_output_size,
                    mode=mode,
                    idx=idx)
                s = len(decoding_ids)
                if s >= branch_length:
                    break

        if decoding_ids is None:
            decoding_ids = token_ids[-1:]

        return decoding_ids, decoding_masks, sizes

    def one_get(self, token_ids: list[int], decoding_length: int = 64,
                branch_length: int = 8, min_input_size: int = 0,
                min_output_size: int = 0, mode: str = 'mix',
                idx: int = 0):
        """Single-branch sliding-window query — faithful reproduction of
        LookaheadCache.one_get."""
        default_mask = np.ones((1, 1), dtype=np.int64)

        if decoding_length <= 1 or branch_length == 0:
            return token_ids[-1:], default_mask, []

        decoding_ids = None
        decoding_masks = default_mask
        sizes = [0, 0]
        for i, t in enumerate(token_ids):
            tree = self.mem.get(t, None)
            if tree is not None:
                ids = token_ids[i + 1:]
                decoding_ids, decoding_masks, sizes = \
                    tree.get_one_branch(
                        ids,
                        max_length=branch_length,
                        mode=mode,
                        idx=idx)
                s = len(decoding_ids)
                if s >= branch_length // 2:
                    break

        if decoding_ids is None:
            decoding_ids = token_ids[-1:]

        return decoding_ids, decoding_masks, sizes

    def get(self, token_ids: list[int], branch_length: int = 8,
            decoding_length: int = 64, min_output_size: int = 0,
            mode: str = 'output', idx: int = 0) -> list[int]:
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
            idx=idx)

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

        # ---- Trie mode initialization ----
        self.trie_table_path: Optional[str] = os.environ.get(
            "VLLM_TRIE_TABLE_PATH", None)
        self._trie_branch_length: int = int(os.environ.get(
            "VLLM_TRIE_BRANCH_LENGTH", "8"))
        self._trie_decoding_length: int = int(os.environ.get(
            "VLLM_TRIE_DECODING_LENGTH", "64"))
        self._trie_cache: Optional[TrieCache] = None
        self._trie_req_last_num_tokens: dict[int, int] = {}
        self._trie_persist_lock = threading.Lock()

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
        use_trie = os.environ.get(_USE_TRIE_ENV, "0") == "1"
        use_hash = os.environ.get(_USE_HASH_ENV, "1") == "1"

        if use_trie:
            return self._propose_trie_mode(
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

    # ------------------------------------------------------------------
    # Private: Trie mode — init / persist / propose
    # ------------------------------------------------------------------

    def _trie_init(self) -> None:
        """Initialize TrieCache, loading from disk if available."""
        if self.trie_table_path and os.path.exists(self.trie_table_path):
            try:
                with open(self.trie_table_path, "rb") as f:
                    mem = pickle.load(f)
                self._trie_cache = TrieCache(
                    max_node=65536, max_output_node=512)
                self._trie_cache.mem = mem
                logger.info(
                    "Loaded trieTable from %s: %d root entries",
                    self.trie_table_path, len(mem))
                return
            except Exception:
                logger.warning(
                    "Failed to load trieTable from %s, starting fresh",
                    self.trie_table_path, exc_info=True)
        self._trie_cache = TrieCache(max_node=65536, max_output_node=512)

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
            context = tokens[-2:] if len(tokens) >= 2 else tokens
            bl = min(k, branch_length)
            dl = self._trie_decoding_length
            min_output_size = max(dl // 2, 1)
            drafts = self._trie_cache.get(
                context, branch_length=bl,
                decoding_length=dl,
                min_output_size=min_output_size,
                mode='output', idx=i)
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
