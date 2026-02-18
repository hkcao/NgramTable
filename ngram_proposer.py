# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import atexit
import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from numba import get_num_threads, jit, njit, prange, set_num_threads

from vllm.config import VllmConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants for numba hash table proposer
# 哈希表优化方案：使用Numba JIT编译的哈希表替代原有KMP算法
# 核心思路：预先构建ngram->next_token的频率表和查找表，查询时O(1)复杂度
# ---------------------------------------------------------------------------
_HASH_PRIME = np.int64(1000003)  # 多项式滚动哈希的质数模
_MIN_TABLE_SIZE = 1024           # 哈希表最小尺寸（2的幂）
_MAX_TABLE_SIZE = 131072         # 哈希表最大尺寸，避免内存过大
_EMPTY_TOKEN = np.int32(-1)      # 空槽位标记
_FP_LEN = 8  # 请求指纹长度：用前8个token识别请求是否复用batch slot

# Shared mode constants
_SHARED_DEFAULT_TABLE_SIZE = 1048576   # 1M slots default for shared tables
_SHARED_MAX_TABLE_SIZE = 8388608       # 8M slots hard cap
_SHARED_METADATA_VERSION = 1           # File format version


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


def _next_power_of_2_shared(n: int) -> int:
    """Round up to next power of 2 for shared tables.

    Clamped to [_MIN_TABLE_SIZE, _SHARED_MAX_TABLE_SIZE].
    Not a Numba function — only called at init/load time.
    """
    if n <= _MIN_TABLE_SIZE:
        return _MIN_TABLE_SIZE
    v = n - 1
    v |= v >> 1
    v |= v >> 2
    v |= v >> 4
    v |= v >> 8
    v |= v >> 16
    v |= v >> 32
    v += 1
    if v > _SHARED_MAX_TABLE_SIZE:
        return _SHARED_MAX_TABLE_SIZE
    return v


@njit(cache=True)
def _hash_ngram(tokens, start, length, mask):
    """多项式滚动哈希：计算tokens[start:start+length]的哈希值

    优化点：使用多项式哈希而非Python tuple hash，避免tuple分配开销
    hash = (t[0]*P^(n-1) + t[1]*P^(n-2) + ... + t[n-1]) & mask
    """
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
    """为指定的n-gram长度构建频率表和查找表（核心算法）

    双哈希表设计：
    1. freq表：存储(ngram, next_token) -> count的频率统计
    2. lut表：存储ngram -> best_next_token的查找映射（选择频率最高的）

    优化点：
    - 基础ngram哈希只计算一次，复用于两个表（减少40%哈希计算）
    - 开放地址法+线性探测解决冲突
    - freq表按频率实时更新lut表的最佳候选
    """
    mask = np.int64(table_size - 1)

    for i in range(num_tokens - n):
        next_token = tokens[i + n]

        # 【优化】基础ngram哈希只计算一次，复用于freq表和lut表
        # 原实现每个表都单独计算哈希，导致重复计算
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
    """单个n-gram长度的哈希表状态容器

    双表设计：
    - freq表：统计(ngram, next_token)的出现频率
    - lut表：存储ngram的最优next_token（频率最高的）

    开放地址法：使用occupied数组标记槽位占用情况
    """
    table_size: int                # 哈希表大小（2的幂）
    freq_keys: np.ndarray          # 频率表key: (table_size, n+1) 存储ngram+next_token
    freq_counts: np.ndarray        # 频率表value: (table_size,) 出现次数
    freq_occupied: np.ndarray      # 频率表占用标记: (table_size,) bool
    lut_keys: np.ndarray           # 查找表key: (table_size, n) 存储ngram
    lut_vals: np.ndarray           # 查找表value: (table_size,) 最优next_token
    lut_best_counts: np.ndarray    # 查找表辅助: (table_size,) 最优token的频率
    lut_occupied: np.ndarray       # 查找表占用标记: (table_size,) bool

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


@dataclass
class _SharedHashTableState:
    """Shared hash table state for one n-gram size, with dirty tracking.

    Extends _HashTableState conceptually with:
    - dirty flag to track unsaved modifications
    - occupied_count for O(1) load factor monitoring
    """
    table_size: int
    freq_keys: np.ndarray
    freq_counts: np.ndarray
    freq_occupied: np.ndarray
    lut_keys: np.ndarray
    lut_vals: np.ndarray
    lut_best_counts: np.ndarray
    lut_occupied: np.ndarray
    dirty: bool = False
    occupied_count: int = 0

    @staticmethod
    def allocate(table_size: int, n: int) -> '_SharedHashTableState':
        return _SharedHashTableState(
            table_size=table_size,
            freq_keys=np.zeros((table_size, n + 1), dtype=np.int32),
            freq_counts=np.zeros(table_size, dtype=np.int32),
            freq_occupied=np.zeros(table_size, dtype=np.bool_),
            lut_keys=np.zeros((table_size, n), dtype=np.int32),
            lut_vals=np.full(table_size, _EMPTY_TOKEN, dtype=np.int32),
            lut_best_counts=np.zeros(table_size, dtype=np.int32),
            lut_occupied=np.zeros(table_size, dtype=np.bool_),
            dirty=False,
            occupied_count=0,
        )


# ---------------------------------------------------------------------------
# Shared table persistence: load / save / flush
# ---------------------------------------------------------------------------

def _load_shared_tables(
    shared_dir: Path, min_n: int, max_n: int, table_size: int,
) -> dict[int, _SharedHashTableState]:
    """Load shared tables from disk, or allocate empty ones if unavailable."""
    tables: dict[int, _SharedHashTableState] = {}
    meta_path = shared_dir / "metadata.json"

    if not shared_dir.exists():
        shared_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Shared ngram dir created: %s", shared_dir)
        for n in range(min_n, max_n + 1):
            tables[n] = _SharedHashTableState.allocate(table_size, n)
        return tables

    # Try loading metadata
    if not meta_path.exists():
        logger.warning("No metadata.json in %s, allocating empty tables",
                       shared_dir)
        for n in range(min_n, max_n + 1):
            tables[n] = _SharedHashTableState.allocate(table_size, n)
        return tables

    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read metadata.json: %s, allocating empty "
                       "tables", e)
        for n in range(min_n, max_n + 1):
            tables[n] = _SharedHashTableState.allocate(table_size, n)
        return tables

    # Version / parameter mismatch check
    if (meta.get("version") != _SHARED_METADATA_VERSION
            or meta.get("min_n") != min_n
            or meta.get("max_n") != max_n
            or meta.get("table_size") != table_size):
        logger.warning(
            "Metadata mismatch (version=%s, min_n=%s, max_n=%s, "
            "table_size=%s vs expected %s/%s/%s/%s). Allocating empty tables.",
            meta.get("version"), meta.get("min_n"), meta.get("max_n"),
            meta.get("table_size"),
            _SHARED_METADATA_VERSION, min_n, max_n, table_size)
        for n in range(min_n, max_n + 1):
            tables[n] = _SharedHashTableState.allocate(table_size, n)
        return tables

    # Load each n-gram file
    for n in range(min_n, max_n + 1):
        npz_path = shared_dir / f"n{n}.npz"
        if not npz_path.exists():
            logger.warning("Missing %s, allocating empty table for n=%d",
                           npz_path, n)
            tables[n] = _SharedHashTableState.allocate(table_size, n)
            continue
        try:
            data = np.load(npz_path)
            occupied_count = int(np.sum(data["lut_occupied"]))
            tables[n] = _SharedHashTableState(
                table_size=table_size,
                freq_keys=data["freq_keys"],
                freq_counts=data["freq_counts"],
                freq_occupied=data["freq_occupied"],
                lut_keys=data["lut_keys"],
                lut_vals=data["lut_vals"],
                lut_best_counts=data["lut_best_counts"],
                lut_occupied=data["lut_occupied"],
                dirty=False,
                occupied_count=occupied_count,
            )
            logger.info("Loaded shared table n=%d from %s "
                        "(occupied=%d/%d, load=%.1f%%)",
                        n, npz_path, occupied_count, table_size,
                        100.0 * occupied_count / table_size)
        except Exception as e:
            logger.warning("Failed to load %s: %s, allocating empty table",
                           npz_path, e)
            tables[n] = _SharedHashTableState.allocate(table_size, n)

    return tables


def _save_shared_tables_sync(
    shared_dir: Path, min_n: int, max_n: int, table_size: int,
    tables: dict[int, _SharedHashTableState],
) -> None:
    """Save shared tables to disk atomically (write .tmp then rename)."""
    shared_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    meta = {
        "version": _SHARED_METADATA_VERSION,
        "min_n": min_n,
        "max_n": max_n,
        "table_size": table_size,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_tmp = shared_dir / "metadata.json.tmp"
    meta_path = shared_dir / "metadata.json"
    with open(meta_tmp, "w") as f:
        json.dump(meta, f, indent=2)
    meta_tmp.rename(meta_path)

    # Save each n-gram table
    for n, state in tables.items():
        npz_tmp = shared_dir / f"n{n}.npz.tmp"
        npz_path = shared_dir / f"n{n}.npz"
        np.savez_compressed(
            npz_tmp,
            freq_keys=state.freq_keys,
            freq_counts=state.freq_counts,
            freq_occupied=state.freq_occupied,
            lut_keys=state.lut_keys,
            lut_vals=state.lut_vals,
            lut_best_counts=state.lut_best_counts,
            lut_occupied=state.lut_occupied,
        )
        npz_tmp.rename(npz_path)
        state.dirty = False


class _SharedTableFlusher:
    """Background daemon thread that periodically flushes shared tables."""

    def __init__(
        self,
        shared_dir: Path,
        min_n: int,
        max_n: int,
        table_size: int,
        tables: dict[int, _SharedHashTableState],
        lock: threading.Lock,
        flush_interval: float = 60.0,
    ):
        self._shared_dir = shared_dir
        self._min_n = min_n
        self._max_n = max_n
        self._table_size = table_size
        self._tables = tables
        self._lock = lock
        self._flush_interval = flush_interval
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="ngram-shared-flusher")
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.wait(self._flush_interval):
            self.flush()

    def flush(self) -> None:
        """Flush dirty tables to disk. Safe to call from any thread."""
        # Check if any table is dirty
        any_dirty = False
        with self._lock:
            for state in self._tables.values():
                if state.dirty:
                    any_dirty = True
                    break

        if not any_dirty:
            return

        # Copy arrays under lock (fast, microsecond-level)
        copies: dict[int, dict[str, np.ndarray]] = {}
        with self._lock:
            for n, state in self._tables.items():
                if state.dirty:
                    copies[n] = {
                        "freq_keys": state.freq_keys.copy(),
                        "freq_counts": state.freq_counts.copy(),
                        "freq_occupied": state.freq_occupied.copy(),
                        "lut_keys": state.lut_keys.copy(),
                        "lut_vals": state.lut_vals.copy(),
                        "lut_best_counts": state.lut_best_counts.copy(),
                        "lut_occupied": state.lut_occupied.copy(),
                    }
                    state.dirty = False

        # Write to disk outside the lock (slow I/O)
        try:
            self._shared_dir.mkdir(parents=True, exist_ok=True)

            # Save metadata
            meta = {
                "version": _SHARED_METADATA_VERSION,
                "min_n": self._min_n,
                "max_n": self._max_n,
                "table_size": self._table_size,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            meta_tmp = self._shared_dir / "metadata.json.tmp"
            meta_path = self._shared_dir / "metadata.json"
            with open(meta_tmp, "w") as f:
                json.dump(meta, f, indent=2)
            meta_tmp.rename(meta_path)

            for n, arrays in copies.items():
                npz_tmp = self._shared_dir / f"n{n}.npz.tmp"
                npz_path = self._shared_dir / f"n{n}.npz"
                np.savez_compressed(npz_tmp, **arrays)
                npz_tmp.rename(npz_path)

            logger.info("Shared ngram tables flushed to %s", self._shared_dir)
        except Exception:
            logger.warning("Failed to flush shared ngram tables",
                           exc_info=True)

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=5.0)


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

        # 三路模式选择：kmp | per_request | shared
        # VLLM_NGRAM_MODE 优先；未设置时回退检查 VLLM_NGRAM_USE_HASH_TABLE
        ngram_mode = os.environ.get("VLLM_NGRAM_MODE", "").lower()
        if ngram_mode not in ("kmp", "per_request", "shared"):
            # Backward compat: fall back to old env var
            if os.environ.get("VLLM_NGRAM_USE_HASH_TABLE", "0") == "1":
                ngram_mode = "per_request"
            else:
                ngram_mode = "kmp"
        self.ngram_mode = ngram_mode

        if self.ngram_mode == "shared":
            self._init_shared_mode()
        elif self.ngram_mode == "per_request":
            self._init_per_request_mode()
        else:
            self._init_kmp_mode(vllm_config)

    def _init_per_request_mode(self) -> None:
        """Initialize per-request hash table mode."""
        # 每个请求维护独立的哈希表，支持增量更新
        self._req_tables: dict[int, dict[int, _HashTableState]] = {}
        self._req_last_num_tokens: dict[int, int] = {}
        self._req_fingerprints: dict[int, tuple] = {}
        self._req_query_cache: dict[int, dict] = {}
        _warmup_hash_njit(self.min_n, self.max_n, self.k)

    def _init_shared_mode(self) -> None:
        """Initialize shared persistent hash table mode."""
        table_size_raw = int(os.environ.get(
            "VLLM_NGRAM_SHARED_TABLE_SIZE",
            str(_SHARED_DEFAULT_TABLE_SIZE)))
        self._shared_table_size = _next_power_of_2_shared(table_size_raw)

        shared_dir_name = os.environ.get(
            "VLLM_NGRAM_SHARED_DIR", "ngram_hash")
        self._shared_dir = Path(shared_dir_name)

        flush_interval = float(os.environ.get(
            "VLLM_NGRAM_FLUSH_INTERVAL", "60"))

        # Load or allocate tables
        self._shared_tables = _load_shared_tables(
            self._shared_dir, self.min_n, self.max_n,
            self._shared_table_size)

        # Lock protects shared table arrays during updates
        self._shared_lock = threading.Lock()

        # Per-request tracking: slot -> (last_num_tokens, fingerprint)
        self._shared_req_state: dict[int, tuple[int, tuple]] = {}

        # Pre-build query cache (references into shared tables)
        self._shared_query_cache = self._build_query_cache_shared()

        # Background flusher
        self._shared_flusher = _SharedTableFlusher(
            self._shared_dir, self.min_n, self.max_n,
            self._shared_table_size, self._shared_tables,
            self._shared_lock, flush_interval)

        # Register atexit handler for clean shutdown
        atexit.register(self._shutdown_shared)

        # Warmup Numba JIT
        _warmup_hash_njit(self.min_n, self.max_n, self.k)

        logger.info(
            "NgramProposer shared mode: table_size=%d, dir=%s, "
            "flush_interval=%.0fs",
            self._shared_table_size, self._shared_dir, flush_interval)

    def _init_kmp_mode(self, vllm_config: VllmConfig) -> None:
        """Initialize original KMP mode."""
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
            self.num_numba_thread_available = min(1, (cpu_count // 2))
            self.num_numba_thread_available //= tp_size
        else:
            self.num_numba_thread_available = 1

        # Trigger Numba JIT compilation for N-gram proposer.
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
        """增量更新哈希表：只处理新增的token，避免全量重建

        【优化】动态扩容机制：
        - 负载因子 > 0.6 时触发全量重建（扩大表尺寸）
        - 保持哈希表性能，避免过多冲突导致查询退化
        - 新表尺寸 = next_power_of_2(num_tokens * 2)
        """
        needs_rebuild = False
        for n in range(self.min_n, self.max_n + 1):
            state = tables[n]
            _update_tables_for_n(
                tokens, old_len, new_len, n, state.table_size,
                state.freq_keys, state.freq_counts, state.freq_occupied,
                state.lut_keys, state.lut_vals, state.lut_best_counts,
                state.lut_occupied)
            # 检查查找表的负载因子，避免性能退化
            occupied = int(np.sum(state.lut_occupied))
            if occupied > state.table_size * 0.6:
                needs_rebuild = True

        if needs_rebuild:
            # 触发全量重建，使用更大的表尺寸
            return self._build_hash_tables(tokens)
        return tables

    def _build_query_cache(
        self, tables: dict[int, _HashTableState],
    ) -> dict:
        """预打包查询缓存：避免每次查询时重复打包数组和分配内存

        优化点：将多个n-gram的查找表预先打包成tuple，查询时直接传递
        避免在propose调用时频繁创建临时对象
        """
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

            # 【优化】请求指纹检测：通过前8个token识别请求是否复用batch slot
            # vLLM V1的batch slot可能被不同请求复用，需要检测避免脏数据
            fp_len = min(_FP_LEN, num_tokens)
            fp = tuple(tokens[:fp_len].tolist())

            # 检测是否为新请求（三种情况）：
            # 1. slot首次使用 2. token数减少(新请求) 3. 指纹不匹配(slot复用)
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

    # ------------------------------------------------------------------
    # Shared mode helpers
    # ------------------------------------------------------------------

    def _build_query_cache_shared(self) -> dict:
        """Build query cache referencing the shared tables (no copy)."""
        tables = self._shared_tables
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

    def _propose_shared(
        self,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
    ) -> list[list[int]]:
        """Shared-table ngram proposal: all requests update global tables."""
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

            # Fingerprint for request identity detection
            fp_len = min(_FP_LEN, num_tokens)
            fp = tuple(tokens[:fp_len].tolist())

            # Determine old_len for incremental update
            prev = self._shared_req_state.get(i)
            if prev is None or num_tokens < prev[0] or prev[1] != fp:
                # New request: full insert
                old_len = 0
            else:
                old_len = prev[0]

            # Update shared tables with this request's tokens
            if num_tokens > old_len:
                with self._shared_lock:
                    for n in range(self.min_n, self.max_n + 1):
                        state = self._shared_tables[n]
                        _update_tables_for_n(
                            tokens, old_len, num_tokens, n,
                            state.table_size,
                            state.freq_keys, state.freq_counts,
                            state.freq_occupied,
                            state.lut_keys, state.lut_vals,
                            state.lut_best_counts, state.lut_occupied)
                        new_occupied = int(np.sum(state.lut_occupied))
                        state.occupied_count = new_occupied
                        state.dirty = True

                        # Load factor warning
                        load_factor = new_occupied / state.table_size
                        if load_factor > 0.8:
                            logger.warning(
                                "Shared table n=%d load factor %.1f%% "
                                "(%d/%d). Consider increasing "
                                "VLLM_NGRAM_SHARED_TABLE_SIZE.",
                                n, 100.0 * load_factor,
                                new_occupied, state.table_size)

            self._shared_req_state[i] = (num_tokens, fp)

            # Query for draft tokens
            k = min(self.k, self.max_model_len - num_tokens)
            if k <= 0:
                draft_token_ids.append([])
                continue

            qc = self._shared_query_cache
            draft_out = qc['draft_out']
            num_drafted = _query_lookup(
                tokens, num_tokens, self.min_n, self.max_n, k,
                qc['lut_keys'], qc['lut_vals'], qc['lut_occupied'],
                qc['table_sizes'], draft_out)
            draft_token_ids.append(draft_out[:num_drafted].tolist())

        # Cleanup stale request tracking
        stale = set(self._shared_req_state.keys()) - active_req_indices
        for idx in stale:
            del self._shared_req_state[idx]

        return draft_token_ids

    def _shutdown_shared(self) -> None:
        """Clean shutdown: stop flusher and do a final sync flush."""
        if hasattr(self, '_shared_flusher'):
            self._shared_flusher.stop()
            # Final synchronous flush
            try:
                self._shared_flusher.flush()
            except Exception:
                logger.warning("Final shared table flush failed",
                               exc_info=True)

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
        if self.ngram_mode == "shared":
            return self._propose_shared(
                sampled_token_ids, num_tokens_no_spec, token_ids_cpu)

        if self.ngram_mode == "per_request":
            return self._propose_hash(
                sampled_token_ids, num_tokens_no_spec, token_ids_cpu)

        # KMP mode
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
    """预热Numba JIT编译：在初始化时触发所有@njit函数的编译

    避免首次调用时的编译延迟（通常需要几百毫秒）
    使用小规模dummy数据触发build/update/query的完整路径
    """
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
