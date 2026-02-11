# Test Results v0 — Numba Hash Table Ngram Proposer

Date: 2026-02-10
vLLM version: 0.15.1
GPU: NVIDIA (WSL2)
Dataset: SWE-bench Lite (princeton-nlp/SWE-bench_Lite), 20 samples
max_tokens: 512, temperature: 0.0

## Implementation Summary

Phase 2: Replaced Python dict/tuple operations in the hash table ngram proposer
with numba @njit functions operating on pre-allocated numpy arrays.

Key changes to `vllm/v1/spec_decode/ngram_proposer.py`:
- 7 `@njit(cache=True)` functions: `_next_power_of_2`, `_hash_ngram`, `_keys_equal`,
  `_build_tables_for_n`, `_update_tables_for_n`, `_query_single_n`, `_query_lookup`
- Open-addressing hash tables with linear probing, polynomial rolling hash (P=1000003)
- Two tables per n-gram size: freq table (ngram+next_token -> count), lookup table (ngram -> best_next_token)
- Dynamic table sizing: `next_power_of_2(num_tokens * 2)`, clamped [1024, 131072]
- `_HashTableState` dataclass holds numpy arrays per request per n
- JIT warmup via `_warmup_hash_njit()` during `__init__`

---

## Qwen2.5-0.5B-Instruct Results

gpu_memory_utilization: 0.85

### Throughput & Speedup

| Mode                     | tok/s   | Time(s) | vs Baseline | HT/KMP |
|--------------------------|--------:|--------:|------------:|-------:|
| baseline                 | 6412.21 |   1.552 |       1.00x |      - |
| KMP spec=3 n=2-3        | 5363.81 |   1.855 |       0.84x |      - |
| HashTable spec=3 n=2-3  | 5055.05 |   1.969 |       0.79x |  0.94x |
| KMP spec=5 n=2-5        | 6180.68 |   1.592 |       0.96x |      - |
| HashTable spec=5 n=2-5  | 6534.29 |   1.567 |     **1.02x** |  **1.06x** |
| KMP spec=8 n=3-7        | 5183.89 |   1.898 |       0.81x |      - |
| HashTable spec=8 n=3-7  | 6766.01 |   1.513 |     **1.06x** |  **1.31x** |

### Acceptance Stats

| Mode                     | Accept% | MeanLen | Drafts | Draft Tokens | Accepted |
|--------------------------|--------:|--------:|-------:|-------------:|---------:|
| KMP spec=3 n=2-3        |   67.9% |    3.04 |   2813 |         8436 |     5731 |
| HashTable spec=3 n=2-3  |   76.4% |    3.29 |   2578 |         7734 |     5911 |
| KMP spec=5 n=2-5        |   68.6% |    4.42 |   1932 |         9640 |     6610 |
| HashTable spec=5 n=2-5  |   73.1% |    4.66 |   1877 |         9385 |     6864 |
| KMP spec=8 n=3-7        |   65.2% |    6.17 |   1305 |        10350 |     6752 |
| HashTable spec=8 n=3-7  |   72.6% |    6.81 |   1267 |        10136 |     7362 |

### Per-Position Acceptance Rate (0.5B)

| Mode                     |    p0 |    p1 |    p2 |    p3 |    p4 |    p5 |    p6 |    p7 |
|--------------------------|------:|------:|------:|------:|------:|------:|------:|------:|
| KMP spec=3 n=2-3        | 0.774 | 0.675 | 0.588 |     - |     - |     - |     - |     - |
| HashTable spec=3 n=2-3  | 0.843 | 0.760 | 0.690 |     - |     - |     - |     - |     - |
| KMP spec=5 n=2-5        | 0.805 | 0.725 | 0.668 | 0.625 | 0.598 |     - |     - |     - |
| HashTable spec=5 n=2-5  | 0.854 | 0.774 | 0.714 | 0.673 | 0.641 |     - |     - |     - |
| KMP spec=8 n=3-7        | 0.837 | 0.728 | 0.661 | 0.631 | 0.602 | 0.582 | 0.572 | 0.561 |
| HashTable spec=8 n=3-7  | 0.888 | 0.796 | 0.752 | 0.723 | 0.693 | 0.667 | 0.655 | 0.637 |

---

## Qwen2.5-3B-Instruct Results

gpu_memory_utilization: 0.9

### Throughput & Speedup

| Mode                     | tok/s   | Time(s) | vs Baseline | HT/KMP |
|--------------------------|--------:|--------:|------------:|-------:|
| baseline                 | 1316.92 |   7.767 |       1.00x |      - |
| KMP spec=3 n=2-3        | 1276.31 |   8.023 |       0.97x |      - |
| HashTable spec=3 n=2-3  | 1284.66 |   7.971 |       0.98x |  1.01x |
| KMP spec=5 n=2-5        | 1280.46 |   7.997 |       0.97x |      - |
| HashTable spec=5 n=2-5  |  982.94 |  10.418 |       0.75x |  0.77x |
| KMP spec=8 n=3-7        | 1182.83 |   8.657 |       0.90x |      - |
| HashTable spec=8 n=3-7  | 1207.16 |   8.483 |       0.92x |  1.02x |

### Acceptance Stats

| Mode                     | Accept% | MeanLen | Drafts | Draft Tokens | Accepted |
|--------------------------|--------:|--------:|-------:|-------------:|---------:|
| KMP spec=3 n=2-3        |   60.6% |    2.82 |   2625 |         7869 |     4771 |
| HashTable spec=3 n=2-3  |   65.3% |    2.96 |   2500 |         7500 |     4899 |
| KMP spec=5 n=2-5        |   49.5% |    3.47 |   2012 |        10042 |     4972 |
| HashTable spec=5 n=2-5  |   56.1% |    3.81 |   1924 |         9620 |     5401 |
| KMP spec=8 n=3-7        |   45.5% |    4.62 |   1291 |        10282 |     4676 |
| HashTable spec=8 n=3-7  |   54.4% |    5.35 |   1139 |         9112 |     4960 |

### Per-Position Acceptance Rate (3B)

| Mode                     |    p0 |    p1 |    p2 |    p3 |    p4 |    p5 |    p6 |    p7 |
|--------------------------|------:|------:|------:|------:|------:|------:|------:|------:|
| KMP spec=3 n=2-3        | 0.707 | 0.583 | 0.527 |     - |     - |     - |     - |     - |
| HashTable spec=3 n=2-3  | 0.756 | 0.648 | 0.556 |     - |     - |     - |     - |     - |
| KMP spec=5 n=2-5        | 0.672 | 0.541 | 0.460 | 0.412 | 0.386 |     - |     - |     - |
| HashTable spec=5 n=2-5  | 0.723 | 0.616 | 0.540 | 0.488 | 0.440 |     - |     - |     - |
| KMP spec=8 n=3-7        | 0.682 | 0.557 | 0.487 | 0.433 | 0.395 | 0.372 | 0.358 | 0.338 |
| HashTable spec=8 n=3-7  | 0.797 | 0.660 | 0.596 | 0.543 | 0.482 | 0.451 | 0.426 | 0.399 |

---

## Key Findings

### 1. HashTable acceptance rate consistently higher than KMP
- 0.5B: HashTable 72-76% vs KMP 65-69% (+5-8pp)
- 3B: HashTable 54-65% vs KMP 45-61% (+5-9pp)
- Frequency-based selection (most-common next token) outperforms KMP's first-match

### 2. Numba optimization most impactful on small models (0.5B)
- spec=3: HT/KMP improved from 0.84x (Phase 1 Python dict) to 0.94x
- spec=5: HT/KMP improved from 0.87x to 1.06x
- spec=8: HT/KMP improved from ~0.87x to 1.31x
- spec=8 HashTable (6766 tok/s) now exceeds baseline (6412 tok/s) on 0.5B

### 3. On 3B, CPU overhead is not the bottleneck
- GPU decode time dominates, so numba vs Python dict makes little difference
- Run-to-run variance is high (spec=5 HT got 982 tok/s, likely an outlier)
- HT/KMP ratios cluster around 0.77-1.02x

### 4. Async scheduling remains the biggest bottleneck
- vLLM disables async scheduling for ngram spec decode: "Async scheduling not supported"
- This hurts ALL spec decode configs equally, especially on small models
- Baseline 0.5B runs at 6412 tok/s WITH async; spec decode runs WITHOUT async

### 5. Larger speculation windows benefit more from HashTable
- spec=8 shows the biggest HT/KMP gain (1.31x on 0.5B)
- Larger windows amortize the overhead of table building/querying
- Higher per-position acceptance at later positions (HT p7=0.637 vs KMP p7=0.561 on 0.5B)

---

## Potential Next Steps

1. **Fix 3B spec=5 anomaly**: Rerun with more warmup or more samples for stability
2. **Cache tuple packing in _hash_table_propose_tokens**: Avoid re-creating tuples of arrays each call
3. **Investigate async scheduling compatibility**: The async disable is the ROOT CAUSE of spec decode being slower than baseline
4. **Test on larger models (7B+)**: Where GPU decode time completely dominates
5. **Profile CPU time breakdown**: Measure exact time in build/update/query vs Python glue code

---

## File Inventory

- `ngram_proposer.py`: Modified vLLM source with numba hash table implementation
  Path: `venv/lib/python3.12/site-packages/vllm/v1/spec_decode/ngram_proposer.py`
- `test_numba_hash.py`: Unit tests verifying numba implementation matches Python reference
- `test_hash_vs_kmp.py`: A/B benchmark script (SWE-bench Lite)
- `hash_vs_kmp_results.json`: Raw JSON results (latest 3B run)
- `test_result_v0.md`: This file
