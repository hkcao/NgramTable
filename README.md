# Speculative Decoding Proposer Benchmark

vLLM 投机解码 draft token 预测方案对比：KMP / Hash / Trie / Suffix Decoding。

## 四种方案概述

| 方案 | 核心思路 | 数据结构 | 复杂度 | 来源 |
|---|---|---|---|---|
| **KMP** | 在当前序列中找与末尾 n-gram 完全匹配的历史位置，复制后续 token | KMP failure function | O(n) 扫描 | vLLM 原生 |
| **Hash** | N-gram 频次统计 → argmax 预计算为 Hash 表 → 链式查表 | FreqTable + HashTable | O(1) 查表 | 本项目 |
| **Trie** | 共享前缀树存储所有 n-gram 后续模式，频率加权贪心展开 | Trie (per-root-token) | O(depth) 查询 | 复现 [LookaheadCache](https://github.com/alipay/PainlessInferenceAcceleration) |
| **Suffix** | 后缀树全局匹配 + 概率过滤 + 动态投机长度 | Suffix Tree (C++) | O(match_len) | [Arctic Inference](https://github.com/snowflakedb/ArcticInference) |

## 文件说明

```
ngram/
├── ngram_proposer.py          # KMP / Hash / Trie 三种模式实现（symlink 到 vllm site-packages）
├── test_ngram.py              # 四方案延迟 & 命中率对比测试
├── test_compare.py            # Trie root-only vs Hash vs Suffix 精简对比
├── test_trie_nodesize.py      # Trie node_size 消融实验（root/internal n-gram）
├── test_batch_acceptance.py   # batch 模式测试
├── test_single_repro.py       # 单请求可复现性验证
├── benchmark_results.json     # 最新测试原始数据
├── compare_results.json       # Trie vs Hash vs Suffix 对比数据
├── trie_nodesize_results.json # node_size 消融实验数据
├── pyproject.toml
└── README.md
```

## 环境

venv 位于 `../venv/`，已安装 vllm 0.15.1 + arctic-inference 0.1.2。

`ngram_proposer.py` 通过 symlink 链入 vllm：

```
Agent/venv/lib/.../vllm/v1/spec_decode/ngram_proposer.py -> Agent/ngram/ngram_proposer.py
```

直接编辑 `ngram/ngram_proposer.py` 即可生效，无需复制。

## 模式切换

通过环境变量控制（优先级从高到低）：

| 环境变量 | 值 | 模式 | 说明 |
|---|---|---|---|
| `VLLM_NGRAM_USE_TRIE` | `1` | Trie | 复现 LookaheadCache，优先级最高 |
| `VLLM_NGRAM_USE_HASH` | `1`（默认） | Hash | 频次统计 + O(1) 查表 |
| `VLLM_NGRAM_USE_HASH` | `0` | KMP | vLLM 原始实现，代码完整保留 |

Trie 模式额外环境变量：

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `VLLM_TRIE_NODE_SIZE` | `1` | 根节点 n-gram 大小（推荐 `3`） |
| `VLLM_TRIE_INTERNAL_NODE_SIZE` | `1` | 内部节点 n-gram 大小（保持 `1` 最优） |
| `VLLM_TRIE_BRANCH_LENGTH` | `8` | 单棵 Trie 最大分支深度 |
| `VLLM_TRIE_DECODING_LENGTH` | `64` | 多分支展开最大节点数 |

Suffix Decoding 通过 `speculative_config={"method": "suffix"}` 启用，走 vLLM 独立的 `SuffixDecodingProposer` 路径。

三种 ngram 模式在 `propose()` 内部以 if/else 分支区分，原有 KMP 代码不做任何改动。

## 各方案实现细节

### KMP（vLLM 原生）

在 reversed token 序列上用 KMP failure function 找最长前缀匹配，等价于在原始序列中找与末尾 n-gram 匹配的最早位置，复制其后续 token。Numba JIT 加速，支持 batch 并行。

### Hash（优化版）

维护三层数据结构：

- **FreqTable**（全局）：`dict[tuple[int,...], Counter[int]]` — n-gram context → 后续 token 频次，跨请求持久化
- **HashTable**：`dict[int, int]` — `hash(context_tuple)` → argmax token，用于快速查询
- **LocalFreqTable**（请求级）：每个请求独立的局部频率表，提升位置特异性

查询时链式查表，配合三项优化：

```
input: [..., A, B, C]  n=3, k=3

step 1: merged_freq((A,B,C)) = global + local*3.0
         confidence = top_count/total → 0.75 > 0.3 ✓
         vote: 3-gram, 2-gram, 4-gram 多数同意 ✓ → E
step 2: merged_freq((B,C,E)) → confidence 0.45 > 0.3 ✓, vote ✓ → G
step 3: merged_freq((C,E,G)) → confidence 0.20 < 0.3 ✗ → 停止
output: [E, G]  (只出高置信 draft，不盲猜满 k 个)
```

**三项优化**：

1. **置信度过滤**：每步计算 `top_count / total`，低于阈值（默认 0.3）时停止链式推理，避免盲猜
2. **多 n-gram 投票**：每步同时查询所有 n-gram 窗口（min_n ~ max_n），多数不一致时终止，缓解链式误差放大
3. **请求局部频率叠加**：全局频率 + 请求局部频率（权重 3.0）合并后取 argmax，提升当前上下文的位置特异性

环境变量配置：

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `VLLM_NGRAM_MIN_CONFIDENCE` | `0.3` | 置信度阈值，低于此值停止 draft |
| `VLLM_NGRAM_LOCAL_WEIGHT` | `3.0` | 局部频率权重 |

### Trie（复现 LookaheadCache + N-gram Root 优化）

完整复现 Alipay [PainlessInferenceAcceleration](https://github.com/alipay/PainlessInferenceAcceleration) 的 LookaheadCache 逻辑，并增加 **n-gram root key 优化**：

- **TrieNode / TrieTree / TrieCache** 三层结构，对应原始 Node / Tree / LookaheadCache
- **写入路径**：prompt token 以 `mode='input'` 写入，生成 token 以 `stream_put(mode='output')` 增量写入
- **查询路径**：`hier_get()` 左到右滑动窗口 → `tree.get()` 多分支 DFS 展开（`_dfs_get_freqs` + `_ravel`）→ 返回 `(ids, mask, sizes)` 树形结果
- **vLLM 适配**：在多分支结果上沿 DFS 首路径提取贪心最优单分支，作为 draft token 序列
- **调用约定**：`mode='mix'`（input/output 频率 1:1 加权）
- **剪枝**：half-decay pruning，`squeeze_branch_counts` 每 1024 次更新触发

**N-gram Root Key 优化**（`VLLM_TRIE_NODE_SIZE`）：

原始 LookaheadCache 用单 token 作为根节点 key，导致上下文区分度低、大量无关模式被匹配。N-gram root key 改为用连续 N 个 token 的 tuple 作为根 key（例如 `(token_A, token_B, token_C)` 而非 `token_C`），写入和查询均以滑动窗口方式生成根 key，显著提升上下文特异性。

内部节点保持单 token（`VLLM_TRIE_INTERNAL_NODE_SIZE=1`），经消融实验验证为最优配置。

### Suffix Decoding（Arctic Inference）

vLLM 内置方案，基于 [Suffix Decoding 论文](https://arxiv.org/pdf/2411.04975)：

- 通过 `speculative_config={"method": "suffix", "num_speculative_tokens": N}` 启用
- 底层为 `arctic-inference` C++ 后缀树实现
- 为每个请求构建后缀树（prompt + 生成 output），找最长后缀匹配
- **概率过滤**：`min_token_prob=0.1`，只投机估算概率 > 10% 的 token
- **动态投机长度**：根据匹配质量自动调节，不强制填满 k 个 draft

## 持久化

### Hash 模式

- `VLLM_NGRAM_TABLE_PATH=/path/to/ngramTable`
- 格式：pickle，`{"freq_table": FreqTable, "hash_table": HashTable}`
- 每 100 步异步刷盘，进程退出时同步写入

### Trie 模式

- `VLLM_TRIE_TABLE_PATH=/path/to/trieTable`
- 格式：pickle，存储 `TrieCache.mem`
- 请求完成时异步刷盘

## 运行测试

```bash
cd ngram

# 四方案对比（KMP / Hash / Trie / Suffix + baseline）
../venv/bin/python test_ngram.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --num-samples 20 --max-tokens 512 --gpu-mem 0.8

# Trie root-only vs Hash vs Suffix 精简对比
../venv/bin/python test_compare.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --num-samples 20 --max-tokens 512 --gpu-mem 0.8

# Trie node_size 消融实验（root/internal n-gram 各种组合）
../venv/bin/python test_trie_nodesize.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --num-samples 20 --max-tokens 512 --gpu-mem 0.8
```

---

## Benchmark 结果

**测试环境**：

| 项目 | 配置 |
|---|---|
| 模型 | Qwen/Qwen2.5-3B-Instruct |
| 数据集 | SWE-bench Lite（20 samples） |
| 推理模式 | 单请求顺序模式（sequential） |
| 温度 | 0（greedy decoding） |
| 最大输出 | 512 tokens/request |
| GPU 内存 | 80% |
| 框架 | vLLM 0.15.1 + arctic-inference 0.1.2 |

### 1. 最新性能总览（spec=5，含 Trie n-gram root 优化）

| Mode | AvgLat | MedLat | P90Lat | tok/s | Speedup | Accept% | MeanLen |
|---|---|---|---|---|---|---|---|
| baseline | 5.422s | 5.828s | 5.974s | 87.28 | 1.00x | - | - |
| KMP | 3.395s | 3.779s | 5.267s | 150.8 | 1.70x | 50.6% | 3.52 |
| Hash | 2.926s | 2.627s | 4.981s | 151.3 | **1.85x** | 43.3% | 3.03 |
| Trie-1g (原始) | 2.981s | 3.081s | 4.998s | 158.1 | 1.82x | 29.1% | 2.42 |
| **Trie-2g root** | 2.803s | 2.749s | 4.516s | 158.4 | **1.93x** | 39.8% | 2.99 |
| **Trie-3g root** | **2.795s** | **2.549s** | 4.644s | **159.9** | **1.94x** | **55.7%** | **3.67** |
| Suffix | **2.699s** | **2.418s** | 4.609s | **160.9** | **2.01x** | 54.2% | 2.50 |

> **Trie-3g root** = `VLLM_TRIE_NODE_SIZE=3`（3-gram 根 key，内部节点保持单 token）

### 2. Trie N-gram Root 优化效果

| Trie 配置 | Speedup | Accept% | MeanLen | 提升 vs 原始 |
|---|---|---|---|---|
| Trie-1g (原始) | 1.82x | 29.1% | 2.42 | baseline |
| **Trie-2g root** | **1.93x** | **39.8%** (+10.7pp) | 2.99 | +6.0% speedup |
| **Trie-3g root** | **1.94x** | **55.7%** (+26.6pp) | 3.67 | +6.6% speedup |

**Root n-gram 的原理**：原始 Trie 用单 token 做根 key（上下文区分度低），改为用 N 个连续 token 的 tuple 做根 key 后，上下文特异性大幅提升，命中率从 29.1% 跃升至 55.7%。

### 3. Trie Node Size 消融实验

测试了根节点和内部节点分别使用不同 n-gram 大小的组合：

| 配置 | Root NS | Internal NS | Speedup | Accept% | MeanLen |
|---|---|---|---|---|---|
| Trie-1g (原始) | 1 | 1 | 1.82x | 29.1% | 2.42 |
| **Trie-2g root-only** | **2** | **1** | **1.93x** | **39.8%** | **2.99** |
| **Trie-3g root-only** | **3** | **1** | **1.94x** | **55.7%** | **3.67** |
| Trie-2g full | 2 | 2 | 1.87x | 42.8% | 2.95 |
| Trie-3g full | 3 | 3 | 1.90x | 53.1% | 3.55 |
| Trie-2g root+3g int | 2 | 3 | 1.81x | 40.5% | 2.88 |

**结论**：**Root-only n-gram（内部节点保持单 token）是最优配置**。内部节点使用多 token key 反而降低性能，因为：
- 匹配粒度变粗，边界处短 tuple 与完整 tuple 的 key 不同导致匹配错过
- 内部节点处于树的深层分支，单 token 匹配的灵活性更有优势

### 4. Draft 质量详细对比（spec=5）

| 指标 | Hash | Trie-2g | **Trie-3g** | Suffix |
|---|---|---|---|---|
| Draft 轮数 | 2,290 | 2,429 | **1,818** | 3,279 |
| Draft token 总数 | 10,743 | 12,144 | **8,732** | 9,080 |
| 命中 token 总数 | 4,652 | 4,836 | **4,863** | 4,920 |
| 命中率 | 43.3% | 39.8% | **55.7%** | 54.2% |
| 平均接受长度 | 3.03 | 2.99 | **3.67** | 2.50 |
| 每轮有效 token | 3.03 | 2.99 | **3.67** | 2.50 |

**逐位置接受率**：

```
Position   Hash    Trie-2g  Trie-3g  Suffix
pos-1     62.8%   59.8%    77.0%    52.0%
pos-2     47.7%   45.0%    61.9%    34.6%
pos-3     37.0%   36.8%    53.4%    26.2%
pos-4     30.5%   31.1%    45.7%    19.9%
pos-5     25.1%   26.4%    29.5%    17.3%
```

**Trie-3g root 在 draft 质量维度全面领先**：
- **Draft 轮数最少（1,818）**：Suffix 需要 3,279 轮，说明 Trie-3g 每轮产出效率远高
- **命中率最高（55.7%）**：超过 Suffix 的 54.2%
- **平均接受长度最长（3.67）**：Suffix 仅 2.50，每轮多接受 1.17 个 token
- **逐位置接受率全面领先**：pos1=77.0% 远超 Suffix 的 52.0%

### 5. 延迟 vs Draft 质量解读

Trie-3g root 的 draft 质量全面优于 Suffix，但延迟略高（1.94x vs 2.01x），原因：
- **Suffix 底层为 arctic-inference C++ 实现**，后缀树匹配和候选生成在 C++ 层完成，CPU 开销极低
- **Trie 为纯 Python 实现**，dict 查找、DFS 展开、mask 构建均在 Python 层，存在固定的解释器开销
- 如果 Trie 方案也用 C++ 优化底层数据结构，基于其更优的 draft 质量，有潜力超越 Suffix

### 6. 历史基线对比（spec=5，优化前 Trie）

| Mode | Speedup (优化前) | Speedup (优化后) | Accept% (前→后) |
|---|---|---|---|
| Trie-1g | 1.79x | 1.82x | 26.8% → 29.1% |
| **Trie-3g root** | - | **1.94x** | - → **55.7%** |

Trie 方案从 1.79x 提升至 1.94x（+8.4%），主要归功于 n-gram root key 优化将命中率翻倍。

## 接受率差异的本质分析

### KMP — 50.6%：精确匹配，"宁缺毋滥"

在当前 prompt + 已生成序列中，找一个与末尾 n 个 token 完全相同的历史位置，直接复制后续 token。精确匹配 = 强上下文关联，但找不到就不 draft，导致 draft 频率低。

### Hash — 43.3%（优化后）：置信度过滤 + 多 n-gram 投票

全局 + 请求局部频率合并后取 argmax。三项优化（置信度过滤、多 n-gram 投票、局部频率叠加）将加速比从 1.75x 提升至 1.85x。

### Trie-3g root — 55.7%：n-gram 根 key + mix 模式

用 3 个连续 token 的 tuple 做 Trie 根 key，大幅提升上下文特异性。`mode='mix'`（input/output 频率 1:1 加权）让 prompt 和已生成内容的模式都参与预测。**在 draft 质量维度全面领先所有方案。**

### Suffix — 54.2%：概率过滤 + 动态长度

C++ 后缀树找最长后缀匹配，概率过滤（`min_token_prob=0.1`）只提议高置信 token。avg_draft/step=2.50，"猜得少但猜得准"。**C++ 实现带来的低 CPU 开销是其延迟优势的主要来源。**

## 综合结论

### 方案排名（spec=5，最新）

| 排名 | 方案 | Speedup | Accept% | 核心优势 | 核心劣势 |
|---|---|---|---|---|---|
| 1 | **Suffix** | **2.01x** | 54.2% | C++ 后缀树 + 概率过滤，延迟最低 | 依赖 arctic-inference 扩展 |
| 2 | **Trie-3g root** | **1.94x** | **55.7%** | Draft 质量最优，纯 Python 无依赖 | Python 解释器开销限制了延迟 |
| 3 | **Trie-2g root** | **1.93x** | 39.8% | 较好的精度/复杂度平衡 | 同上 |
| 4 | **Hash** | **1.85x** | 43.3% | O(1) 查询 + 置信度控制 | 需参数调优 |
| 5 | **KMP** | **1.70x** | 50.6% | 零配置、零依赖 | draft 频率低 |

### 关键发现

1. **Trie n-gram root 优化是本项目最大突破**：命中率从 29.1% → 55.7%，加速比从 1.82x → 1.94x，仅通过改变根 key 的粒度
2. **Trie-3g 的 draft 质量已经超越 Suffix**（命中率 55.7% vs 54.2%，平均接受长度 3.67 vs 2.50），延迟差距（1.94x vs 2.01x）纯粹来自实现语言差异
3. **Root-only n-gram 是最优配置**，内部节点保持单 token 匹配更灵活
4. **"猜得准"比"猜得多"更重要**：Suffix avg_draft=2.50 最低但加速最快；Trie-3g avg_draft 虽略高但命中率最高

### 优化方向

- **Trie C++ 化**：将 TrieTree/TrieCache 用 C++ 重写，消除 Python 解释器开销，预期可达到甚至超过 Suffix 的延迟水平
- **动态 draft 长度**：为 Trie 加入类似 Suffix 的置信度过滤，在低命中时提前终止
- **跨请求持久化**：Trie 和 Hash 在多请求场景下随历史积累效果会持续提升
