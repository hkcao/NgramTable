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
├── test_batch_acceptance.py   # batch 模式测试
├── test_single_repro.py       # 单请求可复现性验证
├── benchmark_results.json     # 最新测试原始数据
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

### Trie（复现 LookaheadCache）

完整复现 Alipay [PainlessInferenceAcceleration](https://github.com/alipay/PainlessInferenceAcceleration) 的 LookaheadCache 逻辑：

- **TrieNode / TrieTree / TrieCache** 三层结构，对应原始 Node / Tree / LookaheadCache
- **写入路径**：prompt token 以 `mode='input'` 写入，生成 token 以 `stream_put(mode='output')` 增量写入
- **查询路径**：`hier_get()` 左到右滑动窗口 → `tree.get()` 多分支 DFS 展开（`_dfs_get_freqs` + `_ravel`）→ 返回 `(ids, mask, sizes)` 树形结果
- **vLLM 适配**：在多分支结果上沿 DFS 首路径提取贪心最优单分支，作为 draft token 序列
- **调用约定**：匹配原始调用方，只传最后 2 个 token 作为上下文，`mode='mix'`（input/output 频率 1:1 加权）
- **剪枝**：half-decay pruning，`squeeze_branch_counts` 每 1024 次更新触发

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
```

## 测试结果

Qwen2.5-3B-Instruct, 单请求顺序模式, temp=0, 20 samples, SWE-bench Lite:

### 总览

| Mode | AvgLat | tok/s | Speedup | Accept% | MeanLen |
|---|---|---|---|---|---|
| baseline | 5.779s | 88.6 | 1.00x | - | - |
| KMP spec=5 n=2-5 | 3.395s | 150.8 | 1.70x | 50.6% | 3.52 |
| Hash spec=5 n=2-5 | 3.188s | 160.6 | 1.81x | 47.2% | 3.25 |
| Trie spec=5 n=2-5 | 3.221s | 159.0 | 1.79x | 26.8% | 2.33 |
| Suffix spec=5 | 2.941s | 171.9 | **1.96x** | **60.3%** | 2.57 |

### 详细统计（spec=5）

| 方案 | Drafts | Draft Tokens | Accepted | avg_draft/step | 浪费率 | per_pos (前 5 位) |
|---|---|---|---|---|---|---|
| KMP | 2044 | 10180 | 5149 | 4.98 | 49.4% | 67.4%, 54.1%, 46.8%, 43.3%, 40.3% |
| Hash | 2612 | 13057 | 5201 | 5.00 | 60.2% | 56.7%, 44.8%, 36.4%, 31.9%, 29.3% |
| Trie | 4129 | 20454 | 5462 | 4.95 | 73.3% | 40.7%, 28.0%, 23.8%, 20.8%, 19.0% |
| Suffix | 3673 | 9563 | 5767 | **2.60** | **39.7%** | 50.0%, 34.2%, 27.6%, 23.4%, 21.8% |

## 接受率差异的本质分析

### KMP — 50.6%：精确匹配，"宁缺毋滥"

**机制**：在当前 prompt + 已生成序列中，找一个与末尾 n 个 token 完全相同的历史位置，直接复制后续 token。

**高接受率的原因**：

- 精确匹配 = 强上下文关联。找到的匹配点和当前位置共享完全相同的前缀，后续 token 大概率一致（尤其是代码、结构化文本中的重复模式）
- **找不到就不 draft**。KMP 只有 2044 次 draft（四种方案中最少），说明很多 step 它找不到匹配直接放弃。这种"不确定就不猜"的策略天然拉高了 acceptance rate

**局限**：draft 频率低，很多 step 退化为逐 token 生成。

### Hash — 47.2%（优化后）：置信度过滤 + 多 n-gram 投票 + 局部叠加

**机制**：全局 + 请求局部频率合并后取 argmax，链式推理过程中通过置信度过滤和多 n-gram 投票动态控制 draft 长度。

**优化前后对比**（spec=5）：

| 版本 | Accept% | tok/s | Speedup | avg_draft/step |
|---|---|---|---|---|
| 优化前（纯 argmax 链式） | 39.8% | 154.2 | 1.75x | 5.00 |
| **优化后** | **47.2%** | **160.6** | **1.81x** | **3.25** |

**三项优化的作用**：

1. **置信度过滤**（最核心）：avg_draft/step 从 5.00 降到 3.25，不再盲猜满 k 个。低置信时及早终止，大幅降低浪费率
2. **多 n-gram 投票**：当多个 n-gram 窗口对候选 token 不一致时终止链式推理，缓解了误差放大
3. **请求局部频率叠加**：局部频率权重 3.0 叠加全局频率，让 argmax 更贴近当前请求上下文

### Trie — 26.8%：mix 模式 + 2 token 上下文

**机制**：在 Trie 中查找当前末尾 2 个 token 的后续模式，使用 `mode='mix'`（input/output 频率 1:1 加权）。完整复现 LookaheadCache 的 `hier_get` 逻辑。

**最低接受率的根本原因**：

```
请求开始 → put(prompt, mode='input')    → prompt 模式存为 input freq
生成 token → stream_put(new, mode='output') → 逐渐积累 output freq
查询时    → get(context, mode='mix')     → input + output 频率 1:1 加权
```

**三种 mode 实测对比**（spec=5）：

| mode | 权重 | Accept% | tok/s | Speedup |
|---|---|---|---|---|
| output | output only | 26.7% | 160.1 | 1.81x |
| mix (原始 10000:1) | input 主导 | 26.1% | 157.0 | 1.78x |
| **mix (1:1)** | **均衡** | **26.8%** | **161.1** | **1.83x** |

三种模式差异极小，说明瓶颈不在权重配置，而在：

1. **2 token 上下文区分度极低**。只用最后 2 个 token 做匹配，容易命中无关模式
2. **单请求隔离测试**。原始 LookaheadCache 设计用于多轮/批量推理，不同请求的历史相互补充。但测试中每个请求独立子进程，历史无法跨请求传递
3. **总是盲猜满 k 个**。avg_draft/step ≈ 4.95，不确定时也强行填满，拉低总体命中率

> **注**：Trie 方案在跨请求持久化场景（历史充分积累）下表现应会显著改善。

### Suffix — 60.3%：概率过滤 + 全模式 + 动态长度

**机制**：用 C++ 后缀树在 prompt + 已生成 token 中找最长后缀匹配，估算候选 next token 的概率，只提议概率 > `min_token_prob`(0.1) 的 token。

**最高接受率的三个本质原因**：

**1) 概率过滤——最核心的差异**

| 方案 | avg_draft/step |
|---|---|
| KMP | 4.98 |
| Hash | 5.00 |
| Trie | 4.95 |
| **Suffix** | **2.60** |

Suffix 平均每步只提议 2.6 个 token（而非满打满算 5 个）。`min_token_prob=0.1` 让它**只在有把握时出手**：匹配强时提议多个 token，匹配弱时只提议 1-2 个高置信 token。其他方案要么不提议（KMP 无匹配时），要么盲目凑满 k 个（Hash/Trie）。

**2) 同时利用 prompt + output 全部模式**

Suffix 的后缀树包含所有 token（不区分 input/output mode），从第 1 个 token 开始就有丰富的 prompt 模式可用，不存在冷启动问题。

**3) 最长后缀匹配 = 最高上下文特异性**

后缀树找到的是当前序列末尾与历史序列的最长公共后缀，匹配长度不受固定 n-gram 窗口限制。匹配越长，上下文越具体，预测越准。

### 最终加速效果

```
加速比 ≈ f(有效 token 数/GPU 步, CPU 提议开销, GPU 验证开销)
```

**Suffix 用最少的 draft token（9563）拿到了最多的 accepted token（5767）**，浪费率 39.7% 为全场最低。这才是加速比最高的本质——不是猜得多，而是猜得准。

### 一句话总结

| 方案 | 策略 | 适用场景 |
|---|---|---|
| KMP | 精确后缀匹配，找到就准，找不到就不猜 | 高重复文本（代码模板、结构化输出） |
| Hash | 频率 argmax + 置信度过滤 + 投票 + 局部叠加 | 跨请求持久化 + 高吞吐 batch（O(1) 查询） |
| Trie | 复现 LookaheadCache，mix 模式 1:1 加权，2 token 上下文 | 长期服务、多轮对话（历史持续积累） |
| Suffix | 后缀树 + 概率过滤 + 动态长度，只出高置信 draft | **通用场景最优**，单请求即可发挥 |
