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
| 总输入 | 9,224 tokens（20 条 prompt） |
| 总输出 | 10,240 tokens（全部达到 max_tokens） |
| GPU 内存 | 80% |
| 框架 | vLLM 0.15.1 + arctic-inference 0.1.2 |

### 1. 性能总览

#### spec=3（num_speculative_tokens=3）

| Mode | AvgLat | MedianLat | P90Lat | tok/s | Speedup | Accept% | MeanLen | TotalTime |
|---|---|---|---|---|---|---|---|---|
| baseline | 5.779s | 5.786s | 5.828s | 88.6 | 1.00x | - | - | 115.6s |
| KMP n=2-3 | 3.725s | 4.031s | 5.326s | 137.4 | **1.55x** | 57.5% | 2.72 | 74.5s |
| Hash n=2-3 | 3.432s | 3.099s | 5.058s | 149.2 | **1.68x** | 58.1% | 2.73 | 68.6s |
| Trie n=2-3 | 3.562s | 3.390s | 4.996s | 143.7 | **1.62x** | 35.9% | 2.07 | 71.2s |
| Suffix | 3.261s | 2.950s | 5.022s | 155.1 | **1.77x** | **61.4%** | 2.29 | 65.2s |

#### spec=5（num_speculative_tokens=5）

| Mode | AvgLat | MedianLat | P90Lat | tok/s | Speedup | Accept% | MeanLen | TotalTime |
|---|---|---|---|---|---|---|---|---|
| baseline | 5.779s | 5.786s | 5.828s | 88.6 | 1.00x | - | - | 115.6s |
| KMP n=2-5 | 3.395s | 3.779s | 5.267s | 150.8 | **1.70x** | 50.6% | 3.52 | 67.9s |
| Hash n=2-5 | 3.188s | 2.834s | 5.151s | 160.6 | **1.81x** | 47.2% | 3.25 | 63.8s |
| Trie n=2-5 | 3.221s | 2.997s | 4.864s | 159.0 | **1.79x** | 26.8% | 2.33 | 64.4s |
| Suffix | 2.941s | 2.632s | 4.829s | 171.9 | **1.96x** | **60.3%** | 2.57 | 58.8s |

### 2. spec=3 → spec=5 增益分析

| Mode | Speedup (3→5) | tok/s (3→5) | Accept% (3→5) | 延迟改善 |
|---|---|---|---|---|
| KMP | 1.55x → 1.70x (+9.7%) | 137.4 → 150.8 (+9.8%) | 57.5% → 50.6% (-6.9pp) | -0.330s |
| Hash | 1.68x → 1.81x (+7.7%) | 149.2 → 160.6 (+7.6%) | 58.1% → 47.2% (-10.9pp) | -0.244s |
| Trie | 1.62x → 1.79x (+10.5%) | 143.7 → 159.0 (+10.6%) | 35.9% → 26.8% (-9.1pp) | -0.341s |
| Suffix | 1.77x → 1.96x (+10.7%) | 155.1 → 171.9 (+10.8%) | 61.4% → 60.3% (-1.1pp) | -0.320s |

**关键发现**：

- 所有方案从 spec=3 升至 spec=5 均获得 7.7%~10.8% 的吞吐提升，说明增加投机长度在该工作负载下仍有正收益
- **接受率普遍下降**是符合预期的：更多投机位置 → 后续位置命中率递减 → 总体接受率被稀释
- **Suffix 接受率几乎不变**（60.3% vs 61.4%，仅降 1.1pp），因其概率过滤机制只在高置信时才延长 draft，avg_draft/step 仅从约 2.1 增至 2.6
- Trie 在 spec=5 下加速比（1.79x）反而接近 Hash（1.81x），因为更长投机序列让其"蒙对"的收益增大，部分抵消了低接受率的损失

### 3. Draft 效率详细统计

#### spec=3

| 方案 | Drafts | Draft Tokens | Accepted | avg_draft/step | 接受率 | 浪费 tokens | 效率 |
|---|---|---|---|---|---|---|---|
| KMP | 2,636 | 7,902 | 4,540 | 3.00 | 57.5% | 3,362 | 57.5% |
| Hash | 2,901 | 8,646 | 5,021 | 2.98 | 58.1% | 3,625 | 58.1% |
| Trie | 4,629 | 13,791 | 4,952 | 2.98 | 35.9% | 8,839 | 35.9% |
| Suffix | 4,120 | 8,627 | 5,299 | 2.09 | 61.4% | 3,328 | **61.4%** |

#### spec=5

| 方案 | Drafts | Draft Tokens | Accepted | avg_draft/step | 接受率 | 浪费 tokens | 效率 |
|---|---|---|---|---|---|---|---|
| KMP | 2,044 | 10,180 | 5,149 | 4.98 | 50.6% | 5,031 | 50.6% |
| Hash | 2,416 | 11,545 | 5,447 | 4.78 | 47.2% | 6,098 | 47.2% |
| Trie | 4,126 | 20,372 | 5,467 | 4.94 | 26.8% | 14,905 | 26.8% |
| Suffix | 3,673 | 9,563 | 5,767 | **2.60** | **60.3%** | 3,796 | **60.3%** |

**效率指标解读**：

- **Suffix 以最少的 draft tokens（9,563）获得了最多的 accepted tokens（5,767）**，浪费率仅 39.7%
- Trie 浪费最严重：20,372 draft tokens 中只有 5,467 被接受（73.2% 浪费），但仍获得了最多的绝对 accepted tokens 之一
- KMP draft 次数最少（2,044），说明大量 step 找不到匹配，退化为逐 token 生成
- Hash 的 avg_draft/step=4.78（低于理论最大 5），正是置信度过滤在起作用

### 4. 逐位置接受率衰减（spec=5）

```
Position    KMP     Hash    Trie    Suffix
pos-1      67.4%   56.7%   40.7%   50.0%
pos-2      54.1%   44.8%   28.0%   34.2%
pos-3      46.8%   36.4%   23.8%   27.6%
pos-4      43.3%   31.9%   20.8%   23.4%
pos-5      40.3%   29.3%   19.0%   21.8%
衰减率     -40.2%  -48.3%  -53.3%  -56.4%
```

> 衰减率 = (pos-5 - pos-1) / pos-1

**分析**：

- **KMP 衰减最平缓**（-40.2%）：因为它只在"精确匹配"时 draft，匹配到的历史位置后续 token 有很强的连贯性
- **Suffix 衰减最陡**（-56.4%），但 pos-1 仍然较高（50.0%）。这正是"概率过滤 + 动态长度"策略的体现：大多数时候只提议 1-2 个高置信 token，少数时候才延伸到 pos-4/5
- **Trie 全位置都低**：根本问题是 2-token 上下文区分度不足，从 pos-1 开始就只有 40.7%
- **Hash** 介于 KMP 和 Trie 之间，置信度过滤使其在不确定时提前终止

### 5. 延迟分布特征

| Mode (spec=5) | avg | median | P90 | std | min | max | range |
|---|---|---|---|---|---|---|---|
| baseline | 5.779s | 5.786s | 5.828s | 0.042s | 5.710s | 5.833s | 0.124s |
| KMP | 3.395s | 3.779s | 5.267s | 1.180s | 1.474s | 5.455s | 3.981s |
| Hash | 3.188s | 2.834s | 5.151s | 1.242s | 1.146s | 5.171s | 4.026s |
| Trie | 3.221s | 2.997s | 4.864s | 1.181s | 1.166s | 5.284s | 4.118s |
| Suffix | 2.941s | 2.632s | 4.829s | 1.198s | 1.153s | 5.188s | 4.034s |

**关键观察**：

- **Baseline 几乎无方差**（std=0.042s），因为每个请求都生成固定 512 tokens，无投机波动
- **所有投机方案方差相当**（std ≈ 1.18-1.24s），延迟跨度约 4s，说明投机解码效果高度依赖请求内容的可预测性
- **median vs avg 的差异**揭示了分布形态：
  - KMP: median(3.779) > avg(3.395) — 少量请求被大幅加速（拉低均值），但多数请求加速有限
  - Hash/Trie/Suffix: median < avg — 多数请求加速良好，少数难预测请求拖高均值
- **P90 ≈ 5s** 对所有方案都成立，说明约 10% 的请求几乎无法投机加速（内容缺乏可复用模式）
- **最佳单请求加速**：Hash 和 Suffix 的最小延迟约 1.15s，相当于 baseline 的 **5x 加速**

## 接受率差异的本质分析

### KMP — 50.6%：精确匹配，"宁缺毋滥"

**机制**：在当前 prompt + 已生成序列中，找一个与末尾 n 个 token 完全相同的历史位置，直接复制后续 token。

**高接受率的原因**：

- 精确匹配 = 强上下文关联。找到的匹配点和当前位置共享完全相同的前缀，后续 token 大概率一致（尤其是代码、结构化文本中的重复模式）
- **找不到就不 draft**。KMP 只有 2,044 次 draft（四种方案中最少），说明很多 step 它找不到匹配直接放弃。这种"不确定就不猜"的策略天然拉高了 acceptance rate

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
| Hash | 4.78 |
| Trie | 4.94 |
| **Suffix** | **2.60** |

Suffix 平均每步只提议 2.6 个 token（而非满打满算 5 个）。`min_token_prob=0.1` 让它**只在有把握时出手**：匹配强时提议多个 token，匹配弱时只提议 1-2 个高置信 token。

**2) 同时利用 prompt + output 全部模式**

Suffix 的后缀树包含所有 token（不区分 input/output mode），从第 1 个 token 开始就有丰富的 prompt 模式可用，不存在冷启动问题。

**3) 最长后缀匹配 = 最高上下文特异性**

后缀树找到的是当前序列末尾与历史序列的最长公共后缀，匹配长度不受固定 n-gram 窗口限制。匹配越长，上下文越具体，预测越准。

### 加速效果的本质

```
加速比 ≈ f(有效 token 数/GPU 步, CPU 提议开销, GPU 验证开销)
```

**Suffix 用最少的 draft token（9,563）拿到了最多的 accepted token（5,767）**，浪费率 39.7% 为全场最低。这才是加速比最高的本质——不是猜得多，而是猜得准。

## 综合结论

### 方案排名（spec=5）

| 排名 | 方案 | Speedup | 核心优势 | 核心劣势 |
|---|---|---|---|---|
| 1 | **Suffix** | **1.96x** | 概率过滤 + 动态长度，效率最高 | 依赖 arctic-inference C++ 扩展 |
| 2 | **Hash** | **1.81x** | O(1) 查询 + 置信度控制，可跨请求持久化 | 需要参数调优（置信度阈值、权重） |
| 3 | **Trie** | **1.79x** | 结构化知识存储，多轮对话潜力大 | 单请求场景下上下文不足 |
| 4 | **KMP** | **1.70x** | 零配置、零依赖、实现简单 | draft 频率低，无法跨 n-gram 积累 |

### 适用场景推荐

| 方案 | 最佳场景 | 不适合场景 |
|---|---|---|
| **KMP** | 高重复文本（代码模板、结构化输出）、快速部署 | 低重复率文本、需要高吞吐 |
| **Hash** | 跨请求持久化服务、高吞吐 batch 推理（O(1) 查询） | 冷启动场景（无历史积累） |
| **Trie** | 长期服务、多轮对话（历史持续积累） | 单请求隔离测试、短对话 |
| **Suffix** | **通用场景最优**，单请求即可发挥全部能力 | 无 arctic-inference 的环境 |

### 设计启示

1. **"猜得准"比"猜得多"更重要**。Suffix 的 avg_draft=2.60 远低于其他方案的 ~5.0，但加速比最高。这说明在投机解码中，**精准控制 draft 长度**是第一优先级
2. **置信度机制是通用优化方向**。Hash 通过置信度过滤将加速比从 1.75x 提升至 1.81x，验证了"知道什么时候不该猜"和"猜得准"同样重要
3. **上下文长度决定预测质量**。Trie 的 2-token 上下文是其瓶颈所在；KMP 的全序列精确匹配则提供了最强的上下文保证
4. **动态投机长度应成为标配**。固定填满 k 个 draft 的策略（KMP/Trie）在后续位置浪费严重，动态控制（Suffix/Hash-optimized）才是正确方向
