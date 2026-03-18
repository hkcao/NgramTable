# Speculative Decoding N-gram Proposer

vLLM 投机解码 draft token 预测方案对比与优化实验。

## 目录结构

```
ngram/
├── ngram_proposer.py          # 核心实现：KMP / Hash / Trie / SkipGram（symlink 到 vllm）
├── test_compare.py            # 主 benchmark 脚本（所有模式对比）
├── compare_results.json       # 最新 benchmark 原始数据
├── README.md
├── pyproject.toml
├── docs/
│   ├── 现有方案分析.md         # 各方案原理深入分析
│   └── skipgram_方案说明.md    # SkipGram 方案设计文档
├── benchmarks/                # 历史测试脚本和结果
│   ├── test_ngram.py          # 四方案对比（KMP/Hash/Trie/Suffix）
│   ├── test_trie_nodesize.py  # Trie node_size 消融实验
│   ├── test_suffix_minmatch.py # Suffix min_match_len 实验
│   ├── test_pysuffix.py       # PySuffix vs C++ Suffix 对比
│   ├── suffix_decoding.py     # PySuffix 实现（已废弃）
│   ├── ...                    # 其他历史测试脚本
│   └── results/               # 所有历史 benchmark JSON
└── analysis/
    ├── analyze_trie.py        # Trie 结构分析脚本
    ├── trie_dump.pkl          # Trie 数据快照
    └── trie_analysis/         # 分析报告 + 可视化图表
```

## 环境

- **框架**: vLLM 0.15.1 + arctic-inference 0.1.2
- **venv**: `../venv/`
- **Symlink**: `ngram_proposer.py` → vllm site-packages，直接编辑即生效

## 模式切换

通过环境变量控制（优先级从高到低）：

| 环境变量 | 值 | 模式 |
|---|---|---|
| `VLLM_NGRAM_USE_TRIE` | `1` | Trie 模式 |
| `VLLM_NGRAM_USE_SKIPGRAM` | `1` | SkipGram 模式（Hash + skip-gram fallback） |
| `VLLM_NGRAM_USE_HASH` | `1`（默认） | Hash 模式 |
| `VLLM_NGRAM_USE_HASH` | `0` | KMP 模式（vLLM 原始） |

Trie 额外参数：

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `VLLM_TRIE_NODE_SIZE` | `1` | 根节点 n-gram 大小（推荐 `3`） |
| `VLLM_TRIE_FUZZY` | `0` | Fuzzy 匹配（允许 N 个 mismatch） |
| `VLLM_TRIE_SKIPGRAM` | `0` | Trie 内部 skip-gram（sentinel 节点） |

Suffix 通过 `speculative_config={"method": "suffix"}` 启用。

## 运行测试

```bash
cd ngram
../venv/bin/python test_compare.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --num-samples 20 --max-tokens 512 --gpu-mem 0.8
```

---

## 全部实验总结

**测试环境**: Qwen/Qwen2.5-3B-Instruct, SWE-bench Lite 20 samples, spec=5, greedy decoding, vLLM 0.15.1

### 一、基线方案对比

| 排名 | 方案 | Speedup | Accept% | MeanLen | 核心策略 |
|------|------|---------|---------|---------|----------|
| 1 | **Suffix** | **2.04x** | 54.2% | 2.50 | C++ 后缀树精确匹配 + 概率过滤 |
| 2 | **Trie-3g root** | **1.95x** | **55.7%** | **3.67** | 3-gram 根 key + 频率加权 |
| 3 | Hash (优化) | 1.86x | 43.3% | 3.03 | 频次统计 + 置信度过滤 + 多 n-gram 投票 |
| 4 | KMP | 1.70x | 50.6% | 3.52 | 序列内精确匹配复制 |

> Trie-3g draft 质量最优（accept 55.7%, MeanLen 3.67），但 Python 实现的 overhead 使延迟略逊于 C++ Suffix。

### 二、Trie N-gram Root 优化（成功）

**核心思路**：用 N 个连续 token 的 tuple 做 Trie 根 key，提升上下文特异性。

| 配置 | Root NS | Internal NS | Speedup | Accept% | MeanLen |
|------|---------|-------------|---------|---------|---------|
| Trie-1g (原始) | 1 | 1 | 1.82x | 29.1% | 2.42 |
| Trie-2g root | 2 | 1 | 1.93x | 39.8% | 2.99 |
| **Trie-3g root** | **3** | **1** | **1.94x** | **55.7%** | **3.67** |
| Trie-2g full | 2 | 2 | 1.87x | 42.8% | 2.95 |
| Trie-3g full | 3 | 3 | 1.90x | 53.1% | 3.55 |

**结论**: Root-only n-gram（内部节点保持单 token）是最优配置。命中率从 29.1% → 55.7%（+26.6pp）。

### 三、模糊匹配实验（均失败）

以下三种尝试都试图在匹配失败时通过"容忍部分不匹配"提高覆盖率，但均导致 accept rate 下降。

#### 3.1 Trie Fuzzy 匹配（Trie-3g+Fuzzy）

**思路**：Trie lookup 时允许 1 个 token mismatch（DFS 搜索所有 budget 内路径）。

| 模式 | Speedup | Accept% | MeanLen |
|------|---------|---------|---------|
| Trie-3g root | 1.95x | 55.7% | 3.67 |
| **Trie-3g+Fuzzy** | **1.84x** | **43.0%** | **3.04** |

**失败原因**: 3-gram root 匹配后只剩 2 个 internal token 上下文。允许 1 个 mismatch → 实际只有 1 个精确匹配 → 预测接近随机。Fuzzy 走的是某一条具体路径的频率，没有统计聚合。

#### 3.2 Hash SkipGram（Hash + skip-gram fallback）

**思路**：额外维护一张 skip-gram 哈希表（key 中一个位置用 sentinel 替代），精确匹配失败时用 skip-gram 查表，置信度打 0.7 折。

| 模式 | Speedup | Accept% | MeanLen | Draft 数量 | 浪费 |
|------|---------|---------|---------|-----------|------|
| Hash | 1.86x | 43.3% | 3.03 | 10,743 | 6,091 |
| **SkipGram** | **2.06x** | **31.0%** | **2.55** | **17,497** | **12,067** |

**分析**: Speedup 最高（2.06x），但本质是"覆盖率换精度"——每步几乎满 5 draft → GPU 单次验证更多 token → 减少 forward 次数。代价是 accept 仅 31%，浪费严重（12,067 rejected tokens），不适合高并发批处理。

#### 3.3 Trie 内部 SkipGram（Trie+Skip）

**思路**：在 Trie 的 internal node 层加入 skip-gram——build 时对每个序列的每个位置插入 sentinel 变体节点，lookup 时精确匹配失败后走 sentinel 子节点。

| 模式 | Speedup | Accept% | MeanLen |
|------|---------|---------|---------|
| Trie-3g root | 1.94x | 55.7% | 3.67 |
| **Trie+Skip (node_size=1)** | **1.74x** | **25.6%** | **2.27** |

**失败原因**: node_size=1 时 root 为单个高频 token，覆盖率已经很高。sentinel 节点聚合了所有不同 token 的后续，频率分布极其分散 → 预测质量极低。加上 Trie 的 Python overhead，比 Hash SkipGram 更差。

### 四、Suffix min_match_len 实验（无收益）

**思路**: 提高 Suffix 的最低匹配长度要求（类似 Trie n-gram root 的效果）。

| min_match_len | Speedup | Accept% | MeanLen |
|---------------|---------|---------|---------|
| 0 (default) | **2.02x** | 54.2% | 2.50 |
| 1 | 2.02x | 54.2% | 2.50 |
| 2 | 1.91x | 62.1% | 2.68 |
| 3 | 1.82x | 68.5% | 2.95 |

**结论**: Accept% 随 min_match_len 上升，但 draft 数量减少更多，净效果为负。min=0 即为最优。

> **重要发现**: Trie 的 3-gram root key 和 Suffix 的 min_match_len 本质等价——都要求至少匹配 N 个 token 才提取后续。但 suffix 更灵活（不需固定 root 大小，可动态匹配任意长度后缀），后续对比直接用 Suffix 做基准。

### 五、PySuffix 实验（已废弃）

Python 实现的后缀匹配，验证 C++ suffix 的策略优势来源。结论：C++ suffix 严格优于 Python 实现（2.02x vs Python overhead），PySuffix 已移除。

### 六、关键洞察

1. **Draft 质量 vs 实现效率的权衡**: Trie-3g 的 draft 质量全面领先（accept 55.7%, MeanLen 3.67），但 Python 实现的 CPU overhead 使其 speedup 略逊于 C++ Suffix。**Trie C++ 化后有潜力超越 Suffix**。

2. **模糊匹配的困境**: 所有模糊匹配尝试（Fuzzy、SkipGram、Trie+Skip）都面临同一个根本问题——放松匹配条件 → 频率分布分散 → 预测质量下降。"覆盖率"策略只在 GPU 空闲的单请求场景有意义。

3. **"猜得准"比"猜得多"更重要**: Suffix avg_draft=2.50（最少）但 speedup 最高。SkipGram avg_draft=5.0（满额）但大量浪费。精度优先 > 覆盖率优先。

4. **Root-only n-gram 是 Trie 的最优配置**: 内部节点保持单 token 匹配更灵活，多 token 内部 key 反而因边界匹配问题降低性能。

5. **Trie 结构分析**: 分支因子随深度单调递减（1.35→1.02），不存在菱形（中间胖两头窄）结构。少量 hot root（5.8%）贡献大部分预测能力。

### 七、实验时间线

| 日期 | 实验 | 结果 |
|------|------|------|
| 03-06 | KMP / Hash / Trie / Suffix 基线对比 | Suffix 2.01x 最优 |
| 03-07 | Hash 优化（置信度 + 投票 + 局部频率） | 1.75x → 1.86x |
| 03-10 | Trie n-gram root 优化 | 1.82x → 1.95x, accept 29% → 56% |
| 03-11 | Trie node_size 消融实验 | root-only 最优 |
| 03-11 | Suffix min_match_len 实验 | min=0 最优，无收益 |
| 03-13 | PySuffix vs C++ Suffix | C++ 严格优于，废弃 PySuffix |
| 03-14 | 可复现性验证 | 各方案结果稳定 |
| 03-15 | Trie 结构可视化分析 | 无菱形结构，漏斗→竹竿 |
| 03-17 | Trie Fuzzy 匹配 | 失败，accept 56% → 43% |
| 03-17 | Hash SkipGram (fallback) | 2.06x 但 accept 仅 31% |
| 03-18 | Trie 内部 SkipGram | 失败，accept 26%, speedup 1.74x |
