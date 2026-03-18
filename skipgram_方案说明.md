# SkipGram N-gram Proposer 方案说明

## 一、方案概述

SkipGram 是在原有 Hash N-gram 方案基础上增加的 **模糊匹配 fallback 层**。核心思想：当精确 n-gram 匹配失败时，使用 **跳过 1 个 token 的模式（skip-gram）** 进行匹配，以提高 draft 覆盖率。

**环境变量**：`VLLM_NGRAM_USE_SKIPGRAM=1`（与 Hash/Trie 模式互斥）

## 二、数据结构

### 2.1 继承自 Hash 模式的表

| 表 | 类型 | Key | Value | 用途 |
|----|------|-----|-------|------|
| `_freq_table` | `dict[tuple, Counter]` | 精确 n-gram context | token → count | 全局频率统计 |
| `_hash_table` | `dict[int, int]` | hash(context) | best_token | 快速查找 |
| `_req_local_freq` | `dict[int, dict[tuple, Counter]]` | per-request 本地频率 | - | 请求级别覆盖 |

### 2.2 新增的 Skip-gram 表

| 表 | 类型 | Key | Value |
|----|------|-----|-------|
| `_skipgram_table` | `dict[tuple, Counter]` | 含 sentinel 的 n-gram pattern | token → count |

**Key 格式示例**（sentinel = -999）：

对于原始 3-gram `(A, B, C) → D`，生成 3 个 skip-gram 条目：
```
(-999,  B,  C) → Counter({D: 1})   # 跳过第 1 个位置
( A, -999,  C) → Counter({D: 1})   # 跳过第 2 个位置
( A,  B, -999) → Counter({D: 1})   # 跳过第 3 个位置
```

## 三、建表流程

### 3.1 全量建表（新请求到达时）

调用 `_build_skipgram_from_tokens(token_ids)`：

```
输入: token 序列 [t0, t1, t2, t3, t4, ...]
对每个 n ∈ [min_n, max_n]:  (配置: min_n=2, max_n=5)
  对每个位置 i:
    context = token_ids[i : i+n]
    next_token = token_ids[i+n]
    对每个 skip_pos ∈ [0, n):
      pattern = context 的副本，把 pattern[skip_pos] 替换为 -999
      skipgram_table[tuple(pattern)][next_token] += 1
```

**示例**：序列 `[the, quick, brown, fox, jumps]`，n=3

| 位置 i | context | next | skip_pos=0 | skip_pos=1 | skip_pos=2 |
|--------|---------|------|-----------|-----------|-----------|
| 0 | (the, quick, brown) | fox | (-999, quick, brown)→fox | (the, -999, brown)→fox | (the, quick, -999)→fox |
| 1 | (quick, brown, fox) | jumps | (-999, brown, fox)→jumps | (quick, -999, fox)→jumps | (quick, brown, -999)→jumps |

**条目膨胀**：每个原始 n-gram 产生 n 个 skip-gram 条目。对 n=2~5，总膨胀约 (2+3+4+5)/(1+1+1+1) = 3.5 倍。

### 3.2 增量更新（生成新 token 后）

调用 `_update_skipgram_table(token_ids, n_new)`：

只处理包含新 token 的窗口（与 Hash 模式的增量更新逻辑相同），避免全量重建。

```
对每个 n ∈ [min_n, max_n]:
  start = max(0, total - n_new - n)  # 只处理涉及新 token 的窗口
  对每个位置 i ∈ [start, total - n):
    ... (同全量建表)
```

### 3.3 同时维护的表

SkipGram 模式在建表时 **同时维护 3 张表**：
1. `_freq_table`（精确 n-gram，全局）
2. `_req_local_freq`（精确 n-gram，per-request）
3. `_skipgram_table`（skip-gram 模式）

精确表用于第一优先级查询，skip-gram 表只在精确查询失败时作为 fallback。

## 四、Draft 生成流程

调用 `_propose_tokens_skipgram(input_ids, k, req_idx)`：

```
extended = input_ids 的副本  # 会追加已预测的 draft token
min_conf = 0.3              # 最低置信度阈值

重复 k 次 (k=5):
  best_token = None
  best_confidence = 0.0

  ┌─ 第 1 层: 精确 n-gram 匹配 ──────────────────────────────┐
  │ 从最长 n-gram (n=5) 到最短 (n=2) 依次尝试:               │
  │   context = extended 末尾 n 个 token                      │
  │   查 freq_table + local_freq (加权合并)                   │
  │   如果命中 → 取最高频 token 及其 confidence               │
  │   如果 confidence ≥ 0.3 → 提前退出 (够好了)              │
  └──────────────────────────────────────────────────────────┘
           ↓ 如果 best_token == None 或 confidence < 0.3
  ┌─ 第 2 层: Skip-gram fallback ────────────────────────────┐
  │ 从最长 n-gram (n=5) 到最短 (n=2) 依次尝试:               │
  │   context = extended 末尾 n 个 token                      │
  │   对每个 skip_pos ∈ [0, n):                               │
  │     pattern = context, 把 pattern[skip_pos] 替换为 -999   │
  │     查 skipgram_table[pattern]                            │
  │     如果命中 → confidence *= 0.7 (打 7 折)                │
  │     如果优于当前 best → 更新 best_token                   │
  │   如果 confidence ≥ 0.3 → 提前退出                       │
  └──────────────────────────────────────────────────────────┘
           ↓
  终止条件:
    - best_token == None → 停止 (完全无匹配)
    - best_confidence < 0.15 (= 0.3 * 0.5) → 停止 (太不确定)

  drafts.append(best_token)
  extended.append(best_token)  # 用 draft 结果继续预测下一个
```

### 关键设计点

**1. 两层优先级**：精确匹配优先。只有精确匹配失败或置信度不足时才启用 skip-gram。避免 skip-gram 在精确匹配已经很好时引入噪声。

**2. 置信度打折（×0.7）**：skip-gram 的 context 比精确 n-gram 少了一个 token 的约束，统计更粗糙，因此对其 confidence 乘以 0.7 作为惩罚。

**3. 链式自回归**：第 1 个 draft token 预测后追加到 extended，用于预测第 2 个 draft。这意味着如果第 1 个 draft 就是 skip-gram 猜的（不太准），后续 draft 会在错误基础上继续预测，质量逐位衰减。这也解释了 benchmark 中 per-position accept rate 从 pos0=46.5% 递减到 pos4=20.5%。

**4. 宽松终止阈值（0.15）**：对 skip-gram 结果使用更低的终止门槛（min_conf × 0.5），允许低置信度的 draft 通过。这是"宁猜错不漏猜"策略的体现。

## 五、与 Suffix 方案的对比

```
                        SkipGram                    Suffix (C++ arctic-inference)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
数据结构         Python dict (hash table)         C++ suffix array (排序后缀)
匹配算法         固定窗口 n-gram hash lookup       全文最长公共子串匹配
上下文范围       末尾 2~5 个 token                  整个已见文本 (prompt + output)
模糊匹配         ✅ skip-gram (跳 1 token)          ❌ 严格精确匹配
Draft 来源       链式查表预测 (每步重新查表)        直接复制匹配位置后的 token
Draft 数量       几乎总是满额 (avg 5.0/step)       取决于匹配后可复制长度 (avg 2.77)
建表开销         O(L × N × n) per token             O(L log L) 一次性构建
查询开销         O(N × n) hash lookups              O(n log L) 后缀数组查找
内存开销         3.5x 于普通 hash 表                O(L) 后缀数组
实现语言         Python                             C++ (arctic-inference)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  L = 已见文本总长度, N = max_n - min_n + 1 (n-gram 阶数范围), n = 具体阶数
```

### 核心策略差异

| 维度 | SkipGram | Suffix |
|------|----------|--------|
| **覆盖率优先 vs 精度优先** | 覆盖率优先 — 几乎总能出 draft | 精度优先 — 没有好匹配就不出 |
| **Draft 质量** | 低 (accept 31%) | 高 (accept 54%) |
| **Draft 数量** | 多 (17497 tokens / 20 requests) | 少 (9080 tokens / 20 requests) |
| **GPU 利用模式** | 每步验证 5 个 token，batch 大 | 每步验证 2.77 个 token，batch 小 |
| **浪费的计算** | 高 (12067 rejected tokens) | 低 (4160 rejected tokens) |

### 为什么 Speedup 接近？

两者 per-step 有效 token 数几乎相同：
- SkipGram: `1 + 5430/3501 = 2.55`
- Suffix: `1 + 4920/3279 = 2.50`

SkipGram 略快 (2.06x vs 2.04x) 是因为每步提交更多 draft token，GPU 一次 forward 验证更多 → 减少了 forward 次数。但代价是每次 forward 的计算量更大。

## 六、Benchmark 结果 (spec=5, 20 samples)

| 指标 | SkipGram | Suffix | Trie-3g | Hash |
|------|----------|--------|---------|------|
| Speedup | **2.06x** | 2.04x | 1.95x | 1.86x |
| Accept Rate | 31.0% | 54.2% | 55.7% | 43.3% |
| Mean Accepted Len | 2.55 | 2.50 | 3.67 | 3.03 |
| num_drafts | 3501 | 3279 | 1818 | 2290 |
| draft_tokens | 17497 | 9080 | 8732 | 10743 |
| accepted_tokens | 5430 | 4920 | 4863 | 4652 |
| Waste (rejected) | 12067 | 4160 | 3869 | 6091 |

## 七、适用场景与局限

### 适用
- **单请求低延迟场景**：GPU 空闲时"多猜多验"，总比不猜好
- **prompt 较短或重复性强的文本**：skip-gram 表的统计更准确

### 不适用
- **高并发批处理**：大量 rejected draft token 浪费 KV cache 和 GPU 计算
- **长上下文场景**：skip-gram 表膨胀（3.5x），内存占用显著
- **高精度要求**：31% accept rate 意味着近 70% 的 draft 计算被浪费
