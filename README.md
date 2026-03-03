# NgramTable — N-gram 投机解码实验平台

本项目提供两套 N-gram 投机解码（Speculative Decoding）实现，可独立运行：

| 方案 | 分支 | 依赖 | 平台 |
|------|------|------|------|
| **vLLM 版**（原始） | `develop_cc` | vLLM + CUDA | Linux |
| **Transformer 版**（本分支） | `develop_cc_transformer` | HuggingFace transformers | macOS / Linux / CPU |

---

## Transformer 版（当前分支：`develop_cc_transformer`）

无需 vLLM，基于 HuggingFace `transformers` 实现，支持 macOS MPS、CUDA 和 CPU。

### 架构概览

```
speculative/
├── __init__.py           # 懒加载导出
├── engine.py             # SpeculativeEngine：propose → verify → accept 主循环
├── verifier.py           # TransformerVerifier（Qwen2.5-0.5B，greedy，temperature=0）
├── metrics.py            # MetricsTracker：命中率、接受率、加速比
└── proposers/
    ├── base.py           # BaseProposer 抽象接口
    ├── kmp_proposer.py   # KMP 最长后缀匹配
    ├── hash_proposer.py  # 频率哈希表（O(1) 查询）
    └── trie_proposer.py  # Token Trie（对齐 PainlessInferenceAcceleration）
```

### 三种草稿提案器（Proposer）

| 提案器 | 算法 | 查询复杂度 | 状态 |
|--------|------|-----------|------|
| **KMP** | 最长后缀匹配扫描 | O(ctx × max_n) | 无状态 |
| **HashTable** | n-gram 频率哈希表 | O(1) | 增量更新 |
| **Trie** | Token 前缀树（对齐 PainlessIA） | O(max_n) | 增量更新 |

**Trie 设计参考**：[PainlessInferenceAcceleration / lookahead_cache.py](https://github.com/alipay/PainlessInferenceAcceleration/blob/main/lookahead/lookahead/common/lookahead_cache.py)

- `mem[t]` = 以 token `t` 为根的子 Trie（等价于 `LookaheadCache.mem[t] = Tree(t)`）
- 插入：对每个位置 `i`，将 `context[i+1:i+max_n+1]` 作为路径插入 `mem[context[i]]`
- 查询：从 `max_n` 到 `min_n` 依次尝试，取上下文最后 n 个 token 做前缀匹配

### 验证器（Verifier）

`TransformerVerifier` 封装 Qwen2.5-0.5B-Instruct：

- **Greedy 验证**（temperature=0）：`argmax` 接受/拒绝，结果完全可复现
- **KV 缓存管道**：`init_kv_cache()` + `verify_step()` — Context 只编码一次，每步只处理草稿 token
- **简单验证**：`verify()` — 无 KV 缓存，全序列一次 forward（用于正确性测试）
- **设备自动选择**：CUDA → MPS（Apple Silicon）→ CPU

### 快速开始

```bash
# 安装依赖（uv）
uv sync

# 运行单元测试（无需加载模型）
uv run python test_transformer.py -v

# 运行完整 benchmark（需要下载 Qwen2.5-0.5B，约 1GB）
uv run python benchmark_transformer.py \
    --num-samples 20 \
    --proposers kmp hash trie \
    --max-new-tokens 256 \
    --num-speculative-tokens 5
```

### Benchmark 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--verifier-model` | `Qwen/Qwen2.5-0.5B-Instruct` | HuggingFace 模型名 |
| `--proposers` | `kmp hash trie` | 要对比的提案器（可多选） |
| `--num-samples` | `20` | SWE-bench Lite 样本数 |
| `--max-new-tokens` | `256` | 每请求最大生成 token 数 |
| `--num-speculative-tokens` | `5` | 每步草稿 token 数 k |
| `--min-n` / `--max-n` | `2` / `5` | n-gram 窗口范围 |
| `--temperature` | `0.0` | 采样温度（0.0 = greedy） |
| `--no-baseline` | — | 跳过自回归基线（加速测试） |
| `--output` | `results/transformer_benchmark.json` | 结果保存路径 |

### 基准测试结果（参考，10 样本，Qwen2.5-0.5B，MPS）

| 提案器 | 草稿命中率 | Token 接受率 | 平均接受长度 | 理论加速比 |
|--------|-----------|-------------|-------------|----------|
| KMP    | ~56%      | ~56%        | ~2.16       | ~2.16x   |
| Hash   | ~60%      | ~60%        | ~2.21       | ~2.21x   |
| Trie   | ~55%+     | ~55%+       | ~2.1+       | ~2.1x+   |

> 实测加速比（墙时钟）需使用 KV 缓存管道（`verify_step`）才公平；以上为理论值。

### 指标说明

- **草稿命中率（draft_hit_rate）**：步骤中所有草稿 token 均被接受的步骤占比
- **Token 接受率（token_acceptance_rate）**：被接受的草稿 token 数 / 总提议 token 数
- **平均接受长度（mean_accepted_length）**：每步平均接受 token 数（含 bonus token）
- **理论加速比**：`mean_accepted_length`，即相对于自回归基线每步平均减少的 forward pass 次数

### 文件说明

| 文件/目录 | 说明 |
|-----------|------|
| `speculative/` | 投机解码框架核心包 |
| `benchmark_transformer.py` | 完整 benchmark CLI |
| `test_transformer.py` | 单元 + 集成测试（24 项，不依赖模型） |
| `pyproject.toml` | uv 项目配置 |
| `lookahead方案.md` | Lookahead Decoding 详细设计文档 |
| `results/` | benchmark 输出（JSON） |

---

## vLLM 版（分支：`develop_cc`）

基于 vLLM 的 N-gram 投机解码，使用 Numba JIT 加速的自定义哈希表替代原生 KMP 方案。

### 三种模式

| 模式 | 环境变量 | 说明 |
|------|---------|------|
| KMP | `VLLM_NGRAM_MODE=kmp` | 原始 vLLM KMP 最长匹配 |
| Per-request | `VLLM_NGRAM_MODE=per_request` | 每请求独立哈希表，增量更新 |
| Shared | `VLLM_NGRAM_MODE=shared` | 全局共享表，跨请求/会话频率积累 |

详见 `develop_cc` 分支 README。

---

## 参考资料

- [PainlessInferenceAcceleration（Alipay）](https://github.com/alipay/PainlessInferenceAcceleration) — Lookahead Decoding 参考实现
- [Speculative Decoding 论文](https://arxiv.org/abs/2211.17192) — Chen et al., 2022
- [SWE-bench Lite](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite) — 评测数据集
- [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) — 默认验证器模型
