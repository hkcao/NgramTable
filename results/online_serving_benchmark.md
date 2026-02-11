# vLLM Ngram 投机推理 — 在线 Serving 基准测试报告

- 日期: 2026-02-08
- vLLM 版本: 0.15.1 (V1 Engine)
- 模型: Qwen/Qwen2.5-0.5B-Instruct (bfloat16)
- GPU: NVIDIA GeForce RTX 4070 Super 16GB
- 数据集: SWE-bench Lite (10 条)
- 最大生成长度: 512 tokens
- 采样策略: greedy (temperature=0.0)
- 测试方式: vllm serve (OpenAI API) + 串行逐条 streaming 请求
- 预热: 2 条请求

## 延迟对比

| 模式 | Avg TTFT | P50 TTFT | P99 TTFT | Avg 延迟 | P50 延迟 | P99 延迟 |
|------|---------|---------|---------|---------|---------|---------|
| baseline (无投机) | 0.014s | 0.014s | 0.018s | 1.172s | 1.297s | 1.301s |
| ngram spec=5 lookup=2-5 | 0.015s | 0.017s | 0.019s | 1.066s | 1.053s | 2.050s |
| ngram spec=8 lookup=3-7 | 0.016s | 0.017s | 0.022s | 1.231s | 1.198s | 1.882s |

## 延迟加速比

| 模式 | 延迟加速比 | TTFT 加速比 |
|------|----------|-----------|
| ngram spec=5 lookup=2-5 | 1.10x | 0.93x |
| ngram spec=8 lookup=3-7 | 0.95x | 0.88x |

## 命中率 (Acceptance Rate)

| 模式 | 命中率 | 平均接受长度 | draft 轮次 | draft tokens | accepted tokens |
|------|-------|------------|-----------|-------------|----------------|
| ngram spec=5 lookup=2-5 | 50.0% | 3.49 | 791 | 3944 | 1972 |
| ngram spec=8 lookup=3-7 | 48.1% | 4.81 | 573 | 4539 | 2185 |

## 各位置命中率

### ngram spec=5 lookup=2-5

| 位置 | pos0 | pos1 | pos2 | pos3 | pos4 |
|------|------|------|------|------|------|
| 命中率 | 0.669 | 0.556 | 0.484 | 0.440 | 0.344 |

### ngram spec=8 lookup=3-7

| 位置 | pos0 | pos1 | pos2 | pos3 | pos4 | pos5 | pos6 | pos7 |
|------|------|------|------|------|------|------|------|------|
| 命中率 | 0.700 | 0.586 | 0.541 | 0.464 | 0.415 | 0.389 | 0.377 | 0.340 |

## 分析

### 1. 在线 serving vs 离线批量对比

| 指标 | 离线批量 (spec=8) | 在线 serving (spec=8) |
|------|------------------|---------------------|
| 加速比 | 0.82x (变慢) | 0.95x (接近持平) |
| 命中率 | 65.2% | 48.1% |
| 异步调度 | 被禁用 (主要瓶颈) | N/A (serving 模式不同) |

### 2. 关键发现

- **TTFT 基本持平**: ~14-16ms，ngram 投机对首 token 延迟无显著影响
- **spec=5 有微弱加速 (1.10x)**: 在 serving 模式下 ngram 投机略有帮助，但增益不大
- **spec=8 反而略慢 (0.95x)**: 更大的投机窗口在命中率下降时反而增加了开销
- **命中率低于离线模式**: serving 模式下 48-50%（离线 61-65%），可能与请求调度差异有关
- **P99 延迟波动大**: ngram 模式的 P99 延迟有较大尾延迟 (spec=5 的 P99=2.05s vs baseline P99=1.30s)

### 3. 根因

对于 Qwen-0.5B 这样的小模型：
- 单 token 推理极快（~400 tok/s per request），投机验证开销相对较大
- SWE-bench 提示词的代码/文本混合内容重复性不高，ngram 匹配质量有限
- ngram 投机推理更适合大模型（7B+），单 token 推理延迟高时收益更明显

## 注意事项

streaming 模式下 SSE 分块可能导致客户端侧 token 计数不完全准确（spec decode 可能在单次 SSE 事件中发送多个 token），延迟和 prometheus 命中率指标不受影响。
