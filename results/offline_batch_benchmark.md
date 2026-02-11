# vLLM Ngram 投机推理 — 离线批量推理基准测试报告

- 日期: 2026-02-08
- vLLM 版本: 0.15.1 (V1 Engine)
- 模型: Qwen/Qwen2.5-0.5B-Instruct (bfloat16)
- GPU: NVIDIA GeForce RTX 4070 Super 16GB
- 数据集: SWE-bench Lite (20 条 / 共 300 条)
- 最大生成长度: 512 tokens
- 采样策略: greedy (temperature=0.0)
- 预热: 2 轮 (3 条 prompt)

## 性能对比

| 模式 | 吞吐量 (tok/s) | 耗时 (s) | 加速比 | 命中率 | 平均接受长度 |
|------|---------------|---------|--------|-------|------------|
| baseline (无投机) | 6184.28 | 1.609 | 1.00x | - | - |
| ngram spec=3 lookup=2-3 | 1362.89 | 6.962 | 0.22x | 62.1% | 2.86 |
| ngram spec=5 lookup=2-5 | 3088.80 | 3.315 | 0.50x | 61.6% | 4.07 |
| ngram spec=8 lookup=3-7 | 5049.95 | 1.948 | 0.82x | 65.2% | 6.17 |

## Token 统计

| 模式 | 输入 tokens | 输出 tokens | 每条延迟 (s) |
|------|-----------|-----------|------------|
| baseline | 9224 | 9952 | 0.0805 |
| ngram spec=3 | 9224 | 9488 | 0.3481 |
| ngram spec=5 | 9224 | 10240 | 0.1658 |
| ngram spec=8 | 9224 | 9839 | 0.0974 |

## 命中率详情

### ngram spec=3 lookup=2-3
- draft 轮次: 2757
- draft tokens: 8269 | accepted: 5133
- 命中率: 62.1% | 平均接受长度: 2.86

| 位置 | pos0 | pos1 | pos2 |
|------|------|------|------|
| 命中率 | 0.710 | 0.610 | 0.542 |

### ngram spec=5 lookup=2-5
- draft 轮次: 2144
- draft tokens: 10705 | accepted: 6590
- 命中率: 61.6% | 平均接受长度: 4.07

| 位置 | pos0 | pos1 | pos2 | pos3 | pos4 |
|------|------|------|------|------|------|
| 命中率 | 0.747 | 0.646 | 0.585 | 0.561 | 0.534 |

### ngram spec=8 lookup=3-7
- draft 轮次: 1305
- draft tokens: 10350 | accepted: 6752
- 命中率: 65.2% | 平均接受长度: 6.17

| 位置 | pos0 | pos1 | pos2 | pos3 | pos4 | pos5 | pos6 | pos7 |
|------|------|------|------|------|------|------|------|------|
| 命中率 | 0.837 | 0.728 | 0.661 | 0.631 | 0.602 | 0.582 | 0.572 | 0.561 |

## 根因分析

ngram 投机推理在批量离线场景下反而变慢，根因如下：

1. **异步调度被禁用 (主因)**
   vLLM 日志: `WARNING: Async scheduling not supported with ngram-based speculative decoding and will be disabled.`
   baseline 使用异步调度 (async scheduling)，ngram 模式被强制退回同步调度，调度管线效率大幅下降。

2. **小模型推理极快**
   Qwen2.5-0.5B 单 token 推理已极快 (~6k tok/s)，投机验证的额外前向传播开销相对占比大。

3. **spec 窗口越大，amortization 越好**
   spec=8 最接近 baseline (0.82x)，因为每轮验证平均接受 6.17 tokens，更好地摊薄了同步调度开销。

## 结论

- ngram 命中率本身不错 (61-65%)，不是算法问题
- 性能瓶颈在 vLLM 0.15.1 框架层面：ngram spec decode 不支持 async scheduling
- 对于小模型 + 离线批量场景，建议不启用 ngram 投机推理
