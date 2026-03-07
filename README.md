# NGram Hash-Table Speculative Decoding

基于 Hash 表的 draft token 预测方案，替代 vLLM 原有的 KMP 线性扫描方式。

## 核心思路

用空间换时间：将 N-gram 频次统计预计算为 Hash 表，推理时 O(1) 查表链式生成 k 个 draft token，同时在推理过程中增量更新统计信息。

## 数据结构

维护两张表（统一持久化为 `ngramTable` pickle 文件）：

- **FreqTable**：`dict[tuple[int,...], Counter[int]]` — key 为长度 n 的 token 序列（n-gram context），value 为后续各 token 的频次
- **HashTable**：`dict[int, int]` — key 为 `hash(context_tuple)`，value 为 FreqTable 中该 context 频次最高的 token（argmax）。FreqTable 的派生缓存，用于 O(1) 推理查询

## 查询流程

基于输入序列最后 n 个 token，链式查表生成 k 个预测 token：

```
input: [..., A, B, C]  n=3, k=3
hash((A,B,C)) -> E
hash((B,C,E)) -> G
hash((C,E,G)) -> F
output: [E, G, F]
```

查表失败时自动 fallback 到更短的 n-gram。

## 文件说明

```
ngram/
├── ngram_proposer.py          # 核心实现（symlink 到 vllm site-packages）
├── test_batch_acceptance.py   # batch 模式测试：命中率 & 接受率
├── test_single_repro.py       # 单请求可复现性验证
├── test_ngram.py              # 单请求延迟对比测试
├── pyproject.toml
└── README.md
```

## 环境

venv 位于 `../venv/`，已安装 vllm 0.15.1。`ngram_proposer.py` 通过 symlink 链入 vllm：

```
Agent/venv/lib/.../vllm/v1/spec_decode/ngram_proposer.py -> Agent/ngram/ngram_proposer.py
```

直接编辑 `ngram/ngram_proposer.py` 即可生效，无需复制。

## 模式切换

通过环境变量 `VLLM_NGRAM_USE_HASH` 控制：

| 值 | 模式 | 说明 |
|---|---|---|
| `1`（默认） | Hash 表 | 频次统计 + O(1) 查表 |
| `0` | KMP | vLLM 原始实现，代码完整保留 |

两条路径在 `propose()` 内部以 if/else 分支区分，原有 KMP 代码不做任何改动。

## 持久化

- 设置 `VLLM_NGRAM_TABLE_PATH=/path/to/ngramTable` 启用持久化
- 格式：pickle，存储 `{"freq_table": FreqTable, "hash_table": HashTable}`
- 每 100 步异步刷盘，进程退出时同步写入

## 运行测试

```bash
cd ngram

# batch 模式：50 samples，跑 2 轮验证可复现性
.venv/bin/python test_batch_acceptance.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --num-samples 50 --max-tokens 512 --gpu-mem 0.8 --num-runs 2

# 单请求可复现性验证
.venv/bin/python test_single_repro.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --num-samples 20 --max-tokens 512 --gpu-mem 0.8

# 单请求延迟对比（含 baseline）
.venv/bin/python test_ngram.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --num-samples 5 --max-tokens 256 --gpu-mem 0.8
```

## 测试结果

Qwen2.5-3B-Instruct, 单请求模式, temp=0, 20 samples:

| Mode | Accept% | MeanLen | HitRate | Drafts | Accepted |
|---|---|---|---|---|---|
| KMP spec=5 n=2-5 | 50.58% | 3.52 | 0.674 | 2044 | 5149 |
| Hash spec=5 n=2-5 | 39.83% | 2.99 | 0.567 | 2612 | 5201 |

- 两轮运行 delta=0.00，完全可复现
- KMP 精确匹配后续序列，acceptance rate 更高
- Hash 基于全局频次 argmax 预测，命中精度较低但总能给出预测
