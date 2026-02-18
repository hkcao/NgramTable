# NgramTable — vLLM N-gram Speculative Decoding Proposer

基于哈希表的 N-gram 投机解码 Proposer，用于替代 vLLM 原生 KMP 方案，提供 O(1) 查询复杂度和跨会话频率积累能力。

## 特性

- **KMP 模式** — 原始 vLLM KMP 最长匹配算法，Numba JIT 加速
- **Per-request 模式** — 每请求独立哈希表，增量更新，频率驱动的 best-next-token 选择
- **Shared 模式** — 全局共享哈希表，跨请求/会话积累 n-gram 频率统计，异步持久化到磁盘

## 三种模式对比

| 特性 | KMP | Per-request | Shared |
|------|-----|-------------|--------|
| 查询复杂度 | O(n) | O(1) | O(1) |
| 频率统计 | 无（最早匹配） | 请求内 | 跨请求积累 |
| 状态生命周期 | 无状态 | 请求级 | 进程级 + 磁盘持久化 |
| 内存占用 | 低 | 中（per-request 分配） | 固定（共享表） |
| 适用场景 | 通用基线 | 长上下文单请求 | 高并发、重复模式多 |

## 环境变量

### 模式选择

```bash
# 三选一，默认 kmp
export VLLM_NGRAM_MODE=kmp          # 原始 KMP 算法
export VLLM_NGRAM_MODE=per_request  # 每请求独立哈希表
export VLLM_NGRAM_MODE=shared       # 共享持久化哈希表
```

向后兼容：若 `VLLM_NGRAM_MODE` 未设置，回退检查旧环境变量 `VLLM_NGRAM_USE_HASH_TABLE=1`（等价于 `per_request`）。

### Shared 模式专用

```bash
# 共享表槽位数（2 的幂），默认 1048576（1M）
export VLLM_NGRAM_SHARED_TABLE_SIZE=1048576

# 持久化目录，默认 ngram_hash
export VLLM_NGRAM_SHARED_DIR=ngram_hash

# 后台刷盘间隔（秒），默认 60
export VLLM_NGRAM_FLUSH_INTERVAL=60
```

## 架构设计

### 哈希表双表结构

每个 n-gram 长度维护两张哈希表：

1. **freq 表** — 统计 `(ngram, next_token) → count` 频率
2. **lut 表** — 存储 `ngram → best_next_token` 查找映射（频率最高的 token）

两表使用开放地址法 + 线性探测解决哈希冲突，共享基础 ngram 哈希值以减少重复计算。

### Shared 模式持久化

```
ngram_hash/
├── metadata.json    # 版本、min_n、max_n、table_size、时间戳
├── n2.npz           # n=2 的频率表和查找表数组
├── n3.npz           # n=3
├── n4.npz           # n=4
└── ...
```

- 使用 `np.savez_compressed` 存储，压缩率好，无额外依赖
- 原子写入：先写 `.tmp` 文件再 `rename`，避免写入中断导致数据损坏
- 后台 daemon 线程定期刷盘，持锁仅做数组 `.copy()`（微秒级），I/O 在锁外完成
- 进程退出时通过 `atexit` 注册同步刷盘

### 负载因子监控

- Per-request 模式：负载因子 > 60% 时触发全量重建扩容
- Shared 模式：负载因子 > 80% 时输出 warning 日志，提示增大 `VLLM_NGRAM_SHARED_TABLE_SIZE`

## 文件说明

| 文件 | 说明 |
|------|------|
| `ngram_proposer.py` | 核心实现，含 KMP / Per-request / Shared 三种模式 |
| `test_numba_hash.py` | 哈希表正确性单元测试（对比 Python dict 参考实现） |
| `test_hash_vs_kmp.py` | 哈希表 vs KMP 方案的效果对比测试 |
| `test_single_seq.py` | 单序列场景测试 |
| `requirements.txt` | Python 依赖 |

## 测试

```bash
# 运行哈希表正确性测试（需要 vllm 环境）
python test_numba_hash.py

# 运行哈希表 vs KMP 对比测试
python test_hash_vs_kmp.py

# 单序列测试
python test_single_seq.py
```

> 注：测试需要在支持 CUDA 的 Linux 环境中运行（vLLM 依赖）。

## 依赖

- vLLM >= 0.6.0
- PyTorch >= 2.0.0
- NumPy
- Numba
