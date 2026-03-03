# Lookahead Decoding 方案详解

> 参考实现：[PainlessInferenceAcceleration / lookahead_cache.py](https://github.com/alipay/PainlessInferenceAcceleration/blob/main/lookahead/lookahead/common/lookahead_cache.py)

---

## 一、方案设计思想

### 1.1 标准自回归解码的瓶颈

标准自回归解码每次只生成一个 token，每生成 N 个 token 就需要 N 次完整的 Transformer forward pass。推理延迟主要由 **forward pass 次数** 决定，而非单次 forward pass 的计算量（因为 GPU/NPU 通常有足够的并行算力）。

```
标准解码：
  step 1: context → [T1]        (1 次 forward)
  step 2: context + T1 → [T2]   (1 次 forward)
  step 3: context + T1T2 → [T3] (1 次 forward)
  ...
  生成 N 个 token = N 次 forward
```

### 1.2 投机解码（Speculative Decoding）的核心思路

投机解码将生成过程拆为两个阶段：

1. **Propose（草稿阶段）**：用低成本方式快速提出 K 个候选 token（草稿）
2. **Verify（验证阶段）**：用目标模型做一次 forward pass，同时验证所有 K 个草稿 token

```
投机解码：
  Propose: n-gram → draft = [T1, T2, T3, T4, T5]  (近乎零成本)
  Verify:  1 次 forward pass 验证全部 5 个草稿
  结果:    若前 3 个被接受 → 本步骤产出 4 个 token (3 accepted + 1 bonus)
```

关键性质：**Verify 的正确性由目标模型保证，最终输出质量与纯自回归完全一致。**

### 1.3 Lookahead 的额外创新：多分支并行验证

普通 n-gram 投机解码（如我们的 HashTable/Trie 方案）每步只提出**一条线性草稿链**：

```
单链草稿: [T1] → [T2] → [T3] → [T4] → [T5]
```

Lookahead 的核心创新是：通过特殊构造的 **Attention Mask**，在**一次 forward pass** 中同时验证**多条分支**：

```
多分支草稿（树形结构）:
         [T1]
        /    \
      [T2]   [T3]
      /         \
    [T4]        [T5]
```

只要 Attention Mask 正确编码各分支的因果依赖关系，Transformer 就能一次处理整棵草稿树，大幅提升每次 forward pass 平均接受的 token 数。

### 1.4 跨请求 n-gram 积累（持久化记忆）

与 per-request 方案不同，LookaheadCache 支持跨请求的 n-gram 频率积累：

- **Input mode**（`idx >= 0`）：记录当前请求 / 特定客户端的 n-gram 频率，请求结束后清零
- **Output mode**（`idx = -1`）：记录所有请求的历史输出 n-gram，永久累积，形成"全局记忆"
- 查询时可通过 `output_weight` 混合两种频率，动态平衡"当前语境"与"历史经验"

---

## 二、方案实现逻辑

### 2.1 整体数据结构

```
LookaheadCache
│
├── mem: Dict[token_id → Tree]     # 以每个 token 为根的 Trie 森林
│
└── Tree (per root token)
    │
    ├── token_id: int              # 本树的根 token
    ├── n_node: int                # 当前节点总数（用于内存控制）
    └── nodes: Dict[token → Node]  # 根节点的子节点字典
        │
        └── Node
            ├── children: Dict[token → Node]   # 子节点
            └── freqs: Dict[idx → float]       # 频率统计
                    ├── -1   → output_freq      # 跨请求累积频率
                    └── idx  → input_freq       # 当前请求频率
```

**三层嵌套关系**：

```
LookaheadCache.mem[root_token] = Tree
Tree.nodes[child_token]        = Node
Node.children[grandchild]      = Node
...（递归到 branch_length 深度）
```

### 2.2 写入：`put()` 逻辑

```python
def put(self, token_ids, branch_length=8, mode='output', idx=0):
    for i in range(len(token_ids) - 1):
        root_token = token_ids[i]
        path = token_ids[i+1 : i+branch_length+1]   # 长度最多 branch_length
        tree = mem.get(root_token) or Tree(root_token)
        tree.put(path, mode=mode, idx=idx)
```

对 `token_ids` 中**每个位置 i**：
- 以 `token_ids[i]` 为该 Tree 的根 key
- 将其后 `branch_length` 个 token 作为一条路径，插入该 Tree 中
- 路径上每个 Node 的频率 `freqs[idx]` 累加 1

**结构示意**（`branch_length=3`，context = `[A, B, C, D, B, C, E]`）：

```
i=0: mem[A].put([B, C, D])
i=1: mem[B].put([C, D, B])
i=2: mem[C].put([D, B, C])
i=3: mem[D].put([B, C, E])
i=4: mem[B].put([C, E])     ← B 的树中 C 节点新增子节点 E
i=5: mem[C].put([E])        ← C 的树中新增子节点 E
```

### 2.3 Tree 的真实结构：`token_id` 只是标签

理解查询逻辑之前，需要先明确 Tree 的结构：

```python
class Tree():
    def __init__(self, token_id, ...):
        self.token_id = token_id  # 仅仅是索引标签，不是一个节点
        self.nodes = {}           # 这才是第一层实际内容
```

`self.token_id` **不是树中的一个节点**，只是这棵树在 `mem` 字典里的 key 标记。**真正第一层**是 `self.nodes`，包含历史上所有紧跟在该 token 之后出现过的 token：

```
mem = {
  B: Tree(
       token_id = B,          ← 仅作索引，不存数据
       nodes = {              ← 第一层 = B 之后出现过的所有 token
         C: Node(freq=2, children={D: ..., E: ...}),
         X: Node(freq=1, ...),   ← 如果历史上 B 后面也出现过 X
         ...
       }
     )
}
```

因此，整体结构应理解为：

```
mem[token]  →  "token 之后所有观测到的延续序列" 构成的 Trie
               第一层 nodes = 紧接着出现过的所有 token（无任何约束）
               第二层      = 第一层某 token 后面紧接着出现过的所有 token
               ...（最深 branch_length 层）
```

`self.token_id` 唯一的实际用途是在 `tree.get()` 的 fallback 里作为返回候选列表的占位锚点：

```python
ids = [match_token_id or self.token_id]  # 仅用于构造返回值
```

### 2.4 查询：`hier_get()` 的滑动窗口降级机制

```python
def hier_get(self, token_ids, decoding_length=64, branch_length=8, ...):
    decoding_ids = None
    for i, t in enumerate(token_ids):
        tree = mem.get(t)
        if tree:
            ids = token_ids[i + 1:]        # ← 关键：后缀随 i 增大而缩短
            decoding_ids, mask, sizes = tree.get(ids, ...)
            if len(decoding_ids) >= branch_length:
                break                       # 候选够多才提前退出
    if decoding_ids is None:
        decoding_ids = token_ids[-1:]       # 彻底 fallback
    return decoding_ids, mask, sizes
```

**核心设计**：遍历 `token_ids` 中每一个 token 作为潜在的查询根，每次迭代的匹配路径越来越短，形成自动降级：

| i | 用作查询根的 token | 传入 tree 的匹配路径 | 语义 |
|---|---|---|---|
| 0 | `token_ids[0]` | `token_ids[1:]`（最长） | 最精准：完整上下文匹配 |
| 1 | `token_ids[1]` | `token_ids[2:]`（次长） | 降级一步 |
| … | … | … | 持续降级 |
| n-1 | `token_ids[-1]` | `[]`（空） | 最粗：只看最后一个 token 的全部历史后继 |

**每次结果覆盖上一次**，只有 `len(ids) >= branch_length` 才提前 break；否则一路跑完，取**最后一次**有效结果。

**完整降级链条示例**（查询后缀 `[A, B, C, D]`，branch_length=3）：

```
i=0, root=A, 路径=[B,C,D]
  → _match: A树存在，但B不在A树中 → nodes={} → fallback → 返回 [D]（单token）
  → len=1 < 3，继续

i=1, root=B, 路径=[C,D]
  → _match: B→C→D 匹配，D的children={E}
  → 返回 [D, E]
  → len=2 < 3，继续

i=2, root=C, 路径=[D]
  → _match: C→D 匹配，D的children={B, F}
  → DFS 展开：返回 [D, B, F]
  → len=3 >= 3 → break ✓

最终候选：[D, B, F]
```

**极端 fallback**（只有最后一个 token 在 mem 中）：

```
i=n-1, root=token_ids[-1], 路径=[]
  → _match([]): 直接返回 self.nodes（第一层全部子节点）
  → DFS 收集该 token 所有历史后继分支
  → 退化为纯 1-gram 查表
```

### 2.5 Tree 内部：`_match` 的失配行为

```python
def _match(self, token_ids, ...):
    nodes = self.nodes     # 从第一层出发
    token_id = None
    if len(token_ids) == 0:
        return None, nodes # 空路径 → 直接返回第一层全部内容

    for token_id in token_ids:
        node = nodes.get(token_id, None)
        nodes = {}         # 先清空
        if node is None:
            break          # 失配：nodes 保持 {}，停止
        nodes = node.children

    return token_id, nodes  # nodes={} 表示失配
```

失配后 `tree.get()` 检测到 `nodes={}` 触发 fallback：

```python
if len(nodes) == 0:
    token_id = token_ids[-1] if token_ids else self.token_id
    return [token_id], np.ones((1,1)), [0,0]   # 只返回单个 token
```

### 2.6 Tree 内部：前缀匹配 + DFS 收集（概述）

```
_match(suffix):   沿路径向下走，找到最深匹配节点；失配时返回空 nodes
_ravel(nodes):    从匹配节点出发，DFS 收集所有后继分支
                  按 mix_freq = (1-w)*input_freq + w*output_freq 排序
```

### 2.7 Attention Mask 构造

`_ravel` 在 DFS 收集 token 时，同步构建 Attention Mask：

```python
# 父节点 pid 对应的掩码行复制给子节点 rid，并将 rid 自身置 1
mask[rid] = mask[pid]
mask[rid, rid] = 1
```

这确保了每个 token 只能 attend 到自己的祖先路径（causal 依赖），不同分支之间互相不可见。

### 2.8 内存管理：`squeeze()`

当 Tree 节点数超过 `max_node` 或 `max_output_node` 时，触发压缩：

```python
def _squeeze(self, nodes):
    for token, node in nodes.items():
        if node.freqs[-1] > 1.0:
            node.freqs[-1] *= 0.5    # 频率减半（指数衰减）
        else:
            nodes.pop(token)         # 低频节点直接删除
```

**效果**：频率高的路径被保留，低频路径被淘汰，形成类似 LRU 的自适应压缩。

---

## 三、完整例子：Trie 的构建、更新与查找

### 3.1 初始设置

假设：
- `branch_length = 3`
- `mode = 'output'`（idx 固定为 -1）
- 用字母代替 token id：`A=1, B=2, C=3, D=4, E=5, F=6`

---

### 3.2 阶段一：构建（第一次 put）

**输入序列**：`[A, B, C, D, B, C, E]`，长度 7，循环 `i = 0..5`

| i | root_token | path（后 3 个） | 操作 |
|---|---|---|---|
| 0 | A | [B, C, D] | 创建 `mem[A]`，插入路径 B→C→D |
| 1 | B | [C, D, B] | 创建 `mem[B]`，插入路径 C→D→B |
| 2 | C | [D, B, C] | 创建 `mem[C]`，插入路径 D→B→C |
| 3 | D | [B, C, E] | 创建 `mem[D]`，插入路径 B→C→E |
| 4 | B | [C, E]   | `mem[B]` 已存在，C 节点下新增子节点 E |
| 5 | C | [E]       | `mem[C]` 已存在，根节点下新增子节点 E |

**构建后的树结构**：

```
mem[A] (Tree, root=A):
  └── B (freq[-1]=1)
      └── C (freq[-1]=1)
          └── D (freq[-1]=1)

mem[B] (Tree, root=B):
  └── C (freq[-1]=2)       ← i=1 和 i=4 各访问一次，频率为 2
      ├── D (freq[-1]=1)
      │   └── B (freq[-1]=1)
      └── E (freq[-1]=1)

mem[C] (Tree, root=C):
  ├── D (freq[-1]=1)
  │   └── B (freq[-1]=1)
  │       └── C (freq[-1]=1)
  └── E (freq[-1]=1)       ← i=5 插入

mem[D] (Tree, root=D):
  └── B (freq[-1]=1)
      └── C (freq[-1]=1)
          └── E (freq[-1]=1)
```

---

### 3.3 阶段二：更新（第二次 put，新序列到来）

**新输入序列**：`[B, C, D, F]`（新一轮对话输出）

| i | root_token | path | 操作 |
|---|---|---|---|
| 0 | B | [C, D, F] | `mem[B].C.D` 已存在，新增子节点 F；`mem[B].C.D.freq[-1]` 累加 1 → 2 |
| 1 | C | [D, F]    | `mem[C].D` 已存在，`mem[C].D.freq[-1]` 累加 1 → 2；D 下新增子节点 F |
| 2 | D | [F]       | `mem[D]` 已存在，根节点下... wait，D 的 root 是 D，path=[F]，即在 D 的树中插入节点 F |

**更新后 `mem[B]` 的变化**：

```
mem[B] (Tree, root=B) — 更新后:
  └── C (freq[-1]=3)         ← 第三次命中（i=1原始+i=4原始+本次i=0）
      ├── D (freq[-1]=2)     ← 命中两次（原始一次 + 本次）
      │   ├── B (freq[-1]=1)
      │   └── F (freq[-1]=1) ← 新增
      └── E (freq[-1]=1)
```

**关键观察**：同一路径被命中次数越多，频率越高，在后续查询时优先级越高。

> **注意**：`mem[B].nodes`（第一层）只包含历史上**紧跟在 B 之后**的 token。上例中 B 之后总是先接 C，所以第一层只有 `C` 一个节点，D 和 E 都在第二层（C 的 children）。第一层的多样性取决于 root token 直接后继的历史多样性，而非所有可能 token。

---

### 3.4 阶段三：查找（`hier_get`）

**当前 context 后缀**（查询序列）：`[..., B, C, D]`（即最近 3 个 token）

调用 `hier_get(token_ids=[B, C, D], decoding_length=5, branch_length=3)`

**执行过程**：

```
遍历 token_ids:

  i=0, t=B:
    tree = mem[B]  ✓
    suffix = [C, D]

    _match([C, D]) in mem[B]:
      根节点 children 有 C → 进入 C 节点
      C 节点 children 有 D → 进入 D 节点
      D 的 children = {B: ..., F: ...}
      返回 match_token=D, 当前层 nodes = {B: Node(freq=1), F: Node(freq=1)}

    _ravel(nodes={B,F}, ...):
      排序（均 freq=1，假设 B 先）:
        ids = [D, B]     → B 无子节点（在此路径下已到末端），停止
        ids = [D, B, F]  → 回溯，添加 F
        (注：实际上 D 下的 B 还有子节点，见上面树结构)

      完整 DFS 展开:
        ids    = [D,  B,  F]         (D 是匹配到的最深 token，B 和 F 是分支)
        实际上 B 下还有子节点，但已达 branch_length=3 深度，停止

      len(ids) = 3 < branch_length=3... 取决于具体阈值；继续看 i=1

  i=1, t=C:
    tree = mem[C]  ✓
    suffix = [D]

    _match([D]) in mem[C]:
      C 树根节点 children = {D: ..., E: ...}
      有 D → 进入 D 节点
      D 节点 children = {B: ..., F: ...}（更新后）
      返回 match_token=D, 当前层 nodes = {B: ..., F: ...}

    _ravel(nodes={B, F}, ...):
      ids = [D, B, F]  或  [D, F, B]  取决于频率排序
      此例中 B 和 F 频率相同（均=1），
      但 B 节点还有子节点 C（freq=1），可继续扩展

      DFS 展开:
        ids[0] = D  (match anchor)
        ids[1] = B  (freq=1, 优先)
        ids[2] = C  (B 的子节点, branch_length 还剩 1 层)
        ids[3] = F  (回溯，D 的另一个子节点)

      ids = [D, B, C, F], len=4 ≥ branch_length=3 → 足够，停止
```

**最终返回**：

```
decoding_ids = [D, B, C, F]
```

---

### 3.5 Attention Mask 构造详解

对应 `ids = [D, B, C, F]`（索引 0,1,2,3）：

**因果依赖关系**（谁是谁的祖先）：

```
树形结构:
  [D]           ← 索引 0（匹配锚点）
   ├── [B]      ← 索引 1（D 的子节点）
   │    └── [C] ← 索引 2（B 的子节点）
   └── [F]      ← 索引 3（D 的子节点，与 B 同级）
```

**Attention Mask**（1=可以 attend，0=不可以）：

```
        D  B  C  F
   D  [ 1  0  0  0 ]   D 只能看自己
   B  [ 1  1  0  0 ]   B 能看 D（父）和自己
   C  [ 1  1  1  0 ]   C 能看 D（祖）、B（父）和自己
   F  [ 1  0  0  1 ]   F 能看 D（父）和自己，不能看 B 或 C（兄弟分支）
```

**关键**：F 不能 attend 到 B/C（它们是平行分支），确保了不同假设路径之间的因果隔离。

**一次 forward pass 验证所有分支**：

Transformer 用这个 mask 做一次完整的前向计算：

```
输入:  context + [D, B, C, F]
Mask:  上面的 4×4 树形因果矩阵（附加在 context 的全因果 mask 之上）

输出（同时得到）:
  - 位置 D 的预测 → 验证 D 是否被接受
  - 位置 B 的预测 → 基于 context+D，验证 B
  - 位置 C 的预测 → 基于 context+D+B，验证 C
  - 位置 F 的预测 → 基于 context+D，验证 F（与 B/C 无关）
```

若模型接受路径 D→B→C，则本步骤一次 forward 贡献了 **3 个 token**（D,B,C）+ 1 个 bonus token，等效于普通解码的 4 步。

---

## 四、与我们的 HashTable/Trie 方案对比总结

| 维度 | PainlessIA LookaheadCache | 我们的 HashTable/Trie |
|---|---|---|
| **数据结构** | 每个 root token 一棵独立 Trie | 单一全局 Trie 或 n-gram 哈希表 |
| **草稿形式** | 多分支树（最多 `decoding_length` 个节点） | 单条贪心链（最多 k 个 token） |
| **验证方式** | 一次 forward + 树形 Attention Mask | 一次 forward + 线性 causal mask |
| **每步最大收益** | 高（多分支中任意一条被接受即可） | 低（只有唯一的草稿链） |
| **n-gram 存储粒度** | 完整 `branch_length` 路径（历史观测序列） | 逐步单 token 预测（argmax 链） |
| **跨请求记忆** | 支持（output mode 永久积累） | 不支持（per-request 重建） |
| **内存管理** | 有（max_node + 指数衰减压缩） | 无 |
| **复杂度** | 较高（mask 构造、多客户端频率） | 简单（纯 Python dict/trie） |

**算法等价性**（你之前的判断）：在 top-1 单链、同一 context、相同窗口大小条件下，两者产出相同草稿。差异完全来自多分支 + Attention Mask 这一层。
