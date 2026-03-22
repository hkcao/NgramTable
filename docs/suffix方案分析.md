# Suffix Decoding 方案深度分析

> 基于 arctic-inference C++ 源码 (`suffix_tree.h` + `suffix_tree.cc`) 的逐行分析
> Python 复现：`py_suffix_tree.py`（100% 输出一致，含 float32 精度匹配）

---

## 一、整体架构

### 1.1 双树设计

```
SuffixDecodingCache
├── _global_tree: SuffixTree      # 全局共享，存储所有已完成请求的 response
├── _local_trees: {req_id: SuffixTree}  # 每个活跃请求独立，存储 prompt + 当前 response
├── _req_to_seq_id: {req_id: seq_id}    # 请求 ID → 全局树的序列 ID
└── max_tree_depth: int = 24            # 后缀树最大深度（默认值）
```

**设计思路**：
- **Local tree**：当前请求的 prompt 是最强的预测信号（自我重复模式），每个请求独立一棵树
- **Global tree**：历史请求的 response 跨请求共享，利用不同请求间的模式复用
- **Speculation 时两棵树各自推测，取 score 更高的结果**

### 1.2 请求生命周期

```
start_request(req_id, prompt)
  → local_tree.extend(0, prompt)          # prompt 进入 local 树
  → 分配 global seq_id                    # 预留 global 树位置

add_active_response(req_id, sampled_ids)   # 每步调用
  → local_tree.extend(0, sampled_ids)     # 新 token 进入 local 树
  → global_tree.extend(seq_id, sampled_ids) # 新 token 也进入 global 树

speculate(req_id, context, ...)
  → draft1 = local_tree.speculate(context)
  → draft2 = global_tree.speculate(context)
  → return score 更高的那个

stop_request(req_id)
  → 删除 local 树
  → global 树数据保留（供后续请求使用）

evict_cached_response(req_id)
  → 从 global 树中移除该请求的数据
  → FIFO 驱逐，max_cached_requests 控制上限
```

---

## 二、核心数据结构 — 压缩后缀树

### 2.1 Node 结构

```cpp
struct Node {
    int64_t count;      // 经过此节点的后缀数（= 该子串在文本中出现的次数）
    int token;          // 首 token ID（边的第一个字符）
    int length;         // 路径压缩长度（边可代表多个 token）
    int ref_seq;        // 引用序列 ID（路径压缩用）
    int ref_idx;        // 引用起始位置（路径压缩用）

    Map<int, Node*> children;   // 子节点（key = 子节点的首 token）
    Map<int, int> endpoints;    // 哪些后缀在此节点结束 (seq_id → end_idx)

    Node* parent;               // 父节点
    Node* head_child;           // count 最大的子节点（兄弟链表头）
    Node* tail_child;           // count 最小的子节点（兄弟链表尾）
    Node* prev_sibling;         // 前一个兄弟
    Node* next_sibling;         // 后一个兄弟
    Group* group;               // 所属 count 分组
};
```

### 2.2 路径压缩

普通后缀 Trie 每个节点只表示一个 token，空间 O(n²)。压缩后缀树的关键优化：**一个节点可以代表多个连续 token**（`length > 1`），通过 `(ref_seq, ref_idx)` 引用原始序列中的数据，无需复制。

```
普通后缀 Trie:              压缩后缀树:
root                        root
├─ A ─ B ─ C ─ D ─ $       ├─ [ABCD$]  (token=A, length=5, ref="ABCD$")
├─ B ─ C ─ D ─ $           ├─ [BCD$]   (token=B, length=4)
├─ C ─ D ─ $               ├─ [CD$]    (token=C, length=2)
└─ D ─ $                   └─ [D$]     (token=D, length=2)

当出现分叉时:
root                        root
├─ A ─ B ─ C ─ D            ├─ [AB] (length=2)
│           └─ G            │   ├─ [CD] (length=2)
└─ ...                      │   └─ [CG] ← 分裂！
                             └─ ...

实际上分裂后:
root ─ [AB] ─ C ─┬─ [D]
                  └─ [G]
    length=2  length=1  length=1
```

**空间复杂度**：O(n) — 节点数至多 2n（每个后缀末尾对应一个叶节点，内部节点 < n）。

### 2.3 兄弟链表与 Group 机制

**目的**：`speculate_path` 需要 O(1) 找到 count 最大的子节点（`head_child`）。

每个父节点的子节点按 **count 降序** 组织为双向链表：

```
head_child ←→ sibling2 ←→ sibling3 ←→ ... ←→ tail_child
count=10      count=7       count=7            count=2
[Group A]     [   Group B   ]                  [Group C]
```

**Group** 将相同 count 的连续兄弟分组。好处：
- `_increment_count(node)` 时，node 需要移动到正确位置。通过 Group 可以 O(1) 找到插入点
- 等 count 的节点内部排序规则：**后提升者排在前面**（`_insert_into_siblings_before(node, group->head)`）

---

## 三、构建流程 — `append(seq_id, token)`

### 3.1 整体框架

每追加一个 token，需要更新所有"活跃后缀"。活跃后缀通过 `active_nodes` 双端队列维护：

```python
def append(seq_id, token):
    # 1. 在队列末尾追加 root（开始一个新后缀）
    active_nodes.append(root)
    root.count += 1

    # 2. 限制深度：队列超过 max_depth 时弹出最旧的
    if len(active_nodes) > max_depth:
        active_nodes.popleft()

    # 3. 将 token 追加到序列
    seq.append(token)

    # 4. 遍历所有活跃节点，逐一向下扩展
    for active_node in active_nodes:
        extend_active_node(active_node, token)
```

**直觉**：每个活跃节点代表一个"正在生长的后缀"。追加 token 时，每个后缀都需要向下走一步。

### 3.2 六种扩展情况

对每个 active_node，检查它是否有 token 对应的子节点 child：

```
                    child 存在？
                   /            \
                  否              是
                 /                \
          count==1              node.count == child.count+1
          且非root？             且非root？
           /    \                /          \
          是     否             是            否
        Case1a  Case1b        Case2         Case3
                             (a/b)         (a/b)
```

#### Case 1a: 无子节点 + 叶节点独占

当前节点是叶节点（count=1），只有当前后缀经过。直接**就地延长**：

```
Before: node[ABC] (length=3, count=1)
Append token D:
After:  node[ABCD] (length=4, count=1)   ← 不创建新节点
```

#### Case 1b: 无子节点 + 非叶或 root

当前节点有多个后缀经过，需要**创建新子节点**分叉：

```
Before: node[AB] (count=3)       ← 3 个后缀经过
Append token X (之前没出现过):
After:  node[AB] (count=3)
          └── child[X] (count=1, length=1)  ← 新节点
```

新子节点插入到兄弟链表**尾部**（`_insert_into_siblings_after(new_child, tail_child)`），因为 count=1 是最小的。

#### Case 2: 有子节点 + 融合条件

条件：`node.count == child.count + 1`（除了当前后缀，所有经过 node 的后缀都走 child）。

意味着：node 只有一个子节点，且当前后缀是唯一不走 child 的。追加 token 后当前后缀也要走 child，node 和 child 可以**合并**。

- **Case 2a**（child.length == 1）：**融合** — child 吸收 node，length 合并
  ```
  Before: parent → node[A](count=2) → child[B](count=1, length=1)
  Append B:
  After:  parent → merged[AB](count=2, length=2)  ← node 和 child 合并
  ```

- **Case 2b**（child.length > 1）：**延长 node** — node 长度 +1，child 缩短 1
  ```
  Before: node[A](count=2, length=1) → child[BCD](count=1, length=3)
  Append B:
  After:  node[AB](count=2, length=2) → child[CD](count=1, length=2)
  ```

#### Case 3: 有子节点 + 普通扩展

当前后缀向下走到 child。

- **Case 3a**（child.length == 1）：直接移动，`_increment_count(child)`
  ```
  Before: node[AB](count=5) → child[D](count=2)
  Append D:
  After:  node[AB](count=5) → child[D](count=3)  ← count+1
  active_node 更新为 child
  ```

- **Case 3b**（child.length > 1）：需要**分裂 child**，插入中间节点
  ```
  Before: node[AB](count=5) → child[DEF](count=2, length=3)
  Append D:
  After:  node[AB](count=5) → new[D](count=3, length=1)
                                  └── child[EF](count=2, length=2)
  new 节点继承 child 在兄弟链中的位置，然后 _increment_count(new)
  ```

### 3.3 完整构建示例

序列 `[A, B, C, A, B, D]`，max_depth=64：

```
append(A):
  active_nodes = [root]
  root(count=1) → A(count=1, length=1)     [Case 1b: root 创建子节点]
  active_nodes = [A节点]

append(B):
  active_nodes = [A节点, root]              [root 新加入]
  root(count=2)
  A节点: 无子 B → Case 1a (count=1, 叶): A[AB] length=2
  root:   无子 B → Case 1b: 创建 B(count=1)
  active_nodes = [A[AB]节点, B节点]

append(C):
  active_nodes = [A[AB]节点, B节点, root]
  root(count=3)
  A[AB]: 无子 C → Case 1a: A[ABC] length=3
  B:     无子 C → Case 1a: B[BC] length=2
  root:  无子 C → Case 1b: 创建 C(count=1)
  active_nodes = [A[ABC]节点, B[BC]节点, C节点]

append(A):
  active_nodes = [..., root]
  root(count=4)
  A[ABC]: 无子 A → Case 1a: A[ABCA] length=4
  B[BC]:  无子 A → Case 1a: B[BCA] length=3
  C:      无子 A → Case 1a: C[CA] length=2
  root:   有子 A → Case 3b: A[ABCA] 需要分裂！
    分裂: root → A'(count=2, length=1) → [BCA](count=1, length=3)
    A' 的 active_node 指向 A'
  active_nodes = [A[ABCA], B[BCA], C[CA], A']

append(B):
  root(count=5)
  A[ABCA]: Case 1a → A[ABCAB] length=5
  B[BCA]:  Case 1a → B[BCAB] length=4
  C[CA]:   Case 1a → C[CAB] length=3
  A':      有子 B → 但这里 child 是 [BCA](length=3)
           Case 3b: 分裂 → A' → B'(count=2) → [CA](count=1)
  root:    有子 B → Case 3a: B[BCA] length>1, 分裂 → B''(count=2) → [CA](count=1)

  ... 以此类推

最终树结构编码了所有后缀的共享前缀关系。
```

### 3.4 `_increment_count` — 兄弟链表维护

当子节点的 count 增加时，它在兄弟链表中的位置可能需要更新（保持 count 降序）：

```python
def _increment_count(node):
    if not node.parent:
        node.count += 1; return

    if not node.prev_sibling or node.prev_sibling.count > node.count + 1:
        # 不需要移动（前面没人，或前面的 count 足够大）
        node.count += 1
        # 可能需要从当前 group 分离出新 group
    else:
        # 需要移动到前面的 group 头部之前
        target = node.prev_sibling.group.head
        _remove_from_siblings(node)
        node.count += 1
        _insert_into_siblings_before(node, target)
```

**关键规则**：`_insert_into_siblings_before(node, target)` — **新提升的节点插在同 count group 的最前面**。这决定了平局打断行为：当两个子节点 count 相同时，最近被提升的那个排在前面，`head_child` 会选择它。

---

## 四、查询流程 — `speculate(context, ...)`

### 4.1 外层循环：枚举 match_len

```cpp
Draft speculate(context, max_spec_tokens, max_spec_factor,
                max_spec_offset, min_token_prob, use_tree_spec) {
    Draft best_draft;
    for (int match_len = 1; match_len < context.size(); match_len++) {
        // 取 context 末尾 match_len 个 token 作为 pattern
        auto [node, idx] = _match_context(context.last(match_len));
        if (node == nullptr) break;  // 无匹配则停止（更长也不可能匹配）

        int max_tokens = min(max_spec_tokens,
                             int(match_len * max_spec_factor + max_spec_offset));
        Draft draft = _speculate_path(node, idx, max_tokens, min_token_prob);

        if (draft.score >= best_draft.score) {  // >= 偏好更长的 match_len
            best_draft = draft;
            best_draft.match_len = match_len;
        }
    }
    return best_draft;
}
```

**要点**：
- match_len 范围是 `[1, len(context)-1]`，**不包含** context 本身的完整长度
- 对每个 match_len 独立计算 draft 和 score，取 score 最高的
- `>=` 意味着当 score 相等时偏好更长的 match_len
- `break` 优化：如果 match_len=k 无匹配，match_len=k+1 也不可能匹配（后缀树性质）

### 4.2 `_match_context` — 在后缀树中匹配 pattern

```cpp
pair<Node*, int> _match_context(span<int> context) {
    Node* node = root;
    int idx = 0;
    for (int token : context) {
        if (idx >= node->length) {
            // 当前节点走完，查找子节点
            auto it = node->children.find(token);
            if (it == node->children.end())
                return {nullptr, -1};   // 无匹配
            node = it->second;
            idx = 0;                    // 进入子节点
        }
        // 检查节点内部 token 是否匹配
        if (_seqs[node->ref_seq][node->ref_idx + idx] != token)
            return {nullptr, -1};       // 节点内部失配
        idx++;
    }
    return {node, idx};  // 返回匹配到的节点 + 节点内偏移
}
```

返回值 `(node, idx)` 表示匹配到了 node 节点的第 idx 个位置。后续 speculate 从这个位置继续向下推测。

### 4.3 `_speculate_path` — 贪心推测（默认模式）

```cpp
Draft _speculate_path(Node* node, int idx, int max_spec_tokens, float min_token_prob) {
    Draft ret;
    float prob = 1.0f;  // float32 精度！
    while (ret.size() < max_spec_tokens && prob >= min_token_prob) {
        if (idx < node->length) {
            // 还在当前节点内部（路径压缩的后续 token）
            // 概率不变：这些 token 是确定的（同一条压缩边）
            ret.push(ref_token[idx], prob);
            idx++;
        } else {
            // 当前节点走完，跳到最优子节点
            Node* child = node->head_child;  // O(1) 取 count 最大的子节点
            if (child == nullptr) break;      // 叶节点，停止

            // 概率衰减：累积乘积
            prob *= float(child->count) / float(node->count);
            node = child;
            idx = 0;  // 子节点的 token 会在下一轮循环输出
        }
    }
    return ret;
}
```

**要点**：
- 始终跟随 `head_child`（count 最大的子节点），O(1) 取得
- 概率是**累积乘积**：`prob *= child.count / parent.count`，越深越低
- 节点内部（`idx < length`）的 token 概率不变——这些 token 在路径压缩中是确定的
- 使用 **float32** 精度计算概率（C++ `float` 类型），Python 复现需要 `struct.pack/unpack`
- 当 `prob < min_token_prob` 时停止，宁可少猜不乱猜

### 4.4 `_speculate_tree` — 多分支推测（可选模式）

```cpp
Draft _speculate_tree(Node* node, int idx, int max_spec_tokens, float min_token_prob) {
    // 优先队列（max-heap，按 prob 排序）
    priority_queue<HeapItem> queue;
    queue.push({prob=1.0, node, idx, parent=-1});

    while (ret.size() < max_spec_tokens && !queue.empty()) {
        auto item = queue.pop();
        if (item.idx < item.node->length) {
            // 节点内部：输出 token，继续在同一节点
            ret.push(token, item.prob);
            queue.push({item.prob, item.node, item.idx+1, current_idx});
        } else {
            // 遍历所有子节点（不只是 head_child），按 count 降序
            for (child = item.node->head_child; child; child = child->next_sibling) {
                float child_prob = item.prob * child->count / item.node->count;
                if (child_prob < min_token_prob) break;  // 兄弟链按 count 降序，后面更小
                queue.push({child_prob, child, 0, item.parent});
            }
        }
    }
    return ret;
}
```

vLLM 默认使用 Path 模式（`use_tree_spec=False`）。

### 4.5 完整查询示例

序列 `[A, B, C, D, E, A, B, C, G, H]`，context = `[X, A, B, C]`（len=4）

```
match_len=1: pattern=[C]
  _match_context([C]): root → C 节点 (count=2, 出现在 pos 2 和 pos 7)
  head_child: 看 C 节点的子节点
    D(count=1) 和 G(count=1) — 等 count
    假设 G 后被 increment → head_child = G
  prob = 1/2 = 0.5
  max_drafts = int(1 × 1.0) = 1
  draft = [G], score = 0.5

match_len=2: pattern=[B, C]
  _match_context([B, C]): root → B → C (count=2)
  同上: head_child 指向 G 或 D
  prob = 0.5
  max_drafts = int(2 × 1.0) = 2
  draft = [G, H] 或 [D, E], score = 0.5 + 0.5 = 1.0

match_len=3: pattern=[A, B, C]
  _match_context([A, B, C]): root → A → B → C (如果 ABC 被压缩为一个节点)
  count=2, 同样的子节点
  max_drafts = int(3 × 1.0) = 3
  draft = [G, H, ...] 或 [D, E, ...], score 可能更高

最终取 score 最高的 match_len。
```

### 4.6 `min_match_len` 过滤（Proposer 层）

`min_match_len` 不在 `SuffixTree.speculate` 内部实现，而是在上层 `SuffixDecodingProposer.propose()` 中作为**后置过滤**：

```python
# suffix_decoding.py (vLLM)
draft = self.suffix_cache.speculate(req_id, pattern, ...)

# 过滤：match_len 不够长的 draft 直接丢弃
if self.min_match_len > 0 and draft.match_len < self.min_match_len:
    draft_token_ids.append([])  # 不使用此 draft
else:
    draft_token_ids.append(draft.token_ids)
```

**参数**：
- C++ Suffix：`VLLM_SUFFIX_MIN_MATCH_LEN`（默认 0）
- PySuffix：`VLLM_PYSUFFIX_MIN_MATCH_LEN`（默认 0）

**效果**：设置 `min_match_len=N` 后，只有当 context 末尾至少 N 个 token 在后缀树中有完整匹配时，才会返回 draft token。相当于要求更高的匹配置信度。

**与 Trie 3-gram root 的关系**：Trie 的 `node_size=3`（3-gram root key）和 Suffix 的 `min_match_len=3` 本质等价 — 都要求至少匹配 3 个 token 才开始预测。但 Suffix 更灵活：不需要固定 root 大小，可以动态匹配任意长度后缀，`min_match_len` 只是一个过滤阈值。

**Benchmark 结论**（SWE-bench Lite, spec=5）：

| min_match_len | Speedup | Accept% | MeanLen | Drafts |
|---|---|---|---|---|
| 0（默认） | **2.02x** | 54.2% | 2.50 | 3480 |
| 1 | 2.02x | 54.2% | 2.50 | 3480 |
| 2 | 1.91x | 62.1% | 2.68 | 2894 |
| 3 | 1.82x | 68.5% | 2.95 | 2315 |

**结论**：`min_match_len > 0` 无收益。虽然 Accept% 随阈值提升（54% → 69%），但 draft 数量大幅减少（3480 → 2315），净效果为负。`min=0` 即为最优。

原因：`min_match_len=0` 实际等价于 `min_match_len=1`（C++ speculate 的 match_len 总是 ≥ 1）。提高阈值过滤掉了大量"短匹配但正确"的 draft — 这些 draft 虽然 match_len 短，但 prob 可能很高（head_child 唯一），丢弃它们得不偿失。

---

## 五、删除流程 — `remove(seq_id)`

当 `evict_cached_response` 或 `stop_request` 需要清理时：

```python
def remove(seq_id):
    seq = _seqs[seq_id]
    for start in range(len(seq)):      # 遍历所有后缀
        node = root
        node.count -= 1
        idx = start
        while idx < len(seq):
            child = node.children[seq[idx]]
            if child.count > 1:
                _decrement_count(child)      # count-1 并调整链表位置
            else:
                _remove_from_siblings(child)  # count=1 → 整棵子树删除
                del node.children[token]
                break
            idx += child.length
            node = child
        # 删除后可能出现单子节点可合并的情况
        if node.children.size() == 1 and node.count == only_child.count:
            merge(node, only_child)  # 合并：恢复路径压缩
    del _seqs[seq_id]
```

---

## 六、关键设计决策与影响

### 6.1 为什么枚举 match_len 而不是只用最长匹配？

**最长匹配不一定最优**。考虑：
- match_len=1: pattern=[X]，在树中出现 100 次，全部指向同一 token → prob=1.0，1 个 draft
- match_len=5: pattern=[A,B,C,D,X]，只出现 1 次 → prob=1.0，但 `max_drafts = 5×1.0 = 5`，可以给 5 个 draft

更长的 match_len 允许更多的 draft 数量（受 `max_spec_factor × match_len` 限制），但不一定有更高的概率。算法通过 **score = sum(probs)** 综合评估，让长匹配和短匹配公平竞争。

### 6.2 为什么 score 用 `>=` 而非 `>`？

**偏好更长的 match_len**。当 score 相同时：
- 长匹配意味着更精确的上下文定位
- 长匹配的 draft 虽然 max_drafts 更多，但在 score 相同时说明概率更集中

### 6.3 为什么 head_child 平局打断选"后提升者"？

`_increment_count` 将新提升的节点插入到 group 头部（`_insert_into_siblings_before`）。效果：**最近被观察到的模式优先**。

在实际推理中，最近生成的 token 建立的模式更可能是当前请求的局部模式，比早期的模式更有预测价值。

### 6.4 为什么用 float32 而不是 float64？

C++ 原生 `float` 类型。在 GPU 推理框架中，float32 是标准精度，额外的 double 精度没有实际收益，反而浪费带宽。Python 复现时必须用 `struct.pack('f', x)` 匹配这一精度，否则在 `prob >= min_token_prob` 边界处会产生判断分歧。

### 6.5 为什么 min_match_len 默认为 0？

直觉上，要求更长的匹配（min_match_len=2 或 3）应该提高预测质量。实测结果相反：

- **Accept% 上升但 draft 数量暴跌**：min=3 时 accept 68.5%（+14%），但 drafts 从 3480 降到 2315（-33%）
- **净效果为负**：speedup 从 2.02x 降到 1.82x
- **根因**：短匹配（match_len=1）的 draft 虽然上下文信息少，但大多数情况下 head_child 唯一（prob=1.0），预测正确率并不低。丢弃它们 = 丢弃了大量"简单但正确"的预测

### 6.6 max_tree_depth=24 的意义

限制后缀树索引的最大深度（每个后缀最多追踪 24 个 token）。效果：
- **内存控制**：避免为超长序列建立过深的后缀树
- **匹配范围**：context 也截取末尾 max_tree_depth 个 token
- **实测最优**：太深的匹配上下文窗口带来的增益边际递减

---

## 七、Python 复现要点

### 7.1 py_suffix_tree.py 中的关键实现

| 组件 | C++ | Python 对应 |
|------|-----|-------------|
| Node 结构 | `struct Node` | `class STNode` with `__slots__` |
| 子节点查找 | `Int32Map<unique_ptr<Node>>` | `dict[int, STNode]` |
| 兄弟链表 | `head_child/tail_child/next_sibling/prev_sibling` | 同名属性 |
| Group | `shared_ptr<Group>` | Python `Group` 对象 |
| `_increment_count` | 原地修改指针 | 逐行对应 |
| `_speculate_path` | `float prob = 1.0f` | `prob = _f32(1.0)` via `struct.pack` |
| `append` 6 种 case | 完整的 if-else 链 | 逐 case 对应 |

### 7.2 验证结果

| 测试 | 结果 |
|------|------|
| 单元测试（500 随机查询） | 100% token 精确匹配 |
| Dual-tree cache（1632 查询） | 100% 匹配 |
| SWE-bench Lite（20 samples, spec=5） | Drafts=3480/3480, DraftTok=9277/9277, Accept=53.2%/53.2% |

---

## 八、与其他方案的本质区别

| | Suffix Tree | Trie (LookaheadCache) | KMP | Hash |
|---|---|---|---|---|
| **索引结构** | 所有后缀的压缩共享 | 每个首 token 一棵独立 Trie | 无（每次重算） | dict[context, Counter] |
| **匹配方式** | 后缀子串匹配（任意位置） | 首 token 精确 + 后续前缀匹配 | 全序列 KMP 后缀匹配 | 哈希精确查找 |
| **概率估计** | 内建（count ratio） | 频率排序（无累积衰减） | 无 | 频率投票 |
| **match_len 选择** | 枚举所有，取最优 score | 滑窗遍历，取首个足够长的 | 全局最长后缀 | 从长到短，首个命中 |
| **跨请求利用** | global tree 共享 | output 频率共享 | 不支持 | 全局频率表 |
| **空间** | O(n) 路径压缩 | O(V×L) V=词汇命中数 | O(n) 临时 | O(n) |
| **时间** | O(m) per query | O(m + 候选数) | O(n) per query | O(n×m) per build |
