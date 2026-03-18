"""
Analyze Trie structure after running inference on SWE-bench Lite (20 samples).

Two phases:
  Phase 1: Run Trie (node_size=1) inference, save trie to pickle
  Phase 2: Load trie, collect stats, generate visualizations + MD report

Usage:
  # Full run (inference + analysis):
  python analyze_trie.py --model Qwen/Qwen2.5-3B-Instruct --gpu-mem 0.8

  # Analysis only (skip inference, load existing trie):
  python analyze_trie.py --analyze-only
"""

import argparse
import gc
import json
import math
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRIE_PICKLE = "trie_dump.pkl"
OUTPUT_DIR = "trie_analysis"
NUM_SAMPLES = 20
MAX_TOKENS = 512

# ---------------------------------------------------------------------------
# Phase 1: Run inference and save trie
# ---------------------------------------------------------------------------

def build_swe_prompts(num_samples=20, max_prompt_len=2048):
    from datasets import load_dataset
    print("Loading SWE-bench Lite ...")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    prompts = []
    for sample in ds:
        if len(prompts) >= num_samples:
            break
        problem = sample["problem_statement"] or ""
        repo = sample["repo"] or ""
        hints = sample["hints_text"] or ""
        text = (
            f"You are a software engineer fixing a bug in `{repo}`.\n\n"
            f"## Problem Description\n\n{problem}\n\n"
        )
        if hints:
            text += f"## Hints\n\n{hints}\n\n"
        text += (
            "## Task\n\n"
            "Analyze the bug and provide a fix as a unified diff patch. "
            "Think step by step:\n"
            "1. Identify the root cause.\n"
            "2. Determine which file(s) to modify.\n"
            "3. Provide the patch.\n\n"
            "```diff\n"
        )
        if len(text) > max_prompt_len:
            text = text[:max_prompt_len]
        prompts.append(text)
    print(f"Built {len(prompts)} prompts")
    return prompts


def _inference_worker(prompt_texts, model_name, gpu_mem, max_tokens, trie_path):
    """Top-level function for spawn-based multiprocessing."""
    import torch as _torch
    os.environ["VLLM_NGRAM_USE_HASH"] = "0"
    os.environ["VLLM_NGRAM_USE_TRIE"] = "1"
    os.environ["VLLM_TRIE_NODE_SIZE"] = "1"
    os.environ["VLLM_TRIE_TABLE_PATH"] = trie_path

    from vllm import LLM, SamplingParams

    spec_config = {
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 5,
        "prompt_lookup_min": 2,
    }

    print(f"\nStarting Trie (node_size=1) inference on {len(prompt_texts)} samples...")
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_mem,
        speculative_config=spec_config,
        disable_log_stats=False,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    # Warmup
    print("Warmup ...")
    llm.generate([prompt_texts[0]], sampling_params)

    # Run all samples
    print(f"Running {len(prompt_texts)} samples ...")
    for idx, prompt in enumerate(prompt_texts):
        t0 = time.perf_counter()
        outputs = llm.generate([prompt], sampling_params)
        t1 = time.perf_counter()
        o = outputs[0]
        out_tok = len(o.outputs[0].token_ids)
        print(f"  [{idx+1}/{len(prompt_texts)}] out={out_tok} lat={t1-t0:.2f}s")

    # Shutdown triggers atexit -> _trie_persist_sync -> saves to trie_path
    print("Shutting down engine (trie will be saved via atexit) ...")
    llm.llm_engine.engine_core.shutdown()
    del llm
    gc.collect()
    _torch.cuda.empty_cache()


def run_inference_and_save_trie(model_name, gpu_mem):
    """Run Trie (node_size=1) inference in a subprocess.

    vLLM v1 LLM uses SyncMPClient — the EngineCore (and NgramProposer)
    lives in a child process.  We set VLLM_TRIE_TABLE_PATH so the atexit
    handler in NgramProposer persists the trie when the engine shuts down.
    After the subprocess exits we wrap the raw mem dict into our standard
    pickle format.
    """
    import multiprocessing as mp

    trie_raw_path = os.path.abspath(TRIE_PICKLE + ".raw")

    prompts = build_swe_prompts(NUM_SAMPLES)

    ctx = mp.get_context("spawn")
    p = ctx.Process(target=_inference_worker,
                    args=(prompts, model_name, gpu_mem, MAX_TOKENS, trie_raw_path))
    p.start()
    p.join(timeout=1800)
    if p.is_alive():
        p.terminate()
        p.join()
        print("ERROR: subprocess timed out")
        return

    if not os.path.exists(trie_raw_path):
        print(f"ERROR: trie not saved to {trie_raw_path}")
        print("The atexit handler may not have fired. Check logs.")
        return

    # Wrap raw mem dict into our standard format
    with open(trie_raw_path, "rb") as f:
        mem = pickle.load(f)
    with open(TRIE_PICKLE, "wb") as f:
        pickle.dump({
            "mem": mem,
            "node_size": 1,
            "model": model_name,
            "num_samples": NUM_SAMPLES,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.remove(trie_raw_path)
    print(f"Trie saved: {TRIE_PICKLE} ({len(mem)} root entries)")


# ---------------------------------------------------------------------------
# Phase 2: Trie Analysis
# ---------------------------------------------------------------------------

def load_trie(path=TRIE_PICKLE):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "mem" in data:
        return data["mem"], data.get("node_size", 1), data.get("model", "unknown")
    else:
        # Raw mem dict (from VLLM_TRIE_TABLE_PATH)
        return data, 1, "unknown"


def traverse_trie_tree(nodes, depth=0, stats=None):
    """DFS traverse a TrieTree's nodes, collecting per-depth stats."""
    if stats is None:
        stats = defaultdict(lambda: {
            "node_count": 0,
            "token_freq": Counter(),    # token_id -> total_freq
            "input_freqs": [],
            "output_freqs": [],
            "mix_freqs": [],
            "branching_factors": [],
            "children_tokens": [],      # (parent_token, child_token, freq)
        })

    for token_id, node in nodes.items():
        s = stats[depth]
        s["node_count"] += 1

        fi = sum(v for k, v in node.freqs.items() if k != -1)
        fo = node.freqs.get(-1, 0.0)
        fm = fi + fo
        total_freq = fm

        s["token_freq"][token_id] += total_freq
        if fi > 0:
            s["input_freqs"].append(fi)
        if fo > 0:
            s["output_freqs"].append(fo)
        if fm > 0:
            s["mix_freqs"].append(fm)

        n_children = len(node.children)
        s["branching_factors"].append(n_children)

        # Record edges
        for child_id, child_node in node.children.items():
            child_fo = child_node.freqs.get(-1, 0.0)
            child_fi = sum(v for k, v in child_node.freqs.items() if k != -1)
            s["children_tokens"].append((token_id, child_id, child_fi + child_fo))

        if n_children > 0:
            traverse_trie_tree(node.children, depth + 1, stats)

    return stats


def collect_global_stats(mem):
    """Collect comprehensive statistics across all root entries."""
    global_stats = {
        "num_roots": len(mem),
        "per_depth": defaultdict(lambda: {
            "node_count": 0,
            "token_freq": Counter(),
            "input_freqs": [],
            "output_freqs": [],
            "mix_freqs": [],
            "branching_factors": [],
            "edges": [],  # (parent_tok, child_tok, freq)
        }),
        "root_sizes": [],        # (root_key, total_nodes)
        "root_freq_totals": [],  # (root_key, total_freq)
    }

    for root_key, tree in mem.items():
        # Count nodes in this tree
        sizes = [0]
        _count_nodes(tree.nodes, sizes)
        total_nodes = sizes[0]
        global_stats["root_sizes"].append((root_key, total_nodes))

        # Traverse tree
        tree_stats = traverse_trie_tree(tree.nodes)

        # Merge into global
        total_freq = 0
        for depth, ds in tree_stats.items():
            gd = global_stats["per_depth"][depth]
            gd["node_count"] += ds["node_count"]
            gd["token_freq"].update(ds["token_freq"])
            gd["input_freqs"].extend(ds["input_freqs"])
            gd["output_freqs"].extend(ds["output_freqs"])
            gd["mix_freqs"].extend(ds["mix_freqs"])
            gd["branching_factors"].extend(ds["branching_factors"])
            gd["edges"].extend(ds["children_tokens"])
            if depth == 0:
                total_freq += sum(ds["mix_freqs"])

        global_stats["root_freq_totals"].append((root_key, total_freq))

    return global_stats


def _count_nodes(nodes, sizes):
    sizes[0] += len(nodes)
    for t, n in nodes.items():
        if len(n.children) > 0:
            _count_nodes(n.children, sizes)


def get_tokenizer(model_name):
    """Load tokenizer for decoding token IDs."""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        print(f"Warning: Could not load tokenizer: {e}")
        return None


def tok_to_str(tokenizer, token_id):
    """Convert token ID to readable string."""
    if tokenizer is None:
        return str(token_id)
    try:
        s = tokenizer.decode([token_id])
        # Escape for markdown
        s = s.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
        s = s.replace("|", "\\|")
        if not s.strip():
            s = repr(s)
        return s
    except:
        return str(token_id)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_depth_distribution(stats, output_dir):
    """Bar chart: node count per depth level."""
    depths = sorted(stats["per_depth"].keys())
    counts = [stats["per_depth"][d]["node_count"] for d in depths]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(depths, counts, color='steelblue', edgecolor='white')
    ax.set_xlabel('Depth (层级)', fontsize=12)
    ax.set_ylabel('Node Count (节点数)', fontsize=12)
    ax.set_title('Trie Node Count per Depth Level', fontsize=14)
    ax.set_xticks(depths)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{count:,}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "depth_node_count.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_branching_factor(stats, output_dir):
    """Box plot: branching factor distribution per depth."""
    depths = sorted(stats["per_depth"].keys())
    data = []
    labels = []
    for d in depths:
        bfs = stats["per_depth"][d]["branching_factors"]
        if bfs:
            data.append(bfs)
            labels.append(str(d))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Box plot
    ax = axes[0]
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.set_xlabel('Depth', fontsize=12)
    ax.set_ylabel('Branching Factor', fontsize=12)
    ax.set_title('Branching Factor Distribution per Depth', fontsize=13)

    # Mean branching factor line
    ax2 = axes[1]
    means = [np.mean(d) for d in data]
    medians = [np.median(d) for d in data]
    x = range(len(labels))
    ax2.plot(x, means, 'o-', color='steelblue', label='Mean', linewidth=2)
    ax2.plot(x, medians, 's--', color='coral', label='Median', linewidth=2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_xlabel('Depth', fontsize=12)
    ax2.set_ylabel('Branching Factor', fontsize=12)
    ax2.set_title('Mean/Median Branching Factor', fontsize=13)
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "branching_factor.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_freq_distribution(stats, output_dir):
    """Histogram: frequency distribution at each depth (input vs output)."""
    depths = sorted(stats["per_depth"].keys())[:5]  # Top 5 depths

    fig, axes = plt.subplots(1, len(depths), figsize=(4*len(depths), 4),
                             squeeze=False)
    axes = axes[0]

    for i, d in enumerate(depths):
        ax = axes[i]
        ifreqs = stats["per_depth"][d]["input_freqs"]
        ofreqs = stats["per_depth"][d]["output_freqs"]

        all_freqs = ifreqs + ofreqs
        if not all_freqs:
            continue
        max_f = np.percentile(all_freqs, 95) if all_freqs else 1
        bins = np.linspace(0, max_f, 30)

        if ifreqs:
            ax.hist(ifreqs, bins=bins, alpha=0.6, color='steelblue',
                    label='Input', density=True)
        if ofreqs:
            ax.hist(ofreqs, bins=bins, alpha=0.6, color='coral',
                    label='Output', density=True)
        ax.set_title(f'Depth {d}', fontsize=11)
        ax.set_xlabel('Frequency')
        if i == 0:
            ax.set_ylabel('Density')
        ax.legend(fontsize=8)

    plt.suptitle('Frequency Distribution (Input vs Output) per Depth',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "freq_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_top_roots(stats, tokenizer, output_dir, top_n=30):
    """Bar chart: top-N root tokens by total subtree size."""
    root_sizes = sorted(stats["root_sizes"], key=lambda x: x[1], reverse=True)[:top_n]
    labels = [tok_to_str(tokenizer, rk) for rk, _ in root_sizes]
    sizes = [s for _, s in root_sizes]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(labels))
    ax.bar(x, sizes, color='steelblue', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=8)
    ax.set_ylabel('Subtree Size (total nodes)', fontsize=12)
    ax.set_title(f'Top {top_n} Root Tokens by Subtree Size', fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, "top_roots.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_top_tokens_per_depth(stats, tokenizer, output_dir, top_n=15):
    """Horizontal bar chart: top tokens at each depth."""
    depths = sorted(stats["per_depth"].keys())[:5]

    fig, axes = plt.subplots(1, len(depths), figsize=(4*len(depths), 6),
                             squeeze=False)
    axes = axes[0]

    for i, d in enumerate(depths):
        ax = axes[i]
        tf = stats["per_depth"][d]["token_freq"]
        top_tokens = tf.most_common(top_n)
        if not top_tokens:
            continue

        labels = [tok_to_str(tokenizer, tid) for tid, _ in top_tokens]
        freqs = [f for _, f in top_tokens]

        y = range(len(labels))
        ax.barh(y, freqs, color='steelblue')
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f'Depth {d}', fontsize=11)
        ax.set_xlabel('Total Freq')

    plt.suptitle(f'Top {top_n} Tokens per Depth Level', fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "top_tokens_per_depth.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_edge_heatmap(stats, tokenizer, output_dir, depth=0, top_n=20):
    """Heatmap: top parent→child token transition frequencies at a given depth."""
    edges = stats["per_depth"][depth]["edges"]
    if not edges:
        return None

    # Aggregate edge freqs
    edge_freq = Counter()
    for parent, child, freq in edges:
        edge_freq[(parent, child)] += freq

    # Get top parent and child tokens
    parent_freq = Counter()
    child_freq = Counter()
    for (p, c), f in edge_freq.items():
        parent_freq[p] += f
        child_freq[c] += f

    top_parents = [t for t, _ in parent_freq.most_common(top_n)]
    top_children = [t for t, _ in child_freq.most_common(top_n)]

    # Build matrix
    matrix = np.zeros((len(top_parents), len(top_children)))
    p_idx = {t: i for i, t in enumerate(top_parents)}
    c_idx = {t: i for i, t in enumerate(top_children)}
    for (p, c), f in edge_freq.items():
        if p in p_idx and c in c_idx:
            matrix[p_idx[p], c_idx[c]] = f

    p_labels = [tok_to_str(tokenizer, t) for t in top_parents]
    c_labels = [tok_to_str(tokenizer, t) for t in top_children]

    fig, ax = plt.subplots(figsize=(max(10, top_n*0.6), max(8, top_n*0.5)))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(c_labels)))
    ax.set_xticklabels(c_labels, rotation=60, ha='right', fontsize=8)
    ax.set_yticks(range(len(p_labels)))
    ax.set_yticklabels(p_labels, fontsize=8)
    ax.set_xlabel(f'Child Tokens (Depth {depth+1})', fontsize=11)
    ax.set_ylabel(f'Parent Tokens (Depth {depth})', fontsize=11)
    ax.set_title(f'Token Transition Heatmap: Depth {depth} → {depth+1}', fontsize=13)
    plt.colorbar(im, ax=ax, label='Frequency')
    plt.tight_layout()
    path = os.path.join(output_dir, f"edge_heatmap_d{depth}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_sample_subtree(mem, tokenizer, output_dir, top_k_roots=3, max_depth=4, max_children=5):
    """Draw a few sample subtrees to show the actual tree structure."""
    # Pick the top-K largest root trees
    root_sizes = []
    for root_key, tree in mem.items():
        sizes = [0]
        _count_nodes(tree.nodes, sizes)
        root_sizes.append((root_key, tree, sizes[0]))
    root_sizes.sort(key=lambda x: x[2], reverse=True)

    figs_paths = []
    for rank, (root_key, tree, total) in enumerate(root_sizes[:top_k_roots]):
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.set_xlim(-1, 1)
        ax.set_ylim(-0.1, 1.1)
        ax.axis('off')

        root_label = tok_to_str(tokenizer, root_key)
        ax.set_title(f'Subtree #{rank+1}: root="{root_label}" (id={root_key}, {total} nodes)',
                     fontsize=13)

        # BFS layout
        _draw_subtree(ax, tree.nodes, tokenizer, root_label,
                      max_depth=max_depth, max_children=max_children)

        plt.tight_layout()
        path = os.path.join(output_dir, f"subtree_{rank}.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        figs_paths.append(path)

    return figs_paths


def _draw_subtree(ax, nodes, tokenizer, root_label, max_depth=4, max_children=5):
    """Draw a tree using simple coordinate layout."""
    # Build layout via BFS
    # Each level: list of (x, y, label, freq_str, parent_xy)
    levels = []
    # Root
    root_x, root_y = 0.5, 1.0
    ax.text(root_x, root_y, f'[{root_label}]', ha='center', va='center',
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', edgecolor='steelblue'))

    # Children of root
    queue = []  # (nodes_dict, parent_x, parent_y, depth, x_left, x_right)
    queue.append((nodes, root_x, root_y, 0, 0.0, 1.0))

    while queue:
        cur_nodes, px, py, depth, xl, xr = queue.pop(0)
        if depth >= max_depth:
            continue

        # Sort children by freq descending
        children = []
        for tid, node in cur_nodes.items():
            fo = node.freqs.get(-1, 0.0)
            fi = sum(v for k, v in node.freqs.items() if k != -1)
            children.append((tid, node, fi + fo, fi, fo))
        children.sort(key=lambda x: x[2], reverse=True)
        children = children[:max_children]

        if not children:
            continue

        n = len(children)
        cy = py - 0.22
        width = xr - xl
        margin = width * 0.05

        for i, (tid, node, fm, fi, fo) in enumerate(children):
            if n == 1:
                cx = (xl + xr) / 2
            else:
                cx = xl + margin + (width - 2*margin) * i / (n - 1)

            label = tok_to_str(tokenizer, tid)
            freq_str = f'f={fm:.0f}'
            if fi > 0 and fo > 0:
                freq_str = f'i={fi:.0f} o={fo:.0f}'

            # Draw edge
            ax.annotate('', xy=(cx, cy + 0.06), xytext=(px, py - 0.06),
                        arrowprops=dict(arrowstyle='->', color='gray',
                                        lw=max(0.5, min(3, fm/5))))

            # Draw node
            color = 'lightyellow' if fo > fi else 'lavender'
            ax.text(cx, cy, f'{label}\n{freq_str}', ha='center', va='center',
                    fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=color,
                              edgecolor='gray', alpha=0.9))

            # Truncation indicator
            n_gc = len(node.children)
            if n_gc > max_children and depth + 1 < max_depth:
                ax.text(cx, cy - 0.03, f'(+{n_gc - max_children} more)',
                        ha='center', va='top', fontsize=5, color='gray')

            if len(node.children) > 0:
                child_width = width / n if n > 1 else width * 0.5
                child_xl = cx - child_width / 2
                child_xr = cx + child_width / 2
                queue.append((node.children, cx, cy, depth + 1,
                              child_xl, child_xr))


def plot_root_size_distribution(stats, output_dir):
    """Histogram of root subtree sizes."""
    sizes = [s for _, s in stats["root_sizes"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    ax = axes[0]
    ax.hist(sizes, bins=50, color='steelblue', edgecolor='white')
    ax.set_xlabel('Subtree Size (nodes)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Distribution of Root Subtree Sizes', fontsize=13)

    # Log-scale CDF
    ax = axes[1]
    sorted_sizes = np.sort(sizes)[::-1]
    cdf = np.arange(1, len(sorted_sizes) + 1) / len(sorted_sizes)
    ax.plot(sorted_sizes, cdf, color='steelblue', linewidth=2)
    ax.set_xlabel('Subtree Size (nodes)', fontsize=11)
    ax.set_ylabel('Cumulative Fraction of Roots', fontsize=11)
    ax.set_title('CDF: Root Subtree Size (sorted desc)', fontsize=13)
    ax.set_xscale('log')

    plt.tight_layout()
    path = os.path.join(output_dir, "root_size_dist.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Generate Markdown report
# ---------------------------------------------------------------------------

def generate_report(stats, mem, tokenizer, output_dir, model_name):
    """Generate the final markdown report with embedded images."""
    lines = []
    lines.append("# Trie Structure Analysis Report")
    lines.append("")
    lines.append(f"- **Model**: {model_name}")
    lines.append(f"- **Dataset**: SWE-bench Lite ({NUM_SAMPLES} samples)")
    lines.append(f"- **Trie mode**: node_size=1 (original), spec=5")
    lines.append(f"- **Total root entries**: {stats['num_roots']:,}")
    total_nodes = sum(s for _, s in stats["root_sizes"])
    lines.append(f"- **Total nodes (all trees)**: {total_nodes:,}")
    lines.append("")

    # --- Overall depth stats ---
    lines.append("## 1. Depth-Level Overview")
    lines.append("")
    lines.append("### 1.1 Node Count per Depth")
    lines.append("")
    lines.append("![Node count per depth](depth_node_count.png)")
    lines.append("")

    depths = sorted(stats["per_depth"].keys())
    lines.append("| Depth | Nodes | Unique Tokens | Avg Branching | Median Branching | Avg Freq (mix) |")
    lines.append("|-------|-------|---------------|---------------|------------------|----------------|")
    for d in depths:
        pd = stats["per_depth"][d]
        n_nodes = pd["node_count"]
        n_unique = len(pd["token_freq"])
        bfs = pd["branching_factors"]
        avg_bf = np.mean(bfs) if bfs else 0
        med_bf = np.median(bfs) if bfs else 0
        mf = pd["mix_freqs"]
        avg_mf = np.mean(mf) if mf else 0
        lines.append(f"| {d} | {n_nodes:,} | {n_unique:,} | {avg_bf:.2f} | {med_bf:.1f} | {avg_mf:.2f} |")
    lines.append("")

    # --- Branching factor ---
    lines.append("### 1.2 Branching Factor Distribution")
    lines.append("")
    lines.append("![Branching factor](branching_factor.png)")
    lines.append("")
    lines.append("> Branching factor = number of children per node. "
                 "Higher branching means more diverse continuations.")
    lines.append("")

    # --- Frequency distribution ---
    lines.append("### 1.3 Frequency Distribution (Input vs Output)")
    lines.append("")
    lines.append("![Frequency distribution](freq_distribution.png)")
    lines.append("")
    lines.append("> Input freq: from prompt tokens. Output freq: from generated tokens. "
                 "Output freq drives draft selection in 'mix' mode.")
    lines.append("")

    # --- Root analysis ---
    lines.append("## 2. Root Token Analysis")
    lines.append("")
    lines.append("### 2.1 Top Root Tokens by Subtree Size")
    lines.append("")
    lines.append("![Top roots](top_roots.png)")
    lines.append("")

    root_sizes = sorted(stats["root_sizes"], key=lambda x: x[1], reverse=True)
    lines.append("| Rank | Root Token | Token ID | Subtree Nodes | Total Freq |")
    lines.append("|------|-----------|----------|---------------|------------|")
    freq_map = {rk: f for rk, f in stats["root_freq_totals"]}
    for i, (rk, sz) in enumerate(root_sizes[:20]):
        label = tok_to_str(tokenizer, rk)
        freq = freq_map.get(rk, 0)
        lines.append(f"| {i+1} | `{label}` | {rk} | {sz:,} | {freq:.0f} |")
    lines.append("")

    # --- Root size distribution ---
    lines.append("### 2.2 Root Subtree Size Distribution")
    lines.append("")
    lines.append("![Root size distribution](root_size_dist.png)")
    lines.append("")
    sizes = [s for _, s in stats["root_sizes"]]
    lines.append(f"- **Max subtree**: {max(sizes):,} nodes")
    lines.append(f"- **Median subtree**: {int(np.median(sizes)):,} nodes")
    lines.append(f"- **Mean subtree**: {np.mean(sizes):.1f} nodes")
    lines.append(f"- **Roots with 1 node**: {sum(1 for s in sizes if s == 1):,} "
                 f"({sum(1 for s in sizes if s == 1)/len(sizes)*100:.1f}%)")
    lines.append(f"- **Roots with >100 nodes**: {sum(1 for s in sizes if s > 100):,} "
                 f"({sum(1 for s in sizes if s > 100)/len(sizes)*100:.1f}%)")
    lines.append("")

    # --- Top tokens per depth ---
    lines.append("## 3. Token Distribution per Depth")
    lines.append("")
    lines.append("![Top tokens per depth](top_tokens_per_depth.png)")
    lines.append("")

    for d in depths[:5]:
        tf = stats["per_depth"][d]["token_freq"]
        top = tf.most_common(10)
        lines.append(f"### Depth {d} — Top 10 Tokens")
        lines.append("")
        lines.append("| Token | Token ID | Total Freq |")
        lines.append("|-------|----------|------------|")
        for tid, freq in top:
            lines.append(f"| `{tok_to_str(tokenizer, tid)}` | {tid} | {freq:.0f} |")
        lines.append("")

    # --- Edge heatmaps ---
    lines.append("## 4. Token Transition Patterns (Edge Heatmaps)")
    lines.append("")
    for d in [0, 1]:
        if d in stats["per_depth"] and stats["per_depth"][d]["edges"]:
            lines.append(f"### Depth {d} → {d+1}")
            lines.append("")
            lines.append(f"![Edge heatmap depth {d}](edge_heatmap_d{d}.png)")
            lines.append("")
            lines.append(f"> Top 20 parent × child token pairs by frequency at depth {d}→{d+1}.")
            lines.append("")

    # --- Sample subtrees ---
    lines.append("## 5. Sample Subtree Visualizations")
    lines.append("")
    lines.append("> Showing the 3 largest root subtrees (pruned to depth=4, max 5 children per node).")
    lines.append("> Node color: lavender=input-dominant, lightyellow=output-dominant.")
    lines.append("")
    for i in range(3):
        path = f"subtree_{i}.png"
        if os.path.exists(os.path.join(output_dir, path)):
            lines.append(f"### Subtree #{i+1}")
            lines.append("")
            lines.append(f"![Subtree {i}]({path})")
            lines.append("")

    # --- Design implications ---
    lines.append("## 6. Key Observations for Design")
    lines.append("")

    # Compute some derived stats
    d0_bf = stats["per_depth"][0]["branching_factors"] if 0 in stats["per_depth"] else []
    avg_d0_bf = np.mean(d0_bf) if d0_bf else 0
    sizes = [s for _, s in stats["root_sizes"]]
    pct_small = sum(1 for s in sizes if s <= 5) / len(sizes) * 100 if sizes else 0
    pct_large = sum(1 for s in sizes if s > 50) / len(sizes) * 100 if sizes else 0

    lines.append(f"1. **Root分布不均匀**: {pct_small:.1f}% 的 root 只有 ≤5 个节点，"
                 f"而 {pct_large:.1f}% 的 root 有 >50 个节点。少量 hot root 贡献了大部分预测能力。")
    lines.append(f"2. **平均分支因子**: Depth 0 的平均分支因子为 {avg_d0_bf:.2f}，随深度递减。"
                 "深层节点趋于单链，说明长序列的共性减少。")

    d0_ofreqs = stats["per_depth"].get(0, {}).get("output_freqs", [])
    d0_ifreqs = stats["per_depth"].get(0, {}).get("input_freqs", [])
    lines.append(f"3. **Input vs Output 频率**: Depth 0 有 {len(d0_ifreqs):,} 个 input 条目 vs "
                 f"{len(d0_ofreqs):,} 个 output 条目。Input 来自 prompt 扫描，数量远大于 output（生成的 token）。")
    lines.append("")

    # Write report
    report_path = os.path.join(output_dir, "trie_analysis.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report written to {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--gpu-mem", type=float, default=0.8)
    parser.add_argument("--analyze-only", action="store_true",
                        help="Skip inference, load existing trie pickle")
    parser.add_argument("--trie-pickle", default=TRIE_PICKLE)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Phase 1
    if not args.analyze_only:
        run_inference_and_save_trie(args.model, args.gpu_mem)

    # Phase 2
    if not os.path.exists(args.trie_pickle):
        print(f"ERROR: {args.trie_pickle} not found. Run inference first.")
        sys.exit(1)

    print("\n=== Phase 2: Analysis ===")
    mem, node_size, model_name = load_trie(args.trie_pickle)
    if args.model != "Qwen/Qwen2.5-3B-Instruct":
        model_name = args.model
    print(f"Loaded trie: {len(mem)} root entries, node_size={node_size}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = get_tokenizer(model_name)

    # Collect stats
    print("Collecting statistics...")
    stats = collect_global_stats(mem)

    # Generate plots
    print("Generating visualizations...")
    plot_depth_distribution(stats, OUTPUT_DIR)
    plot_branching_factor(stats, OUTPUT_DIR)
    plot_freq_distribution(stats, OUTPUT_DIR)
    plot_top_roots(stats, tokenizer, OUTPUT_DIR)
    plot_top_tokens_per_depth(stats, tokenizer, OUTPUT_DIR)
    plot_root_size_distribution(stats, OUTPUT_DIR)

    # Edge heatmaps for depth 0 and 1
    for d in [0, 1]:
        plot_edge_heatmap(stats, tokenizer, OUTPUT_DIR, depth=d)

    # Sample subtrees
    print("Drawing sample subtrees...")
    plot_sample_subtree(mem, tokenizer, OUTPUT_DIR)

    # Generate report
    print("Generating report...")
    generate_report(stats, mem, tokenizer, OUTPUT_DIR, model_name)

    print(f"\nDone! Report: {OUTPUT_DIR}/trie_analysis.md")


if __name__ == "__main__":
    main()
