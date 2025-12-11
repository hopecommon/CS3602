#!/usr/bin/env python3
"""
Generate comprehensive comparison plots from results/comprehensive/.

Supports three evaluation types:
1. chunked - Chunked evaluation (baseline, ours, mit)
2. decode_loop - Decode-loop evaluation (baseline, ours, mit)
3. kvpress_official - KVPress official evaluation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

# Configuration
RESULTS_DIR = Path("results/comprehensive")
OUTPUT_DIR = Path("results/figures")
DATASETS = ["wikitext", "pg19_20k"]
DATASET_LABELS = {
    "wikitext": "WikiText-103\n(4k tokens)",
    "pg19_20k": "PG19\n(20k tokens)"
}

# Methods for each evaluation type
DECODE_LOOP_METHODS = ["baseline", "ours", "mit"]
CHUNKED_METHODS = ["baseline", "ours", "mit"]

# Color scheme
COLORS = {
    "baseline": "#2E86AB",
    "ours": "#A23B72",
    "mit": "#F18F01",
    "kvpress": "#06A77D",
}

METHOD_LABELS = {
    "baseline": "Baseline",
    "ours": "Ours",
    "mit": "MIT-style",
    "kvpress": "KVPress"
}


def load_result(dataset: str, method: str, eval_type: str) -> Dict:
    """Load result JSON file."""
    path = RESULTS_DIR / f"{dataset}_{method}_{eval_type}.json"
    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def plot_decode_loop_comparison():
    """Generate decode-loop comparison plots (2x2 layout)."""
    print("\n生成 Decode-loop 综合对比图...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Decode-Loop Evaluation Results (pythia-2.8b)", 
                 fontsize=16, fontweight="bold", y=0.995)
    
    x = np.arange(len(DATASETS))
    width = 0.25
    
    # Collect data
    data = {method: {"runtime": [], "ppl": [], "speedup": [], "ppl_increase": []} 
            for method in DECODE_LOOP_METHODS}
    
    for dataset in DATASETS:
        baseline_data = load_result(dataset, "baseline", "decode_loop")
        baseline_time = baseline_data["total_time"]
        baseline_ppl = baseline_data["perplexity"]
        
        for method in DECODE_LOOP_METHODS:
            method_data = load_result(dataset, method, "decode_loop")
            data[method]["runtime"].append(method_data["total_time"])
            data[method]["ppl"].append(method_data["perplexity"])
            
            if method != "baseline":
                speedup = baseline_time / method_data["total_time"]
                ppl_increase = ((method_data["perplexity"] - baseline_ppl) / baseline_ppl) * 100
                data[method]["speedup"].append(speedup)
                data[method]["ppl_increase"].append(ppl_increase)
            else:
                data[method]["speedup"].append(1.0)
                data[method]["ppl_increase"].append(0.0)
    
    # (a) Runtime
    ax1 = axes[0, 0]
    for i, method in enumerate(DECODE_LOOP_METHODS):
        offset = (i - 1) * width
        bars = ax1.bar(x + offset, data[method]["runtime"], width,
                       label=METHOD_LABELS[method], color=COLORS[method], 
                       alpha=0.8, edgecolor="black", linewidth=1.0)
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2.0, height,
                    f'{height:.1f}s', ha='center', va='bottom', fontsize=8)
    ax1.set_ylabel("Runtime (s)", fontweight="bold")
    ax1.set_title("(a) Runtime Comparison", fontweight="bold", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels([DATASET_LABELS[d] for d in DATASETS])
    ax1.legend(frameon=True, shadow=True, fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    
    # (b) PPL
    ax2 = axes[0, 1]
    for i, method in enumerate(DECODE_LOOP_METHODS):
        offset = (i - 1) * width
        bars = ax2.bar(x + offset, data[method]["ppl"], width,
                       label=METHOD_LABELS[method], color=COLORS[method],
                       alpha=0.8, edgecolor="black", linewidth=1.0)
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2.0, height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    ax2.set_ylabel("Perplexity", fontweight="bold")
    ax2.set_title("(b) Perplexity Comparison", fontweight="bold", pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels([DATASET_LABELS[d] for d in DATASETS])
    ax2.legend(frameon=True, shadow=True, fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    
    # (c) Speedup
    ax3 = axes[1, 0]
    for i, method in enumerate(["ours", "mit"]):
        offset = (i - 0.5) * width
        bars = ax3.bar(x + offset, data[method]["speedup"], width,
                       label=METHOD_LABELS[method], color=COLORS[method],
                       alpha=0.8, edgecolor="black", linewidth=1.0)
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2.0, height,
                    f'{height:.2f}×', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax3.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.set_ylabel("Speedup (×)", fontweight="bold")
    ax3.set_title("(c) Speedup vs Baseline", fontweight="bold", pad=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels([DATASET_LABELS[d] for d in DATASETS])
    ax3.legend(frameon=True, shadow=True, fontsize=9)
    ax3.grid(axis="y", alpha=0.3)
    
    # (d) PPL Increase
    ax4 = axes[1, 1]
    for i, method in enumerate(["ours", "mit"]):
        offset = (i - 0.5) * width
        bars = ax4.bar(x + offset, data[method]["ppl_increase"], width,
                       label=METHOD_LABELS[method], color=COLORS[method],
                       alpha=0.8, edgecolor="black", linewidth=1.0)
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2.0, height,
                    f'{height:+.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1.0, alpha=0.5)
    ax4.axhline(y=1, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Target (<1%)')
    ax4.axhline(y=3, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Acceptable (<3%)')
    ax4.set_ylabel("PPL Increase (%)", fontweight="bold")
    ax4.set_title("(d) PPL Increase vs Baseline", fontweight="bold", pad=10)
    ax4.set_xticks(x)
    ax4.set_xticklabels([DATASET_LABELS[d] for d in DATASETS])
    ax4.legend(frameon=True, shadow=True, fontsize=8)
    ax4.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "comprehensive_decode_loop_summary.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ 已保存: {output_path}")


def plot_chunked_comparison():
    """Generate chunked evaluation comparison plots."""
    print("\n生成 Chunked Evaluation 对比图...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Chunked Evaluation Results (pythia-2.8b)", 
                 fontsize=14, fontweight="bold", y=0.98)
    
    x = np.arange(len(DATASETS))
    width = 0.25
    
    # Collect data
    runtime_data = {method: [] for method in CHUNKED_METHODS}
    ppl_data = {method: [] for method in CHUNKED_METHODS}
    
    for dataset in DATASETS:
        for method in CHUNKED_METHODS:
            try:
                data = load_result(dataset, method, "chunked")
                # Chunked results have nested structure
                if method == "baseline":
                    runtime_data[method].append(data.get("baseline", {}).get("runtime_sec", 0))
                    ppl_data[method].append(data.get("baseline", {}).get("perplexity", 0))
                else:
                    runtime_data[method].append(data.get("streaming", {}).get("runtime_sec", 0))
                    ppl_data[method].append(data.get("streaming", {}).get("perplexity", 0))
            except (FileNotFoundError, KeyError):
                runtime_data[method].append(0)
                ppl_data[method].append(0)
    
    # (a) Runtime
    ax1 = axes[0]
    for i, method in enumerate(CHUNKED_METHODS):
        offset = (i - 1) * width
        bars = ax1.bar(x + offset, runtime_data[method], width,
                       label=METHOD_LABELS[method], color=COLORS[method],
                       alpha=0.8, edgecolor="black", linewidth=1.0)
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2.0, height,
                        f'{height:.2f}s', ha='center', va='bottom', fontsize=8)
    ax1.set_ylabel("Runtime (s)", fontweight="bold")
    ax1.set_title("(a) Runtime Comparison", fontweight="bold", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels([DATASET_LABELS[d] for d in DATASETS])
    ax1.legend(frameon=True, shadow=True)
    ax1.grid(axis="y", alpha=0.3)
    
    # (b) PPL
    ax2 = axes[1]
    for i, method in enumerate(CHUNKED_METHODS):
        offset = (i - 1) * width
        bars = ax2.bar(x + offset, ppl_data[method], width,
                       label=METHOD_LABELS[method], color=COLORS[method],
                       alpha=0.8, edgecolor="black", linewidth=1.0)
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2.0, height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    ax2.set_ylabel("Perplexity", fontweight="bold")
    ax2.set_title("(b) Perplexity Comparison", fontweight="bold", pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels([DATASET_LABELS[d] for d in DATASETS])
    ax2.legend(frameon=True, shadow=True)
    ax2.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "comprehensive_chunked_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ 已保存: {output_path}")


def plot_all_methods_comparison():
    """Generate comparison including all methods and evaluation types."""
    print("\n生成全方法对比图...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Complete Evaluation Comparison (All Methods & Types)", 
                 fontsize=16, fontweight="bold", y=0.995)
    
    x = np.arange(len(DATASETS))
    
    # Collect all data
    all_data = {}
    eval_types = ["decode_loop", "chunked"]
    
    for dataset in DATASETS:
        all_data[dataset] = {}
        
        # Decode-loop methods
        for method in DECODE_LOOP_METHODS:
            try:
                data = load_result(dataset, method, "decode_loop")
                all_data[dataset][f"{method}_decode"] = {
                    "runtime": data["total_time"],
                    "ppl": data["perplexity"],
                    "label": f"{METHOD_LABELS[method]}\n(decode-loop)"
                }
            except FileNotFoundError:
                pass
        
        # Chunked methods
        for method in CHUNKED_METHODS:
            try:
                data = load_result(dataset, method, "chunked")
                if method == "baseline":
                    runtime = data.get("baseline", {}).get("runtime_sec", 0)
                    ppl = data.get("baseline", {}).get("perplexity", 0)
                else:
                    runtime = data.get("streaming", {}).get("runtime_sec", 0)
                    ppl = data.get("streaming", {}).get("perplexity", 0)
                all_data[dataset][f"{method}_chunked"] = {
                    "runtime": runtime,
                    "ppl": ppl,
                    "label": f"{METHOD_LABELS[method]}\n(chunked)"
                }
            except (FileNotFoundError, KeyError):
                pass
        
        # KVPress official
        try:
            data = load_result(dataset, "kvpress", "kvpress_official")
            all_data[dataset]["kvpress_official"] = {
                "runtime": data.get("streaming", {}).get("runtime_sec", 0),
                "ppl": data.get("streaming", {}).get("perplexity", 0),
                "label": "KVPress\n(official)"
            }
        except (FileNotFoundError, KeyError):
            pass
    
    # Plot decode-loop only (cleaner comparison)
    methods = ["baseline_decode", "ours_decode", "mit_decode"]
    width = 0.25
    
    # (a) Decode-loop Runtime
    ax1 = axes[0, 0]
    for i, method_key in enumerate(methods):
        runtimes = [all_data[d].get(method_key, {}).get("runtime", 0) for d in DATASETS]
        offset = (i - 1) * width
        method_name = method_key.split("_")[0]
        bars = ax1.bar(x + offset, runtimes, width,
                       label=METHOD_LABELS[method_name], color=COLORS[method_name],
                       alpha=0.8, edgecolor="black", linewidth=1.0)
    ax1.set_ylabel("Runtime (s)", fontweight="bold")
    ax1.set_title("(a) Decode-Loop Runtime", fontweight="bold", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels([DATASET_LABELS[d] for d in DATASETS])
    ax1.legend(frameon=True, shadow=True, fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    
    # (b) Decode-loop PPL
    ax2 = axes[0, 1]
    for i, method_key in enumerate(methods):
        ppls = [all_data[d].get(method_key, {}).get("ppl", 0) for d in DATASETS]
        offset = (i - 1) * width
        method_name = method_key.split("_")[0]
        bars = ax2.bar(x + offset, ppls, width,
                       label=METHOD_LABELS[method_name], color=COLORS[method_name],
                       alpha=0.8, edgecolor="black", linewidth=1.0)
    ax2.set_ylabel("Perplexity", fontweight="bold")
    ax2.set_title("(b) Decode-Loop PPL", fontweight="bold", pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels([DATASET_LABELS[d] for d in DATASETS])
    ax2.legend(frameon=True, shadow=True, fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    
    # (c) Chunked Runtime
    ax3 = axes[1, 0]
    chunked_methods = ["baseline_chunked", "ours_chunked", "mit_chunked"]
    for i, method_key in enumerate(chunked_methods):
        runtimes = [all_data[d].get(method_key, {}).get("runtime", 0) for d in DATASETS]
        offset = (i - 1) * width
        method_name = method_key.split("_")[0]
        bars = ax3.bar(x + offset, runtimes, width,
                       label=METHOD_LABELS[method_name], color=COLORS[method_name],
                       alpha=0.8, edgecolor="black", linewidth=1.0)
    ax3.set_ylabel("Runtime (s)", fontweight="bold")
    ax3.set_title("(c) Chunked Runtime", fontweight="bold", pad=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels([DATASET_LABELS[d] for d in DATASETS])
    ax3.legend(frameon=True, shadow=True, fontsize=9)
    ax3.grid(axis="y", alpha=0.3)
    
    # (d) Chunked PPL
    ax4 = axes[1, 1]
    for i, method_key in enumerate(chunked_methods):
        ppls = [all_data[d].get(method_key, {}).get("ppl", 0) for d in DATASETS]
        offset = (i - 1) * width
        method_name = method_key.split("_")[0]
        bars = ax4.bar(x + offset, ppls, width,
                       label=METHOD_LABELS[method_name], color=COLORS[method_name],
                       alpha=0.8, edgecolor="black", linewidth=1.0)
    ax4.set_ylabel("Perplexity", fontweight="bold")
    ax4.set_title("(d) Chunked PPL", fontweight="bold", pad=10)
    ax4.set_xticks(x)
    ax4.set_xticklabels([DATASET_LABELS[d] for d in DATASETS])
    ax4.legend(frameon=True, shadow=True, fontsize=9)
    ax4.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "comprehensive_all_methods.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ 已保存: {output_path}")


def main():
    """Main function."""
    print("=" * 60)
    print("Comprehensive Results Visualization")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        plot_decode_loop_comparison()
        plot_chunked_comparison()
        plot_all_methods_comparison()
        
        print("\n" + "=" * 60)
        print("✓ 所有comprehensive图表生成完成!")
        print(f"✓ 保存位置: {OUTPUT_DIR}")
        print("=" * 60)
        
        # List generated files
        print("\n生成的图表文件:")
        for fig_file in sorted(OUTPUT_DIR.glob("comprehensive_*.png")):
            size_kb = fig_file.stat().st_size / 1024
            print(f"  - {fig_file.name} ({size_kb:.1f} KB)")
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
