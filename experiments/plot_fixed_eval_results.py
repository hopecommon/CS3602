#!/usr/bin/env python3
"""
Generate comparison plots for fixed evaluation results.

Reads JSON files from results/fixed_eval/ and outputs runtime / PPL bar charts.
Supports both decode-loop methods (baseline, ours, mit) and kvpress official results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Configuration
RESULTS_DIR = Path("results/fixed_eval")
OUTPUT_DIR = Path("results/figures")
DATASETS = ["wikitext", "pg19_20k"]
DATASET_LABELS = {"wikitext": "WikiText-103\n(4k tokens)", "pg19_20k": "PG19\n(20k tokens)"}
METHODS = ["baseline", "ours", "mit"]
METHOD_LABELS = {"baseline": "Baseline", "ours": "Ours", "mit": "MIT-style"}

# Color scheme
COLORS = {
    "baseline": "#2E86AB",  # Blue
    "ours": "#A23B72",      # Purple
    "mit": "#F18F01",       # Orange
    "kvpress": "#06A77D",   # Green
}


def load_result(dataset: str, method: str) -> Dict:
    """Load result JSON file."""
    path = RESULTS_DIR / f"{dataset}_{method}.json"
    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def plot_runtime_comparison():
    """Generate runtime comparison bar chart."""
    print("\n生成 Runtime 对比图...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(DATASETS))
    width = 0.25
    
    for i, method in enumerate(METHODS):
        runtimes = []
        for dataset in DATASETS:
            data = load_result(dataset, method)
            runtimes.append(data["total_time"])
        
        offset = (i - 1) * width
        bars = ax.bar(
            x + offset, 
            runtimes, 
            width, 
            label=METHOD_LABELS[method],
            color=COLORS[method],
            alpha=0.8,
            edgecolor="black",
            linewidth=1.2
        )
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}s",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold"
            )
    
    ax.set_xlabel("Dataset", fontweight="bold", fontsize=12)
    ax.set_ylabel("Runtime (seconds)", fontweight="bold", fontsize=12)
    ax.set_title("Decode-Loop Runtime Comparison", fontweight="bold", fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS])
    ax.legend(frameon=True, shadow=True, fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "fixed_eval_runtime_comparison.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ 已保存: {output_path}")


def plot_ppl_comparison():
    """Generate PPL comparison bar chart."""
    print("\n生成 PPL 对比图...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(DATASETS))
    width = 0.25
    
    for i, method in enumerate(METHODS):
        ppls = []
        for dataset in DATASETS:
            data = load_result(dataset, method)
            ppls.append(data["perplexity"])
        
        offset = (i - 1) * width
        bars = ax.bar(
            x + offset,
            ppls,
            width,
            label=METHOD_LABELS[method],
            color=COLORS[method],
            alpha=0.8,
            edgecolor="black",
            linewidth=1.2
        )
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold"
            )
    
    ax.set_xlabel("Dataset", fontweight="bold", fontsize=12)
    ax.set_ylabel("Perplexity", fontweight="bold", fontsize=12)
    ax.set_title("Decode-Loop Perplexity Comparison", fontweight="bold", fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS])
    ax.legend(frameon=True, shadow=True, fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "fixed_eval_ppl_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ 已保存: {output_path}")


def plot_speedup():
    """Generate speedup bar chart."""
    print("\n生成加速比对比图...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(DATASETS))
    width = 0.35
    
    speedups_ours = []
    speedups_mit = []
    
    for dataset in DATASETS:
        baseline_data = load_result(dataset, "baseline")
        ours_data = load_result(dataset, "ours")
        mit_data = load_result(dataset, "mit")
        
        baseline_time = baseline_data["total_time"]
        speedups_ours.append(baseline_time / ours_data["total_time"])
        speedups_mit.append(baseline_time / mit_data["total_time"])
    
    bars1 = ax.bar(
        x - width/2,
        speedups_ours,
        width,
        label="Ours",
        color=COLORS["ours"],
        alpha=0.8,
        edgecolor="black",
        linewidth=1.2
    )
    
    bars2 = ax.bar(
        x + width/2,
        speedups_mit,
        width,
        label="MIT-style",
        color=COLORS["mit"],
        alpha=0.8,
        edgecolor="black",
        linewidth=1.2
    )
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}×",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold"
            )
    
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Baseline (1.0×)")
    ax.set_xlabel("Dataset", fontweight="bold", fontsize=12)
    ax.set_ylabel("Speedup (×)", fontweight="bold", fontsize=12)
    ax.set_title("Speedup vs Baseline", fontweight="bold", fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS])
    ax.legend(frameon=True, shadow=True, fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "fixed_eval_speedup.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ 已保存: {output_path}")


def plot_ppl_increase():
    """Generate PPL increase percentage bar chart."""
    print("\n生成 PPL 增幅对比图...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(DATASETS))
    width = 0.35
    
    ppl_increases_ours = []
    ppl_increases_mit = []
    
    for dataset in DATASETS:
        baseline_data = load_result(dataset, "baseline")
        ours_data = load_result(dataset, "ours")
        mit_data = load_result(dataset, "mit")
        
        baseline_ppl = baseline_data["perplexity"]
        ppl_increases_ours.append(((ours_data["perplexity"] - baseline_ppl) / baseline_ppl) * 100)
        ppl_increases_mit.append(((mit_data["perplexity"] - baseline_ppl) / baseline_ppl) * 100)
    
    bars1 = ax.bar(
        x - width/2,
        ppl_increases_ours,
        width,
        label="Ours",
        color=COLORS["ours"],
        alpha=0.8,
        edgecolor="black",
        linewidth=1.2
    )
    
    bars2 = ax.bar(
        x + width/2,
        ppl_increases_mit,
        width,
        label="MIT-style",
        color=COLORS["mit"],
        alpha=0.8,
        edgecolor="black",
        linewidth=1.2
    )
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:+.2f}%",
                ha="center",
                va="bottom" if height > 0 else "top",
                fontsize=10,
                fontweight="bold"
            )
    
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1.0, alpha=0.5)
    ax.axhline(y=1, color="orange", linestyle="--", linewidth=1.5, alpha=0.7, label="Target (<1%)")
    ax.axhline(y=3, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Acceptable (<3%)")
    
    ax.set_xlabel("Dataset", fontweight="bold", fontsize=12)
    ax.set_ylabel("PPL Increase (%)", fontweight="bold", fontsize=12)
    ax.set_title("PPL Increase vs Baseline", fontweight="bold", fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS])
    ax.legend(frameon=True, shadow=True, fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "fixed_eval_ppl_increase.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ 已保存: {output_path}")


def plot_first_token_latency():
    """Generate first-token latency comparison."""
    print("\n生成 First-token Latency 对比图...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(DATASETS))
    width = 0.25
    
    for i, method in enumerate(METHODS):
        latencies = []
        for dataset in DATASETS:
            data = load_result(dataset, method)
            latencies.append(data["first_token_latency_sec"] * 1000)  # Convert to ms
        
        offset = (i - 1) * width
        bars = ax.bar(
            x + offset,
            latencies,
            width,
            label=METHOD_LABELS[method],
            color=COLORS[method],
            alpha=0.8,
            edgecolor="black",
            linewidth=1.2
        )
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}ms",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold"
            )
    
    ax.set_xlabel("Dataset", fontweight="bold", fontsize=12)
    ax.set_ylabel("First-token Latency (ms)", fontweight="bold", fontsize=12)
    ax.set_title("First-token Latency Comparison", fontweight="bold", fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS])
    ax.legend(frameon=True, shadow=True, fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "fixed_eval_first_token_latency.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ 已保存: {output_path}")


def plot_comprehensive_summary():
    """Generate comprehensive 2x2 summary plot."""
    print("\n生成综合对比图 (2×2)...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("StreamingLLM Fixed Evaluation Results (pythia-2.8b)", 
                 fontsize=16, fontweight="bold", y=0.995)
    
    x = np.arange(len(DATASETS))
    width = 0.25
    
    # Collect data
    data_by_method = {method: {"runtime": [], "ppl": [], "speedup": [], "ppl_increase": []} 
                      for method in METHODS}
    
    for dataset in DATASETS:
        baseline_data = load_result(dataset, "baseline")
        baseline_time = baseline_data["total_time"]
        baseline_ppl = baseline_data["perplexity"]
        
        for method in METHODS:
            method_data = load_result(dataset, method)
            data_by_method[method]["runtime"].append(method_data["total_time"])
            data_by_method[method]["ppl"].append(method_data["perplexity"])
            
            if method != "baseline":
                speedup = baseline_time / method_data["total_time"]
                ppl_increase = ((method_data["perplexity"] - baseline_ppl) / baseline_ppl) * 100
                data_by_method[method]["speedup"].append(speedup)
                data_by_method[method]["ppl_increase"].append(ppl_increase)
            else:
                data_by_method[method]["speedup"].append(1.0)
                data_by_method[method]["ppl_increase"].append(0.0)
    
    # (a) Runtime
    ax1 = axes[0, 0]
    for i, method in enumerate(METHODS):
        offset = (i - 1) * width
        ax1.bar(x + offset, data_by_method[method]["runtime"], width,
                label=METHOD_LABELS[method], color=COLORS[method], alpha=0.8,
                edgecolor="black", linewidth=1.0)
    ax1.set_ylabel("Runtime (s)", fontweight="bold")
    ax1.set_title("(a) Runtime Comparison", fontweight="bold", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels([DATASET_LABELS[d] for d in DATASETS])
    ax1.legend(frameon=True, shadow=True, fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    
    # (b) PPL
    ax2 = axes[0, 1]
    for i, method in enumerate(METHODS):
        offset = (i - 1) * width
        ax2.bar(x + offset, data_by_method[method]["ppl"], width,
                label=METHOD_LABELS[method], color=COLORS[method], alpha=0.8,
                edgecolor="black", linewidth=1.0)
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
        bars = ax3.bar(x + offset, data_by_method[method]["speedup"], width,
                       label=METHOD_LABELS[method], color=COLORS[method], alpha=0.8,
                       edgecolor="black", linewidth=1.0)
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2.0, height,
                    f"{height:.2f}×", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax3.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
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
        bars = ax4.bar(x + offset, data_by_method[method]["ppl_increase"], width,
                       label=METHOD_LABELS[method], color=COLORS[method], alpha=0.8,
                       edgecolor="black", linewidth=1.0)
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2.0, height,
                    f"{height:+.2f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax4.axhline(y=0, color="black", linestyle="-", linewidth=1.0, alpha=0.5)
    ax4.axhline(y=1, color="orange", linestyle="--", linewidth=1.5, alpha=0.7, label="Target (<1%)")
    ax4.axhline(y=3, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Acceptable (<3%)")
    ax4.set_ylabel("PPL Increase (%)", fontweight="bold")
    ax4.set_title("(d) PPL Increase vs Baseline", fontweight="bold", pad=10)
    ax4.set_xticks(x)
    ax4.set_xticklabels([DATASET_LABELS[d] for d in DATASETS])
    ax4.legend(frameon=True, shadow=True, fontsize=8)
    ax4.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "fixed_eval_comprehensive_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ 已保存: {output_path}")


def main():
    """Main function."""
    print("=" * 60)
    print("StreamingLLM Fixed Evaluation Results Visualization")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        plot_runtime_comparison()
        plot_ppl_comparison()
        plot_speedup()
        plot_ppl_increase()
        plot_first_token_latency()
        plot_comprehensive_summary()
        
        print("\n" + "=" * 60)
        print("✓ 所有图表生成完成!")
        print(f"✓ 保存位置: {OUTPUT_DIR}")
        print("=" * 60)
        
        # List generated files
        print("\n生成的图表文件:")
        for fig_file in sorted(OUTPUT_DIR.glob("fixed_eval_*.png")):
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
