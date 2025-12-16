#!/usr/bin/env python3
"""
绘制消融实验结果

从 results/ablation/ 读取消融实验结果并生成可视化图表
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# 配置
RESULTS_DIR = Path("results/ablation")
OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 颜色方案
COLOR_PPL = "#2E86AB"
COLOR_RUNTIME = "#A23B72"
COLOR_COMPRESSION = "#F18F01"


def load_ablation_results(filename: str):
    """加载消融实验结果"""
    path = RESULTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"结果文件不存在: {path}")
    with open(path, "r") as f:
        return json.load(f)


def plot_window_size_ablation():
    """绘制 Window Size 消融实验结果"""
    print("\n生成 Window Size 消融实验图表...")
    
    data = load_ablation_results("ablation_window_size.json")
    results = data["results"]
    model_name = data.get("model", "unknown")
    
    # 提取数据
    window_sizes = [r["window_size"] for r in results]
    ppls = [r["perplexity"] for r in results]
    runtimes = [r["runtime_sec"] for r in results]
    compressions = [r["compression_ratio"] * 100 for r in results]
    
    # 创建图表 (2x2布局)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fixed_n_sink = results[0].get("n_sink", 4) if results else 4
    fig.suptitle(f"Window Size Ablation Study (n_sink={fixed_n_sink}, {model_name})", 
                 fontsize=16, fontweight="bold", y=0.995)
    
    # (a) PPL vs Window Size
    ax1 = axes[0, 0]
    ax1.plot(window_sizes, ppls, marker='o', linewidth=2.5, markersize=8,
             color=COLOR_PPL, label="Perplexity")
    ax1.set_xlabel("Window Size", fontweight="bold", fontsize=11)
    ax1.set_ylabel("Perplexity", fontweight="bold", fontsize=11)
    ax1.set_title("(a) PPL vs Window Size", fontweight="bold", pad=10)
    ax1.set_xscale("log", base=2)
    ax1.grid(True, alpha=0.3)
    ax1.legend(frameon=True, shadow=True)
    
    # 标注数值
    for x, y in zip(window_sizes, ppls):
        ax1.annotate(f'{y:.2f}', (x, y), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9)
    
    # (b) Runtime vs Window Size
    ax2 = axes[0, 1]
    ax2.plot(window_sizes, runtimes, marker='s', linewidth=2.5, markersize=8,
             color=COLOR_RUNTIME, label="Runtime")
    ax2.set_xlabel("Window Size", fontweight="bold", fontsize=11)
    ax2.set_ylabel("Runtime (s)", fontweight="bold", fontsize=11)
    ax2.set_title("(b) Runtime vs Window Size", fontweight="bold", pad=10)
    ax2.set_xscale("log", base=2)
    ax2.grid(True, alpha=0.3)
    ax2.legend(frameon=True, shadow=True)
    
    # 标注数值
    for x, y in zip(window_sizes, runtimes):
        ax2.annotate(f'{y:.1f}s', (x, y), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9)
    
    # (c) Compression Ratio vs Window Size
    ax3 = axes[1, 0]
    ax3.plot(window_sizes, compressions, marker='^', linewidth=2.5, markersize=8,
             color=COLOR_COMPRESSION, label="Compression Ratio")
    ax3.set_xlabel("Window Size", fontweight="bold", fontsize=11)
    ax3.set_ylabel("Compression Ratio (%)", fontweight="bold", fontsize=11)
    ax3.set_title("(c) Compression Ratio vs Window Size", fontweight="bold", pad=10)
    ax3.set_xscale("log", base=2)
    ax3.grid(True, alpha=0.3)
    ax3.legend(frameon=True, shadow=True)
    
    # 标注数值
    for x, y in zip(window_sizes, compressions):
        ax3.annotate(f'{y:.1f}%', (x, y), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9)
    
    # (d) PPL vs Runtime (Trade-off)
    ax4 = axes[1, 1]
    scatter = ax4.scatter(runtimes, ppls, s=200, c=window_sizes, 
                         cmap='viridis', alpha=0.7, edgecolors='black', linewidth=2)
    ax4.set_xlabel("Runtime (s)", fontweight="bold", fontsize=11)
    ax4.set_ylabel("Perplexity", fontweight="bold", fontsize=11)
    ax4.set_title("(d) PPL-Runtime Trade-off", fontweight="bold", pad=10)
    ax4.grid(True, alpha=0.3)
    
    # 添加colorbar
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label("Window Size", fontweight="bold")
    
    # 标注window size
    for ws, rt, ppl in zip(window_sizes, runtimes, ppls):
        ax4.annotate(f'W={ws}', (rt, ppl), textcoords="offset points",
                    xytext=(5, 5), ha='left', fontsize=8)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "ablation_window_size.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ 已保存: {output_path}")


def plot_n_sink_ablation():
    """绘制 N_sink 消融实验结果"""
    print("\n生成 N_sink 消融实验图表...")
    
    data = load_ablation_results("ablation_n_sink.json")
    results = data["results"]
    model_name = data.get("model", "unknown")
    
    # 提取数据
    n_sinks = [r["n_sink"] for r in results]
    ppls = [r["perplexity"] for r in results]
    runtimes = [r["runtime_sec"] for r in results]
    compressions = [r["compression_ratio"] * 100 for r in results]
    
    # 创建图表 (2x2布局)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fixed_window = results[0].get("window_size", 1024) if results else 1024
    fig.suptitle(f"N_sink Ablation Study (window_size={fixed_window}, {model_name})", 
                 fontsize=16, fontweight="bold", y=0.995)
    
    # (a) PPL vs N_sink
    ax1 = axes[0, 0]
    ax1.plot(n_sinks, ppls, marker='o', linewidth=2.5, markersize=8,
             color=COLOR_PPL, label="Perplexity")
    ax1.set_xlabel("N_sink", fontweight="bold", fontsize=11)
    ax1.set_ylabel("Perplexity", fontweight="bold", fontsize=11)
    ax1.set_title("(a) PPL vs N_sink", fontweight="bold", pad=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(frameon=True, shadow=True)
    ax1.set_xticks(n_sinks)
    
    # 标注数值
    for x, y in zip(n_sinks, ppls):
        ax1.annotate(f'{y:.2f}', (x, y), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9)
    
    # 高亮 n_sink=0 的异常值
    ax1.axhline(y=ppls[0], color='red', linestyle='--', alpha=0.5, linewidth=1.5,
                label=f'n_sink=0 (no sink): {ppls[0]:.2f}')
    ax1.legend(frameon=True, shadow=True, fontsize=9)
    
    # (b) Runtime vs N_sink
    ax2 = axes[0, 1]
    ax2.plot(n_sinks, runtimes, marker='s', linewidth=2.5, markersize=8,
             color=COLOR_RUNTIME, label="Runtime")
    ax2.set_xlabel("N_sink", fontweight="bold", fontsize=11)
    ax2.set_ylabel("Runtime (s)", fontweight="bold", fontsize=11)
    ax2.set_title("(b) Runtime vs N_sink", fontweight="bold", pad=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(frameon=True, shadow=True)
    ax2.set_xticks(n_sinks)
    
    # 标注数值
    for x, y in zip(n_sinks, runtimes):
        ax2.annotate(f'{y:.1f}s', (x, y), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9)
    
    # (c) PPL Improvement vs N_sink
    ax3 = axes[1, 0]
    baseline_ppl = ppls[0]  # n_sink=0
    ppl_improvements = [(baseline_ppl - ppl) / baseline_ppl * 100 for ppl in ppls]
    
    bars = ax3.bar(range(len(n_sinks)), ppl_improvements, 
                   color=[COLOR_PPL if x >= 0 else 'red' for x in ppl_improvements],
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.set_xlabel("N_sink", fontweight="bold", fontsize=11)
    ax3.set_ylabel("PPL Improvement (%)", fontweight="bold", fontsize=11)
    ax3.set_title("(c) PPL Improvement vs Baseline (n_sink=0)", fontweight="bold", pad=10)
    ax3.set_xticks(range(len(n_sinks)))
    ax3.set_xticklabels(n_sinks)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1.0, alpha=0.5)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 标注数值
    for i, (bar, val) in enumerate(zip(bars, ppl_improvements)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2.0, height,
                f'{val:+.1f}%', ha='center', va='bottom' if val >= 0 else 'top',
                fontsize=9, fontweight='bold')
    
    # (d) Summary Table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # 创建表格数据
    table_data = [
        ["N_sink", "PPL", "Runtime (s)", "PPL Improve"],
    ]
    for i, ns in enumerate(n_sinks):
        table_data.append([
            str(ns),
            f"{ppls[i]:.2f}",
            f"{runtimes[i]:.1f}",
            f"{ppl_improvements[i]:+.1f}%"
        ])
    
    # 绘制表格
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.2, 0.25, 0.3, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # 设置表头样式
    for i in range(4):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 高亮最佳值
    best_ppl_idx = ppls.index(min(ppls[1:]))  # 排除n_sink=0
    table[(best_ppl_idx + 1, 1)].set_facecolor('#90EE90')
    
    ax4.set_title("(d) Summary Table", fontweight="bold", pad=20, fontsize=12)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "ablation_n_sink.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ 已保存: {output_path}")


def plot_combined_summary():
    """绘制综合对比图"""
    print("\n生成综合对比图...")
    
    window_data = load_ablation_results("ablation_window_size.json")
    n_sink_data = load_ablation_results("ablation_n_sink.json")
    model_name = window_data.get("model", "unknown")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Ablation Study Summary ({model_name}, WikiText-103)", 
                 fontsize=14, fontweight="bold", y=1.02)
    
    # (a) Window Size 关键指标
    ax1 = axes[0]
    window_results = window_data["results"]
    ws_sizes = [r["window_size"] for r in window_results]
    ws_ppls = [r["perplexity"] for r in window_results]
    
    ax1_twin = ax1.twinx()
    
    line1 = ax1.plot(ws_sizes, ws_ppls, marker='o', linewidth=2.5, markersize=8,
                     color=COLOR_PPL, label="PPL")
    ax1.set_xlabel("Window Size", fontweight="bold", fontsize=11)
    ax1.set_ylabel("Perplexity", fontweight="bold", fontsize=11, color=COLOR_PPL)
    ax1.set_xscale("log", base=2)
    ax1.tick_params(axis='y', labelcolor=COLOR_PPL)
    ax1.grid(True, alpha=0.3)
    
    ws_runtimes = [r["runtime_sec"] for r in window_results]
    line2 = ax1_twin.plot(ws_sizes, ws_runtimes, marker='s', linewidth=2.5, markersize=8,
                          color=COLOR_RUNTIME, label="Runtime")
    ax1_twin.set_ylabel("Runtime (s)", fontweight="bold", fontsize=11, color=COLOR_RUNTIME)
    ax1_twin.tick_params(axis='y', labelcolor=COLOR_RUNTIME)
    
    ax1.set_title("(a) Window Size Impact", fontweight="bold", pad=10)
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', frameon=True, shadow=True)
    
    # (b) N_sink 关键指标
    ax2 = axes[1]
    n_sink_results = n_sink_data["results"]
    ns_values = [r["n_sink"] for r in n_sink_results]
    ns_ppls = [r["perplexity"] for r in n_sink_results]
    
    bars = ax2.bar(range(len(ns_values)), ns_ppls, 
                   color=[COLOR_PPL if ns > 0 else 'red' for ns in ns_values],
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel("N_sink", fontweight="bold", fontsize=11)
    ax2.set_ylabel("Perplexity", fontweight="bold", fontsize=11)
    ax2.set_title("(b) N_sink Impact", fontweight="bold", pad=10)
    ax2.set_xticks(range(len(ns_values)))
    ax2.set_xticklabels(ns_values)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 标注数值
    for bar, ppl in zip(bars, ns_ppls):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, height,
                f'{ppl:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 高亮n_sink=0
    bars[0].set_edgecolor('red')
    bars[0].set_linewidth(3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "ablation_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ 已保存: {output_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("消融实验结果可视化")
    print("=" * 60)
    
    try:
        plot_window_size_ablation()
        plot_n_sink_ablation()
        plot_combined_summary()
        
        print("\n" + "=" * 60)
        print("✓ 所有消融实验图表生成完成!")
        print(f"✓ 保存位置: {OUTPUT_DIR}")
        print("=" * 60)
        
        # 列出生成的文件
        print("\n生成的图表文件:")
        for fig_file in sorted(OUTPUT_DIR.glob("ablation_*.png")):
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
