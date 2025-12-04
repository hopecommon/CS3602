#!/usr/bin/env python3
"""
统一的可视化脚本 - 生成所有实验结果的专业图表

生成的图表:
1. main_comparison.png - 主实验对比图 (2x2布局)
2. ablation_window_size.png - Window Size消融图 (2x1布局)
3. ablation_n_sink.png - N_sink消融图 (2x1布局)
4. results_summary.png - 综合结果表格
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import rcParams

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置专业的学术风格 (不使用 seaborn)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '-'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# 统一配色方案 - 专业学术风格
COLORS = {
    'baseline': '#2E86AB',      # 深蓝色
    'streaming': '#A23B72',     # 紫红色
    'accent': '#F18F01',        # 橙色
    'success': '#06A77D',       # 绿色
    'neutral': '#6C757D',       # 灰色
}

# 图表保存配置
DPI = 300
FIGURE_DIR = Path("results/figures")
RESULTS_DIR = Path("results/streaming_llm")


def setup_figure_dir():
    """创建图表保存目录"""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ 图表保存目录: {FIGURE_DIR}")


def load_json(filepath: Path) -> Dict:
    """加载JSON文件"""
    with open(filepath, 'r') as f:
        return json.load(f)


def format_number(value: float, decimals: int = 2) -> str:
    """格式化数字显示"""
    return f"{value:.{decimals}f}"


def plot_main_comparison():
    """
    生成主实验对比图 (2x2布局)
    - 左上: PPL对比 (Baseline vs StreamingLLM, WikiText & PG19)
    - 右上: Runtime对比 (柱状图)
    - 左下: 加速比 (柱状图,显示数值)
    - 右下: 压缩比 (柱状图,显示百分比)
    """
    print("\n生成主实验对比图...")
    
    # 加载数据
    wikitext = load_json(RESULTS_DIR / "wikitext_result.json")
    pg19 = load_json(RESULTS_DIR / "pg19_result.json")
    
    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('StreamingLLM Performance Comparison', fontsize=16, fontweight='bold', y=0.995)
    
    # === 左上: PPL对比 ===
    ax1 = axes[0, 0]
    datasets = ['WikiText-103', 'PG19']
    baseline_ppls = [wikitext['baseline']['perplexity'], pg19['baseline']['perplexity']]
    streaming_ppls = [wikitext['streaming']['perplexity'], pg19['streaming']['perplexity']]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, baseline_ppls, width, label='Baseline', 
                    color=COLORS['baseline'], alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax1.bar(x + width/2, streaming_ppls, width, label='StreamingLLM',
                    color=COLORS['streaming'], alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax1.set_ylabel('Perplexity', fontweight='bold')
    ax1.set_title('(a) Perplexity Comparison', fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.legend(frameon=True, shadow=True)
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # === 右上: Runtime对比 ===
    ax2 = axes[0, 1]
    baseline_times = [wikitext['baseline']['runtime_sec'] * 1000, 
                      pg19['baseline']['runtime_sec'] * 1000]
    streaming_times = [wikitext['streaming']['runtime_sec'] * 1000,
                       pg19['streaming']['runtime_sec'] * 1000]
    
    bars1 = ax2.bar(x - width/2, baseline_times, width, label='Baseline',
                    color=COLORS['baseline'], alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax2.bar(x + width/2, streaming_times, width, label='StreamingLLM',
                    color=COLORS['streaming'], alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax2.set_ylabel('Runtime (ms)', fontweight='bold')
    ax2.set_title('(b) Runtime Comparison', fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets)
    ax2.legend(frameon=True, shadow=True)
    ax2.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # === 左下: 加速比 ===
    ax3 = axes[1, 0]
    speedups = [wikitext['metrics']['speedup'], pg19['metrics']['speedup']]
    
    bars = ax3.bar(datasets, speedups, color=COLORS['success'], 
                   alpha=0.8, edgecolor='black', linewidth=1.2)
    ax3.set_ylabel('Speedup (×)', fontweight='bold')
    ax3.set_title('(c) Speedup Factor', fontweight='bold', pad=10)
    ax3.grid(axis='y', alpha=0.3)
    ax3.axhline(y=1, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline')
    ax3.legend(frameon=True, shadow=True)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}×',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # === 右下: 压缩比 ===
    ax4 = axes[1, 1]
    compression_ratios = [wikitext['metrics']['compression_ratio'] * 100,
                          pg19['metrics']['compression_ratio'] * 100]
    
    bars = ax4.bar(datasets, compression_ratios, color=COLORS['accent'],
                   alpha=0.8, edgecolor='black', linewidth=1.2)
    ax4.set_ylabel('Compression Ratio (%)', fontweight='bold')
    ax4.set_title('(d) KV Cache Compression', fontweight='bold', pad=10)
    ax4.grid(axis='y', alpha=0.3)
    ax4.set_ylim([0, 100])
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_path = FIGURE_DIR / "main_comparison.png"
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"✓ 已保存: {output_path}")


def plot_ablation_window_size():
    """
    生成Window Size消融图 (2x1布局)
    - 上: PPL vs Window Size (折线图,标注最佳点)
    - 下: Runtime vs Window Size (折线图)
    """
    print("\n生成Window Size消融图...")
    
    # 加载数据
    data = load_json(RESULTS_DIR / "ablation_window_size.json")
    results = data['results']
    
    window_sizes = [r['window_size'] for r in results]
    ppls = [r['perplexity'] for r in results]
    runtimes = [r['runtime_sec'] * 1000 for r in results]  # 转换为ms
    compression_ratios = [r['compression_ratio'] * 100 for r in results]
    
    # 创建2x1子图
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Ablation Study: Window Size Impact', fontsize=16, fontweight='bold', y=0.995)
    
    # === 上: PPL vs Window Size ===
    ax1 = axes[0]
    line1 = ax1.plot(window_sizes, ppls, marker='o', linewidth=2.5, markersize=8,
                     color=COLORS['baseline'], label='Perplexity')
    ax1.set_xlabel('Window Size', fontweight='bold')
    ax1.set_ylabel('Perplexity', fontweight='bold', color=COLORS['baseline'])
    ax1.set_title('(a) Perplexity vs Window Size', fontweight='bold', pad=10)
    ax1.tick_params(axis='y', labelcolor=COLORS['baseline'])
    ax1.set_xscale('log', base=2)
    ax1.grid(True, alpha=0.3)
    
    # 标注最佳点 (最低PPL)
    best_idx = np.argmin(ppls)
    ax1.scatter([window_sizes[best_idx]], [ppls[best_idx]], 
               color='red', s=200, zorder=5, marker='*', 
               edgecolors='black', linewidths=2)
    ax1.annotate(f'Best: {window_sizes[best_idx]}\nPPL: {ppls[best_idx]:.2f}',
                xy=(window_sizes[best_idx], ppls[best_idx]),
                xytext=(20, 20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=2),
                fontsize=10, fontweight='bold')
    
    # 添加压缩比作为第二y轴
    ax1_twin = ax1.twinx()
    line2 = ax1_twin.plot(window_sizes, compression_ratios, marker='s', linewidth=2.5, 
                          markersize=8, color=COLORS['accent'], 
                          label='Compression Ratio', linestyle='--')
    ax1_twin.set_ylabel('Compression Ratio (%)', fontweight='bold', color=COLORS['accent'])
    ax1_twin.tick_params(axis='y', labelcolor=COLORS['accent'])
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', frameon=True, shadow=True)
    
    # === 下: Runtime vs Window Size ===
    ax2 = axes[1]
    ax2.plot(window_sizes, runtimes, marker='o', linewidth=2.5, markersize=8,
            color=COLORS['streaming'], label='Runtime')
    ax2.set_xlabel('Window Size', fontweight='bold')
    ax2.set_ylabel('Runtime (ms)', fontweight='bold')
    ax2.set_title('(b) Runtime vs Window Size', fontweight='bold', pad=10)
    ax2.set_xscale('log', base=2)
    ax2.grid(True, alpha=0.3)
    ax2.legend(frameon=True, shadow=True)
    
    # 标注数值
    for i, (ws, rt) in enumerate(zip(window_sizes, runtimes)):
        ax2.annotate(f'{rt:.1f}ms',
                    xy=(ws, rt), xytext=(0, 10),
                    textcoords='offset points', ha='center',
                    fontsize=8, bbox=dict(boxstyle='round,pad=0.3', 
                    facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    output_path = FIGURE_DIR / "ablation_window_size.png"
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"✓ 已保存: {output_path}")


def plot_ablation_n_sink():
    """
    生成N_sink消融图 (2x1布局)
    - 上: PPL vs N_sink (折线图,标注n_sink=0的恶化)
    - 下: Runtime vs N_sink (折线图)
    """
    print("\n生成N_sink消融图...")
    
    # 加载数据
    data = load_json(RESULTS_DIR / "ablation_n_sink.json")
    results = data['results']
    
    n_sinks = [r['n_sink'] for r in results]
    ppls = [r['perplexity'] for r in results]
    runtimes = [r['runtime_sec'] * 1000 for r in results]  # 转换为ms
    compression_ratios = [r['compression_ratio'] * 100 for r in results]
    
    # 创建2x1子图
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Ablation Study: Sink Tokens Impact', fontsize=16, fontweight='bold', y=0.995)
    
    # === 上: PPL vs N_sink ===
    ax1 = axes[0]
    line1 = ax1.plot(n_sinks, ppls, marker='o', linewidth=2.5, markersize=8,
                     color=COLORS['baseline'], label='Perplexity')
    ax1.set_xlabel('Number of Sink Tokens', fontweight='bold')
    ax1.set_ylabel('Perplexity', fontweight='bold', color=COLORS['baseline'])
    ax1.set_title('(a) Perplexity vs Sink Tokens', fontweight='bold', pad=10)
    ax1.tick_params(axis='y', labelcolor=COLORS['baseline'])
    ax1.grid(True, alpha=0.3)
    
    # 特别标注n_sink=0的点
    if n_sinks[0] == 0:
        ax1.scatter([n_sinks[0]], [ppls[0]], 
                   color='red', s=200, zorder=5, marker='X',
                   edgecolors='black', linewidths=2)
        ax1.annotate(f'n_sink=0\nPPL: {ppls[0]:.2f}\n(No sink tokens)',
                    xy=(n_sinks[0], ppls[0]),
                    xytext=(30, -30), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', lw=2),
                    fontsize=10, fontweight='bold')
    
    # 添加压缩比作为第二y轴
    ax1_twin = ax1.twinx()
    line2 = ax1_twin.plot(n_sinks, compression_ratios, marker='s', linewidth=2.5,
                          markersize=8, color=COLORS['accent'],
                          label='Compression Ratio', linestyle='--')
    ax1_twin.set_ylabel('Compression Ratio (%)', fontweight='bold', color=COLORS['accent'])
    ax1_twin.tick_params(axis='y', labelcolor=COLORS['accent'])
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', frameon=True, shadow=True)
    
    # === 下: Runtime vs N_sink ===
    ax2 = axes[1]
    ax2.plot(n_sinks, runtimes, marker='o', linewidth=2.5, markersize=8,
            color=COLORS['streaming'], label='Runtime')
    ax2.set_xlabel('Number of Sink Tokens', fontweight='bold')
    ax2.set_ylabel('Runtime (ms)', fontweight='bold')
    ax2.set_title('(b) Runtime vs Sink Tokens', fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(frameon=True, shadow=True)
    
    # 标注数值
    for i, (ns, rt) in enumerate(zip(n_sinks, runtimes)):
        ax2.annotate(f'{rt:.1f}ms',
                    xy=(ns, rt), xytext=(0, 10),
                    textcoords='offset points', ha='center',
                    fontsize=8, bbox=dict(boxstyle='round,pad=0.3',
                    facecolor='white', alpha=0.7))
    
    # 特别标注n_sink=0的高runtime
    if n_sinks[0] == 0:
        ax2.scatter([n_sinks[0]], [runtimes[0]],
                   color='red', s=200, zorder=5, marker='X',
                   edgecolors='black', linewidths=2)
    
    plt.tight_layout()
    output_path = FIGURE_DIR / "ablation_n_sink.png"
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"✓ 已保存: {output_path}")


def plot_results_summary():
    """
    生成综合结果表格
    使用matplotlib生成专业的表格图
    """
    print("\n生成综合结果表格...")
    
    # 加载数据
    wikitext = load_json(RESULTS_DIR / "wikitext_result.json")
    pg19 = load_json(RESULTS_DIR / "pg19_result.json")
    
    # 准备表格数据
    headers = ['Metric', 'WikiText-103\nBaseline', 'WikiText-103\nStreaming', 
               'PG19\nBaseline', 'PG19\nStreaming']
    
    data = [
        ['Perplexity', 
         f"{wikitext['baseline']['perplexity']:.2f}",
         f"{wikitext['streaming']['perplexity']:.2f}",
         f"{pg19['baseline']['perplexity']:.2f}",
         f"{pg19['streaming']['perplexity']:.2f}"],
        ['Runtime (ms)',
         f"{wikitext['baseline']['runtime_sec']*1000:.2f}",
         f"{wikitext['streaming']['runtime_sec']*1000:.2f}",
         f"{pg19['baseline']['runtime_sec']*1000:.2f}",
         f"{pg19['streaming']['runtime_sec']*1000:.2f}"],
        ['Speedup',
         '1.00×',
         f"{wikitext['metrics']['speedup']:.2f}×",
         '1.00×',
         f"{pg19['metrics']['speedup']:.2f}×"],
        ['Compression (%)',
         '0.00%',
         f"{wikitext['metrics']['compression_ratio']*100:.2f}%",
         '0.00%',
         f"{pg19['metrics']['compression_ratio']*100:.2f}%"],
        ['PPL Increase',
         '-',
         f"{wikitext['metrics']['ppl_increase_percent']:.2f}%",
         '-',
         f"{pg19['metrics']['ppl_increase_percent']:.2f}%"],
        ['Total Tokens',
         str(wikitext['total_tokens']),
         str(wikitext['total_tokens']),
         str(pg19['total_tokens']),
         str(pg19['total_tokens'])],
    ]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # 创建表格
    table = ax.table(cellText=data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # 设置表头样式
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor(COLORS['baseline'])
        cell.set_text_props(weight='bold', color='white')
    
    # 设置行样式 - 交替颜色
    for i in range(1, len(data) + 1):
        for j in range(len(headers)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#f0f0f0')
            else:
                cell.set_facecolor('white')
            
            # 第一列加粗
            if j == 0:
                cell.set_text_props(weight='bold')
            
            # StreamingLLM列使用不同颜色
            if 'Streaming' in headers[j]:
                cell.set_facecolor('#e8f4f8')
    
    # 添加标题
    plt.title('StreamingLLM Comprehensive Results Summary', 
             fontsize=16, fontweight='bold', pad=20)
    
    # 添加配置信息
    config_text = (f"Model: {wikitext['model']}\n"
                  f"Window Size: {wikitext['streaming_llm']['window_size']}, "
                  f"Sink Tokens: {wikitext['streaming_llm']['n_sink']}\n"
                  f"Device: {wikitext['device']}, Dtype: {wikitext['dtype']}")
    
    plt.figtext(0.5, 0.05, config_text, ha='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    output_path = FIGURE_DIR / "results_summary.png"
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"✓ 已保存: {output_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("StreamingLLM 实验结果可视化")
    print("=" * 60)
    
    # 创建输出目录
    setup_figure_dir()
    
    # 生成所有图表
    try:
        plot_main_comparison()
        plot_ablation_window_size()
        plot_ablation_n_sink()
        plot_results_summary()
        
        print("\n" + "=" * 60)
        print("✓ 所有图表生成完成!")
        print(f"✓ 保存位置: {FIGURE_DIR}")
        print("=" * 60)
        
        # 列出生成的文件
        print("\n生成的图表文件:")
        for fig_file in sorted(FIGURE_DIR.glob("*.png")):
            size_mb = fig_file.stat().st_size / (1024 * 1024)
            print(f"  - {fig_file.name} ({size_mb:.2f} MB)")
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())