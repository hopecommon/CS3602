#!/usr/bin/env python3
"""
Plot experimental results

Generate visualization charts for PPL, speedup, memory usage, and other metrics
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Use default matplotlib settings (no Chinese font needed)
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_results(result_dir: Path) -> Dict[str, Any]:
    """Load all experimental results"""
    results = {}
    
    # Main experiment results
    wikitext_file = result_dir / "wikitext_result.json"
    pg19_file = result_dir / "pg19_result.json"
    
    if wikitext_file.exists():
        results['wikitext'] = json.loads(wikitext_file.read_text())
    
    if pg19_file.exists():
        results['pg19'] = json.loads(pg19_file.read_text())
    
    # Ablation study results
    ablation_window_file = result_dir / "ablation_window_size.json"
    ablation_nsink_file = result_dir / "ablation_n_sink.json"
    
    if ablation_window_file.exists():
        results['ablation_window'] = json.loads(ablation_window_file.read_text())
    
    if ablation_nsink_file.exists():
        results['ablation_nsink'] = json.loads(ablation_nsink_file.read_text())
    
    # Backend-specific files
    backend_results = {"main": {}, "ablation_window": {}, "ablation_nsink": {}}
    for path in result_dir.glob("*_backend-*.json"):
        name = path.stem
        if "wikitext_main_backend-" in name or "pg19_main_backend-" in name:
            data = json.loads(path.read_text())
            ds = "wikitext" if "wikitext" in name else "pg19"
            backend = name.split("backend-")[-1]
            backend_results["main"].setdefault(ds, {})[backend] = data
        elif "ablation_window_size_backend-" in name:
            backend = name.split("backend-")[-1]
            backend_results["ablation_window"][backend] = json.loads(path.read_text())
        elif "ablation_n_sink_backend-" in name:
            backend = name.split("backend-")[-1]
            backend_results["ablation_nsink"][backend] = json.loads(path.read_text())
    results["backend"] = backend_results
    
    return results


def plot_main_comparison(results: Dict[str, Any], output_dir: Path):
    """Plot main experiment comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('StreamingLLM vs Baseline Performance Comparison', fontsize=16, fontweight='bold')
    
    datasets = []
    baseline_ppls = []
    streaming_ppls = []
    speedups = []
    compression_ratios = []
    
    for dataset_name in ['wikitext', 'pg19']:
        if dataset_name in results:
            data = results[dataset_name]
            datasets.append(dataset_name.upper())
            baseline_ppls.append(data['baseline']['perplexity'])
            streaming_ppls.append(data['streaming']['perplexity'])
            speedups.append(data['metrics']['speedup'])
            compression_ratios.append(data['metrics']['compression_ratio'])
    
    if not datasets:
        print("Warning: No main experiment results found")
        return
    
    x = np.arange(len(datasets))
    width = 0.35
    
    # 1. PPL Comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x - width/2, baseline_ppls, width, label='Baseline', color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width/2, streaming_ppls, width, label='StreamingLLM', color='#e74c3c', alpha=0.8)
    ax1.set_ylabel('Perplexity (lower is better)', fontsize=12)
    ax1.set_title('PPL Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9)
    
    # 2. Speedup
    ax2 = axes[0, 1]
    bars = ax2.bar(datasets, speedups, color='#2ecc71', alpha=0.8)
    ax2.set_ylabel('Speedup (higher is better)', fontsize=12)
    ax2.set_title('Runtime Speedup', fontsize=13, fontweight='bold')
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}x',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. Compression Ratio
    ax3 = axes[1, 0]
    bars = ax3.bar(datasets, [r * 100 for r in compression_ratios], color='#9b59b6', alpha=0.8)
    ax3.set_ylabel('Compression Ratio (%)', fontsize=12)
    ax3.set_title('KV Cache Compression Ratio', fontsize=13, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. Runtime Comparison
    ax4 = axes[1, 1]
    baseline_times = [results[ds]['baseline']['runtime_sec'] for ds in ['wikitext', 'pg19'] if ds in results]
    streaming_times = [results[ds]['streaming']['runtime_sec'] for ds in ['wikitext', 'pg19'] if ds in results]
    
    x = np.arange(len(datasets))
    bars1 = ax4.bar(x - width/2, baseline_times, width, label='Baseline', color='#3498db', alpha=0.8)
    bars2 = ax4.bar(x + width/2, streaming_times, width, label='StreamingLLM', color='#e74c3c', alpha=0.8)
    ax4.set_ylabel('Runtime (s) (lower is better)', fontsize=12)
    ax4.set_title('Inference Time Comparison', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(datasets)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}s',
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_file = output_dir / "main_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Main comparison plot saved: {output_file}")
    plt.close()


def plot_ablation_window_size(results: Dict[str, Any], output_dir: Path):
    """Plot window_size ablation study"""
    if 'ablation_window' not in results:
        print("Warning: No window_size ablation results found")
        return
    
    data = results['ablation_window']['results']
    
    window_sizes = [r['window_size'] for r in data]
    ppls = [r['perplexity'] for r in data]
    runtimes = [r['runtime_sec'] for r in data]
    compression_ratios = [r['compression_ratio'] * 100 for r in data]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Window Size Ablation Study', fontsize=16, fontweight='bold')
    
    # 1. PPL vs Window Size
    ax1 = axes[0]
    ax1.plot(window_sizes, ppls, marker='o', linewidth=2, markersize=8, color='#e74c3c')
    ax1.set_xlabel('Window Size', fontsize=12)
    ax1.set_ylabel('Perplexity (lower is better)', fontsize=12)
    ax1.set_title('PPL vs Window Size', fontsize=13, fontweight='bold')
    ax1.set_xscale('log', base=2)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=min(ppls), color='green', linestyle='--', alpha=0.5, label=f'Best PPL: {min(ppls):.2f}')
    ax1.legend()
    
    # 2. Runtime vs Window Size
    ax2 = axes[1]
    ax2.plot(window_sizes, runtimes, marker='s', linewidth=2, markersize=8, color='#3498db')
    ax2.set_xlabel('Window Size', fontsize=12)
    ax2.set_ylabel('Runtime (s) (lower is better)', fontsize=12)
    ax2.set_title('Runtime vs Window Size', fontsize=13, fontweight='bold')
    ax2.set_xscale('log', base=2)
    ax2.grid(True, alpha=0.3)
    
    # 3. Compression Ratio vs Window Size
    ax3 = axes[2]
    ax3.plot(window_sizes, compression_ratios, marker='^', linewidth=2, markersize=8, color='#9b59b6')
    ax3.set_xlabel('Window Size', fontsize=12)
    ax3.set_ylabel('Compression Ratio (%)', fontsize=12)
    ax3.set_title('Compression Ratio vs Window Size', fontsize=13, fontweight='bold')
    ax3.set_xscale('log', base=2)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / "ablation_window_size.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Window size ablation plot saved: {output_file}")
    plt.close()


def plot_ablation_n_sink(results: Dict[str, Any], output_dir: Path):
    """Plot n_sink ablation study"""
    if 'ablation_nsink' not in results:
        print("Warning: No n_sink ablation results found")
        return
    
    data = results['ablation_nsink']['results']
    
    n_sinks = [r['n_sink'] for r in data]
    ppls = [r['perplexity'] for r in data]
    runtimes = [r['runtime_sec'] for r in data]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('N_sink Ablation Study (Verifying Attention Sink Hypothesis)', fontsize=16, fontweight='bold')
    
    # 1. PPL vs N_sink
    ax1 = axes[0]
    ax1.plot(n_sinks, ppls, marker='o', linewidth=2, markersize=10, color='#e74c3c')
    ax1.set_xlabel('N_sink (Number of Sink Tokens)', fontsize=12)
    ax1.set_ylabel('Perplexity (lower is better)', fontsize=12)
    ax1.set_title('PPL vs N_sink', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=4, color='green', linestyle='--', alpha=0.5, label='Recommended: n_sink=4')
    ax1.legend()
    
    # Annotate key points
    for i, (x, y) in enumerate(zip(n_sinks, ppls)):
        if x in [0, 4]:
            ax1.annotate(f'{y:.2f}', xy=(x, y), xytext=(5, 5), 
                        textcoords='offset points', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    # 2. Runtime vs N_sink
    ax2 = axes[1]
    ax2.plot(n_sinks, runtimes, marker='s', linewidth=2, markersize=10, color='#3498db')
    ax2.set_xlabel('N_sink (Number of Sink Tokens)', fontsize=12)
    ax2.set_ylabel('Runtime (s)', fontsize=12)
    ax2.set_title('Runtime vs N_sink', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / "ablation_n_sink.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ N_sink ablation plot saved: {output_file}")
    plt.close()


def plot_summary_table(results: Dict[str, Any], output_dir: Path):
    """Generate results summary table"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    headers = ['Dataset', 'Baseline PPL', 'StreamingLLM PPL', 'PPL Increase',
               'Baseline Time', 'StreamingLLM Time', 'Speedup', 'Compression']
    
    for dataset_name in ['wikitext', 'pg19']:
        if dataset_name in results:
            data = results[dataset_name]
            row = [
                dataset_name.upper(),
                f"{data['baseline']['perplexity']:.2f}",
                f"{data['streaming']['perplexity']:.2f}",
                f"{data['metrics']['ppl_increase_percent']:.2f}%",
                f"{data['baseline']['runtime_sec']:.3f}s",
                f"{data['streaming']['runtime_sec']:.3f}s",
                f"{data['metrics']['speedup']:.2f}x",
                f"{data['metrics']['compression_ratio']*100:.1f}%"
            ]
            table_data.append(row)
    
    if not table_data:
        print("Warning: No data available for table generation")
        return
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.1, 0.12, 0.15, 0.1, 0.12, 0.15, 0.08, 0.08])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Set header style
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Set row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
    
    plt.title('StreamingLLM Experimental Results Summary', fontsize=14, fontweight='bold', pad=20)
    
    output_file = output_dir / "results_summary_table.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Results summary table saved: {output_file}")
    plt.close()

def plot_backend_comparison(results: Dict[str, Any], output_dir: Path):
    backend = results.get("backend", {})
    main = backend.get("main", {})
    if not main:
        return
    datasets = [ds for ds in ["wikitext", "pg19"] if ds in main]
    if not datasets:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Backend Comparison (math vs flash)", fontsize=16, fontweight="bold")
    x = np.arange(len(datasets))
    width = 0.35
    runtimes_math = []
    runtimes_flash = []
    ppls_math = []
    ppls_flash = []
    for ds in datasets:
        math_data = main[ds].get("math")
        flash_data = main[ds].get("flash")
        if math_data:
            runtimes_math.append(math_data["streaming"]["runtime_sec"])
            ppls_math.append(math_data["streaming"]["perplexity"])
        else:
            runtimes_math.append(0.0)
            ppls_math.append(0.0)
        if flash_data:
            runtimes_flash.append(flash_data["streaming"]["runtime_sec"])
            ppls_flash.append(flash_data["streaming"]["perplexity"])
        else:
            runtimes_flash.append(0.0)
            ppls_flash.append(0.0)
    ax1 = axes[0]
    ax1.bar(x - width/2, runtimes_math, width, label="StreamingLLM (math)", color="#5555aa", alpha=0.85)
    ax1.bar(x + width/2, runtimes_flash, width, label="StreamingLLM (flash)", color="#aa5555", alpha=0.85)
    ax1.set_title("Runtime (s)", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([ds.upper() for ds in datasets])
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    ax2 = axes[1]
    ax2.bar(x - width/2, ppls_math, width, label="StreamingLLM (math)", color="#5555aa", alpha=0.85)
    ax2.bar(x + width/2, ppls_flash, width, label="StreamingLLM (flash)", color="#aa5555", alpha=0.85)
    ax2.set_title("Perplexity", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([ds.upper() for ds in datasets])
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    output_file = output_dir / "backend_math_vs_flash.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✓ Backend comparison plot saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot StreamingLLM experimental results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/streaming_llm"),
        help="Experimental results directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures"),
        help="Chart output directory"
    )
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Plotting StreamingLLM Experimental Results")
    print(f"{'='*60}")
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}\n")
    
    # Load results
    print("Loading experimental results...")
    results = load_results(args.results_dir)
    
    if not results:
        print("Error: No experimental results found")
        return 1
    
    print(f"Found {len(results)} result files\n")
    
    # Generate plots
    print("Generating plots...")
    plot_main_comparison(results, args.output_dir)
    plot_ablation_window_size(results, args.output_dir)
    plot_ablation_n_sink(results, args.output_dir)
    plot_summary_table(results, args.output_dir)
    plot_backend_comparison(results, args.output_dir)
    
    print(f"\n{'='*60}")
    print(f"All plots generated successfully!")
    print(f"{'='*60}")
    print(f"Output directory: {args.output_dir}")
    print(f"  - main_comparison.png: Main experiment comparison")
    print(f"  - ablation_window_size.png: Window size ablation")
    print(f"  - ablation_n_sink.png: N_sink ablation")
    print(f"  - results_summary_table.png: Results summary table")
    print(f"  - backend_math_vs_flash.png: Backend comparison")
    print(f"{'='*60}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
