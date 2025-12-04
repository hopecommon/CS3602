#!/usr/bin/env python3
"""
Plot Comparison Results

Generate comparison plots between our implementation and kvpress implementation.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Use default matplotlib settings
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_comparison_results(result_dir: Path) -> Dict[str, Any]:
    """Load comparison results from both implementations"""
    results = {}
    
    # Load our implementation results
    our_files = list(result_dir.glob("streaming_llm/*_comparison.json"))
    for file in our_files:
        dataset = file.stem.replace("_comparison", "")
        with open(file) as f:
            results[f"ours_{dataset}"] = json.load(f)
    
    # Load kvpress results
    kvpress_files = list(result_dir.glob("kvpress/*_comparison.json"))
    for file in kvpress_files:
        dataset = file.stem.replace("_comparison", "")
        with open(file) as f:
            results[f"kvpress_{dataset}"] = json.load(f)
    
    return results


def plot_side_by_side_comparison(results: Dict[str, Any], output_dir: Path):
    """Plot side-by-side comparison of our implementation vs kvpress"""
    
    # Extract datasets
    datasets = set()
    for key in results.keys():
        if key.startswith("ours_"):
            datasets.add(key.replace("ours_", ""))
    
    if not datasets:
        print("Warning: No comparison results found")
        return
    
    datasets = sorted(list(datasets))
    n_datasets = len(datasets)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Our Implementation vs kvpress StreamingLLM', fontsize=18, fontweight='bold')
    
    # Prepare data
    our_ppls = []
    kvpress_ppls = []
    our_times = []
    kvpress_times = []
    baseline_ppls = []
    baseline_times = []
    labels = []
    
    for dataset in datasets:
        our_key = f"ours_{dataset}"
        kvpress_key = f"kvpress_{dataset}"
        
        if our_key in results and kvpress_key in results:
            our_data = results[our_key]
            kvpress_data = results[kvpress_key]
            
            labels.append(dataset.upper())
            
            # PPL data
            baseline_ppls.append(our_data["baseline"]["perplexity"])
            our_ppls.append(our_data["streaming"]["perplexity"])
            kvpress_ppls.append(kvpress_data["kvpress"]["perplexity"])
            
            # Time data
            baseline_times.append(our_data["baseline"]["runtime_sec"])
            our_times.append(our_data["streaming"]["runtime_sec"])
            kvpress_times.append(kvpress_data["kvpress"]["runtime_sec"])
    
    if not labels:
        print("Warning: No matching comparison data found")
        return
    
    x = np.arange(len(labels))
    width = 0.25
    
    # 1. PPL Comparison
    ax1 = axes[0, 0]
    ax1.bar(x - width, baseline_ppls, width, label='Baseline', color='gray', alpha=0.7)
    ax1.bar(x, our_ppls, width, label='Our Implementation', color='blue', alpha=0.7)
    ax1.bar(x + width, kvpress_ppls, width, label='kvpress', color='green', alpha=0.7)
    ax1.set_ylabel('Perplexity (lower is better)', fontsize=12)
    ax1.set_title('PPL Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (b, o, k) in enumerate(zip(baseline_ppls, our_ppls, kvpress_ppls)):
        ax1.text(i - width, b, f'{b:.1f}', ha='center', va='bottom', fontsize=9)
        ax1.text(i, o, f'{o:.1f}', ha='center', va='bottom', fontsize=9)
        ax1.text(i + width, k, f'{k:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Runtime Comparison
    ax2 = axes[0, 1]
    ax2.bar(x - width, baseline_times, width, label='Baseline', color='gray', alpha=0.7)
    ax2.bar(x, our_times, width, label='Our Implementation', color='blue', alpha=0.7)
    ax2.bar(x + width, kvpress_times, width, label='kvpress', color='green', alpha=0.7)
    ax2.set_ylabel('Runtime (s) (lower is better)', fontsize=12)
    ax2.set_title('Runtime Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (b, o, k) in enumerate(zip(baseline_times, our_times, kvpress_times)):
        ax2.text(i - width, b, f'{b:.1f}', ha='center', va='bottom', fontsize=9)
        ax2.text(i, o, f'{o:.1f}', ha='center', va='bottom', fontsize=9)
        ax2.text(i + width, k, f'{k:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Speedup Comparison
    ax3 = axes[1, 0]
    our_speedups = [b/o if o > 0 else 0 for b, o in zip(baseline_times, our_times)]
    kvpress_speedups = [b/k if k > 0 else 0 for b, k in zip(baseline_times, kvpress_times)]
    
    ax3.bar(x - width/2, our_speedups, width, label='Our Implementation', color='blue', alpha=0.7)
    ax3.bar(x + width/2, kvpress_speedups, width, label='kvpress', color='green', alpha=0.7)
    ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Baseline (1.0x)')
    ax3.set_ylabel('Speedup (higher is better)', fontsize=12)
    ax3.set_title('Speedup Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (o, k) in enumerate(zip(our_speedups, kvpress_speedups)):
        ax3.text(i - width/2, o, f'{o:.2f}x', ha='center', va='bottom', fontsize=9)
        ax3.text(i + width/2, k, f'{k:.2f}x', ha='center', va='bottom', fontsize=9)
    
    # 4. PPL Increase Comparison
    ax4 = axes[1, 1]
    our_ppl_increases = [((o-b)/b)*100 for b, o in zip(baseline_ppls, our_ppls)]
    kvpress_ppl_increases = [((k-b)/b)*100 for b, k in zip(baseline_ppls, kvpress_ppls)]
    
    ax4.bar(x - width/2, our_ppl_increases, width, label='Our Implementation', color='blue', alpha=0.7)
    ax4.bar(x + width/2, kvpress_ppl_increases, width, label='kvpress', color='green', alpha=0.7)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='No increase')
    ax4.set_ylabel('PPL Increase (%)', fontsize=12)
    ax4.set_title('PPL Degradation Comparison', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (o, k) in enumerate(zip(our_ppl_increases, kvpress_ppl_increases)):
        ax4.text(i - width/2, o, f'{o:.1f}%', ha='center', va='bottom' if o > 0 else 'top', fontsize=9)
        ax4.text(i + width/2, k, f'{k:.1f}%', ha='center', va='bottom' if k > 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    
    output_file = output_dir / "implementation_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Implementation comparison plot saved: {output_file}")


def plot_detailed_metrics_table(results: Dict[str, Any], output_dir: Path):
    """Generate detailed metrics comparison table"""
    
    # Extract datasets
    datasets = set()
    for key in results.keys():
        if key.startswith("ours_"):
            datasets.add(key.replace("ours_", ""))
    
    if not datasets:
        print("Warning: No comparison results found")
        return
    
    datasets = sorted(list(datasets))
    
    # Prepare table data
    table_data = []
    headers = ['Dataset', 'Implementation', 'Baseline PPL', 'Compressed PPL', 
               'PPL Δ%', 'Baseline Time', 'Compressed Time', 'Speedup']
    
    for dataset in datasets:
        our_key = f"ours_{dataset}"
        kvpress_key = f"kvpress_{dataset}"
        
        if our_key in results:
            our_data = results[our_key]
            baseline_ppl = our_data["baseline"]["perplexity"]
            our_ppl = our_data["streaming"]["perplexity"]
            baseline_time = our_data["baseline"]["runtime_sec"]
            our_time = our_data["streaming"]["runtime_sec"]
            our_speedup = baseline_time / our_time if our_time > 0 else 0
            our_ppl_delta = ((our_ppl - baseline_ppl) / baseline_ppl) * 100
            
            table_data.append([
                dataset.upper(),
                'Ours',
                f'{baseline_ppl:.2f}',
                f'{our_ppl:.2f}',
                f'{our_ppl_delta:+.2f}%',
                f'{baseline_time:.2f}s',
                f'{our_time:.2f}s',
                f'{our_speedup:.2f}x'
            ])
        
        if kvpress_key in results:
            kvpress_data = results[kvpress_key]
            baseline_ppl = kvpress_data["baseline"]["perplexity"]
            kvpress_ppl = kvpress_data["kvpress"]["perplexity"]
            baseline_time = kvpress_data["baseline"]["runtime_sec"]
            kvpress_time = kvpress_data["kvpress"]["runtime_sec"]
            kvpress_speedup = baseline_time / kvpress_time if kvpress_time > 0 else 0
            kvpress_ppl_delta = ((kvpress_ppl - baseline_ppl) / baseline_ppl) * 100
            
            table_data.append([
                '',
                'kvpress',
                f'{baseline_ppl:.2f}',
                f'{kvpress_ppl:.2f}',
                f'{kvpress_ppl_delta:+.2f}%',
                f'{baseline_time:.2f}s',
                f'{kvpress_time:.2f}s',
                f'{kvpress_speedup:.2f}x'
            ])
    
    if not table_data:
        print("Warning: No data available for table generation")
        return
    
    # Create table
    fig, ax = plt.subplots(figsize=(14, len(table_data) * 0.5 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colWidths=[0.12, 0.12, 0.12, 0.12, 0.10, 0.12, 0.12, 0.10]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Set header style
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')
    
    # Set row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#E7E6E6')
            else:
                cell.set_facecolor('#F2F2F2')
    
    plt.title('Implementation Comparison - Detailed Metrics', 
              fontsize=14, fontweight='bold', pad=20)
    
    output_file = output_dir / "comparison_metrics_table.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comparison metrics table saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot comparison results between implementations"
    )
    
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Results directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures"),
        help="Output directory for plots"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Plotting Implementation Comparison Results")
    print(f"{'='*60}")
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}\n")
    
    # Load results
    print("Loading comparison results...")
    results = load_comparison_results(args.results_dir)
    
    if not results:
        print("Error: No comparison results found")
        return 1
    
    print(f"Found {len(results)} result files\n")
    
    # Generate plots
    print("Generating comparison plots...")
    plot_side_by_side_comparison(results, args.output_dir)
    plot_detailed_metrics_table(results, args.output_dir)
    
    print(f"\nAll comparison plots generated successfully!")
    print(f"{'='*60}")
    print(f"Output directory: {args.output_dir}")
    print(f"  - implementation_comparison.png: Side-by-side comparison")
    print(f"  - comparison_metrics_table.png: Detailed metrics table")
    print(f"{'='*60}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())