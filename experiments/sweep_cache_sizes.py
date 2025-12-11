#!/usr/bin/env python3
"""
扫描不同 cache size 的 decoding latency

复现论文图 10 的实验设置
"""

import argparse
import os
import subprocess
import json
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="扫描不同 cache size 的 decoding latency"
    )
    
    parser.add_argument(
        "--cache-sizes",
        type=int,
        nargs="+",
        default=[256, 512, 1024, 2048, 4096],
        help="要测试的 cache size 列表"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=os.environ.get("MODEL_NAME", "EleutherAI/pythia-2.8b"),
        help="模型名称"
    )
    parser.add_argument(
        "--n-sink",
        type=int,
        default=4,
        help="Sink token 数量"
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=2000,
        help="生成的 token 数量"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="每个配置重复运行次数"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/decoding/cache_size_sweep"),
        help="输出目录"
    )
    
    return parser.parse_args()


def run_evaluation(
    cache_size: int,
    model_name: str,
    n_sink: int,
    num_tokens: int,
    num_runs: int,
    output_path: Path,
) -> Dict:
    """运行单个 cache size 的评估"""
    print(f"\n{'='*60}")
    print(f"评估 Cache Size = {cache_size}")
    print(f"{'='*60}")
    
    cmd = [
        "python", "experiments/eval_decoding_latency.py",
        "--model-name", model_name,
        "--cache-size", str(cache_size),
        "--n-sink", str(n_sink),
        "--num-tokens", str(num_tokens),
        "--num-runs", str(num_runs),
        "--output", str(output_path),
    ]
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"错误: Cache size {cache_size} 评估失败")
        return None
    
    # 读取结果
    if output_path.exists():
        with open(output_path, 'r') as f:
            return json.load(f)
    
    return None


def plot_results(results: List[Dict], output_dir: Path):
    """绘制结果图表"""
    cache_sizes = []
    baseline_latencies = []
    baseline_stds = []
    streaming_latencies = []
    streaming_stds = []
    speedups = []
    
    for result in results:
        if result is None:
            continue
        
        cache_sizes.append(result["config"]["cache_size"])
        baseline_latencies.append(result["baseline"]["mean_latency_ms"])
        baseline_stds.append(result["baseline"]["std_latency_ms"])
        streaming_latencies.append(result["streaming_llm"]["mean_latency_ms"])
        streaming_stds.append(result["streaming_llm"]["std_latency_ms"])
        speedups.append(result["comparison"]["speedup"])
    
    # 检查是否有有效数据
    if len(cache_sizes) == 0:
        print("警告: 没有有效的结果数据,跳过绘图")
        return
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 图 1: Latency vs Cache Size
    ax1.errorbar(
        cache_sizes, baseline_latencies, yerr=baseline_stds,
        marker='o', label='Baseline (Full KV Cache)', capsize=5
    )
    ax1.errorbar(
        cache_sizes, streaming_latencies, yerr=streaming_stds,
        marker='s', label='StreamingLLM', capsize=5
    )
    ax1.set_xlabel('Cache Size')
    ax1.set_ylabel('Per-Token Latency (ms)')
    ax1.set_title('Decoding Latency vs Cache Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    
    # 图 2: Speedup vs Cache Size
    ax2.plot(cache_sizes, speedups, marker='o', linewidth=2)
    ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='No speedup')
    ax2.set_xlabel('Cache Size')
    ax2.set_ylabel('Speedup (Baseline / StreamingLLM)')
    ax2.set_title('Speedup vs Cache Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = output_dir / "cache_size_sweep.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存到: {output_path}")
    
    # 也保存为 PDF
    pdf_path = output_dir / "cache_size_sweep.pdf"
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"PDF 已保存到: {pdf_path}")
    
    plt.close()


def print_summary_table(results: List[Dict]):
    """打印汇总表格"""
    print("\n" + "="*80)
    print("汇总表格")
    print("="*80)
    print(f"{'Cache Size':<12} {'Baseline (ms)':<18} {'StreamingLLM (ms)':<20} {'Speedup':<10}")
    print("-"*80)
    
    for result in results:
        if result is None:
            continue
        
        cache_size = result["config"]["cache_size"]
        baseline = result["baseline"]["mean_latency_ms"]
        baseline_std = result["baseline"]["std_latency_ms"]
        streaming = result["streaming_llm"]["mean_latency_ms"]
        streaming_std = result["streaming_llm"]["std_latency_ms"]
        speedup = result["comparison"]["speedup"]
        
        print(f"{cache_size:<12} {baseline:.3f}±{baseline_std:.3f}      "
              f"{streaming:.3f}±{streaming_std:.3f}        {speedup:.2f}x")
    
    print("="*80)


def main():
    args = parse_args()
    
    # 创建输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Cache Size Sweep 评估")
    print(f"{'='*60}")
    print(f"模型: {args.model_name}")
    print(f"Cache Sizes: {args.cache_sizes}")
    print(f"N_sink: {args.n_sink}")
    print(f"Num Tokens: {args.num_tokens}")
    print(f"Num Runs: {args.num_runs}")
    print(f"输出目录: {args.output_dir}")
    print(f"{'='*60}\n")
    
    # 运行所有评估
    results = []
    for cache_size in args.cache_sizes:
        output_path = args.output_dir / f"cache_{cache_size}.json"
        
        result = run_evaluation(
            cache_size=cache_size,
            model_name=args.model_name,
            n_sink=args.n_sink,
            num_tokens=args.num_tokens,
            num_runs=args.num_runs,
            output_path=output_path,
        )
        
        results.append(result)
    
    # 保存汇总结果
    summary = {
        "model": args.model_name,
        "n_sink": args.n_sink,
        "num_tokens": args.num_tokens,
        "num_runs": args.num_runs,
        "results": results,
    }
    
    summary_path = args.output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n汇总结果已保存到: {summary_path}")
    
    # 打印汇总表格
    print_summary_table(results)
    
    # 绘制图表
    plot_results(results, args.output_dir)
    
    print(f"\n{'='*60}")
    print(f"Cache Size Sweep 完成!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
