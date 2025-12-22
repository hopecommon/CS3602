#!/usr/bin/env python3
"""
StreamingLLM 评估脚本

评估我们从零实现的 StreamingLLM 各种配置在预设模型上的性能
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from streaming_llm import StartRecentKVCache, StreamingLLMWrapper
from eval_utils import (
    load_tokenized_dataset,
    compute_perplexity,
    save_results,
    print_results,
)

DEFAULT_MAX_EVAL_TOKENS = int(os.environ.get("MAX_EVAL_TOKENS", "4096"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="评估 StreamingLLM 在指定模型上的性能"
    )
    
    # 模型参数
    parser.add_argument(
        "--model-name",
        type=str,
        default=os.environ.get("MODEL_NAME", "EleutherAI/pythia-2.8b"),
        help="模型名称"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="数据类型"
    )
    
    # 数据集参数
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="wikitext",
        help="数据集名称"
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="wikitext-103-v1",
        help="数据集配置"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="数据集分割"
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="文本列名"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=64,
        help="最大样本数"
    )
    parser.add_argument(
        "--max-eval-tokens",
        type=int,
        default=DEFAULT_MAX_EVAL_TOKENS,
        help="最大评估 token 数"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="信任远程代码"
    )
    
    # 评估参数
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="单次评估的最大长度"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="滑动步长"
    )
    
    # StreamingLLM 参数
    parser.add_argument(
        "--n-sink",
        type=int,
        default=4,
        help="Sink token 数量"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=1024,
        help="滑动窗口大小"
    )
    parser.add_argument(
        "--streaming-mode",
        type=str,
        choices=["ours", "mit"],
        default="ours",
        help="StreamingLLM implementation to use for the decode loop"
    )
    # Flash Attention Split Args
    parser.add_argument(
        "--flash-prefill",
        type=int,
        default=1,
        help="Enable Flash Attention for prefill (0/1)"
    )
    parser.add_argument(
        "--flash-decode",
        type=int,
        default=0,
        help="Enable Flash Attention for decode (0/1)"
    )
    parser.add_argument(
        "--flash-split",
        action="store_true",
        help="Shortcut for --flash-prefill 1 --flash-decode 0"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["both", "baseline", "streaming"],
        default="both",
        help="评估模式: both=基线+Streaming, baseline=仅基线, streaming=仅 Streaming"
    )
    parser.add_argument(
        "--baseline-results",
        type=Path,
        help="已存在的基线结果 JSON (mode=streaming 时可复用)"
    )
    
    # 输出参数
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/streaming_llm/wikitext_result.json"),
        help="输出文件路径"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Handle flash-split shortcut
    if args.flash_split:
        args.flash_prefill = 1
        args.flash_decode = 0
        
    target_cache = args.n_sink + args.window_size
    if args.max_length < target_cache:
        print(
            f"警告: max_length({args.max_length}) < n_sink + window_size ({target_cache}), "
            f"已自动调整 max_length = {target_cache}"
        )
        args.max_length = target_cache
    
    # 设置设备和数据类型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_mapping[args.dtype]
    
    if device.type == "cpu" and torch_dtype != torch.float32:
        print("警告: CPU 不支持 fp16/bf16,切换到 fp32")
        torch_dtype = torch.float32
    
    print(f"\n{'='*60}")
    print(f"StreamingLLM 评估")
    print(f"{'='*60}")
    print(f"模型: {args.model_name}")
    print(f"数据集: {args.dataset_name}:{args.dataset_config}")
    print(f"设备: {device}")
    print(f"数据类型: {torch_dtype}")
    print(f"n_sink: {args.n_sink}")
    print(f"window_size: {args.window_size}")
    print(f"模式: {args.mode}")
    print(f"[CONFIG] flash_prefill={args.flash_prefill} flash_decode={args.flash_decode} streaming={args.streaming_mode}")
    print(f"{'='*60}\n")

    streaming_cache_name = (
        "StartRecentKVCache" if args.streaming_mode == "mit" else "StreamingKVCache"
    )
    
    # 加载 tokenizer
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载数据集
    print("加载数据集...")
    encoded_dataset = load_tokenized_dataset(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        text_column=args.text_column,
        max_samples=args.max_samples,
        tokenizer=tokenizer,
        max_eval_tokens=args.max_eval_tokens,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"数据集大小: {encoded_dataset.shape[1]} tokens")
    
    # 加载模型
    print("加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
    ).to(device)
    model.eval()
    
    baseline_metrics = None
    baseline_source = None
    baseline_first_token = None
    baseline_peak_memory_mb = 0.0
    
    if args.mode in {"both", "baseline"}:
        print("\n" + "="*60)
        print("评估基线 (无压缩)")
        print("="*60)
        baseline_stats = compute_perplexity(
            model=model,
            encoded_dataset=encoded_dataset,
            device=device,
            max_length=args.max_length,
            stride=args.stride,
            use_streaming=False,
            max_cache_size=args.n_sink + args.window_size,
            flash_prefill=bool(args.flash_prefill),
            flash_decode=bool(args.flash_decode),
        )
        baseline_ppl = baseline_stats.perplexity
        baseline_time = baseline_stats.runtime_sec
        baseline_prefill = baseline_stats.prefill_sec
        baseline_first_token = baseline_stats.first_token_latency_sec
        if torch.cuda.is_available():
            baseline_peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        baseline_first_token_value = baseline_first_token if baseline_first_token is not None else 0.0
        baseline_metrics = {
            "perplexity": baseline_ppl,
            "runtime_sec": baseline_time,
            "prefill_sec": baseline_prefill,
            "first_token_latency_sec": baseline_first_token_value,
            "peak_memory_mb": baseline_peak_memory_mb,
        }
        baseline_source = "computed"
        print(f"基线 PPL: {baseline_ppl:.2f}")
        print(f"基线 Runtime: {baseline_time:.3f}s")
        print(f"基线 Prefill: {baseline_prefill:.3f}s")
    elif args.mode == "streaming" and args.baseline_results:
        if not args.baseline_results.exists():
            raise FileNotFoundError(f"基线结果文件不存在: {args.baseline_results}")
        print(f"\n加载已有基线结果: {args.baseline_results}")
        cached = json.loads(args.baseline_results.read_text())
        baseline_metrics = cached.get("baseline")
        if baseline_metrics is None:
            raise ValueError("提供的基线结果文件不包含 baseline 字段")
        baseline_source = f"loaded:{args.baseline_results}"
    else:
        print("\nStreaming 模式未提供基线, 将重新计算基线以便对比")
        baseline_stats = compute_perplexity(
            model=model,
            encoded_dataset=encoded_dataset,
            device=device,
            max_length=args.max_length,
            stride=args.stride,
            use_streaming=False,
            max_cache_size=args.n_sink + args.window_size,
            flash_prefill=bool(args.flash_prefill),
            flash_decode=bool(args.flash_decode),
        )
        baseline_ppl = baseline_stats.perplexity
        baseline_time = baseline_stats.runtime_sec
        baseline_prefill = baseline_stats.prefill_sec
        baseline_first_token = baseline_stats.first_token_latency_sec
        if torch.cuda.is_available():
            baseline_peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        baseline_first_token_value = baseline_first_token if baseline_first_token is not None else 0.0
        baseline_metrics = {
            "perplexity": baseline_ppl,
            "runtime_sec": baseline_time,
            "prefill_sec": baseline_prefill,
            "first_token_latency_sec": baseline_first_token_value,
            "peak_memory_mb": baseline_peak_memory_mb,
        }
        baseline_source = "computed"
    
    if args.mode == "baseline":
        results = {
            "model": args.model_name,
            "dataset": f"{args.dataset_name}:{args.dataset_config}",
            "split": args.split,
            "max_length": args.max_length,
            "stride": args.stride,
            "max_samples": args.max_samples,
            "max_eval_tokens": args.max_eval_tokens,
            "total_tokens": encoded_dataset.shape[1],
            "streaming_llm": {
                "n_sink": args.n_sink,
                "window_size": args.window_size,
                "implementation": args.streaming_mode,
                "cache_type": streaming_cache_name,
                "max_cache_size": target_cache,
            },
            "baseline": baseline_metrics,
            "baseline_source": baseline_source,
            "device": str(device),
            "dtype": str(torch_dtype),
        }
        print_results(results)
        save_results(results, args.output)
        return

    if baseline_metrics and baseline_first_token is None:
        baseline_first_token = baseline_metrics.get("first_token_latency_sec")
        baseline_peak_memory_mb = baseline_metrics.get("peak_memory_mb", baseline_peak_memory_mb)

    streaming_metrics = None
    compression_ratio = None
    
    if args.mode in {"both", "streaming"}:
        print("\n" + "="*60)
        print("评估 StreamingLLM (我们的实现)")
        print("="*60)
        cache_impl = None
        if args.streaming_mode == "mit":
            cache_impl = StartRecentKVCache(
                start_size=args.n_sink,
                recent_size=args.window_size,
                k_seq_dim=2,
                v_seq_dim=2,
            )
        streaming_wrapper = StreamingLLMWrapper(
            model=model,
            n_sink=args.n_sink,
            window_size=args.window_size,
            cache=cache_impl
        )
        streaming_cache_name = streaming_wrapper.cache_name
        streaming_stats = compute_perplexity(
            model=model,
            encoded_dataset=encoded_dataset,
            device=device,
            max_length=args.max_length,
            stride=args.stride,
            use_streaming=True,
            streaming_wrapper=streaming_wrapper,
            max_cache_size=args.n_sink + args.window_size,
            flash_prefill=bool(args.flash_prefill),
            flash_decode=bool(args.flash_decode),
        )
        streaming_ppl = streaming_stats.perplexity
        streaming_time = streaming_stats.runtime_sec
        streaming_prefill = streaming_stats.prefill_sec
        streaming_first_token = streaming_stats.first_token_latency_sec
        streaming_peak_memory_mb = 0.0
        if torch.cuda.is_available():
            streaming_peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        streaming_metrics = {
            "perplexity": streaming_ppl,
            "runtime_sec": streaming_time,
            "prefill_sec": streaming_prefill,
            "first_token_latency_sec": streaming_first_token,
            "peak_memory_mb": streaming_peak_memory_mb,
        }
        compression_ratio = streaming_wrapper.get_compression_ratio(
            encoded_dataset.shape[1]
        )
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"StreamingLLM PPL: {streaming_ppl:.2f}")
        print(f"StreamingLLM Runtime: {streaming_time:.3f}s")
        print(f"StreamingLLM Prefill: {streaming_prefill:.3f}s")
    
    results = {
        "model": args.model_name,
        "dataset": f"{args.dataset_name}:{args.dataset_config}",
        "split": args.split,
        "max_length": args.max_length,
        "stride": args.stride,
        "max_samples": args.max_samples,
        "max_eval_tokens": args.max_eval_tokens,
        "total_tokens": encoded_dataset.shape[1],
        "streaming_llm": {
            "n_sink": args.n_sink,
            "window_size": args.window_size,
            "implementation": args.streaming_mode,
            "cache_type": streaming_cache_name,
            "max_cache_size": target_cache,
        },
        "device": str(device),
        "dtype": str(torch_dtype),
        "baseline_source": baseline_source,
    }
    
    if baseline_metrics:
        results["baseline"] = baseline_metrics
    if streaming_metrics:
        results["streaming"] = streaming_metrics
    
    metrics_block = {}
    if streaming_metrics:
        metrics_block["compression_ratio"] = compression_ratio
        metrics_block["first_token_latency_sec"] = streaming_metrics.get("first_token_latency_sec", 0.0)
        streaming_peak = streaming_metrics.get("peak_memory_mb", 0.0)
        metrics_block["peak_memory_mb"] = streaming_peak
        if baseline_metrics and streaming_metrics["runtime_sec"] > 0:
            metrics_block["speedup"] = (
                baseline_metrics["runtime_sec"] / streaming_metrics["runtime_sec"]
            )
        if baseline_metrics:
            metrics_block["ppl_increase_percent"] = (
                (streaming_metrics["perplexity"] - baseline_metrics["perplexity"])
                / baseline_metrics["perplexity"]
            ) * 100
            baseline_peak = baseline_metrics.get("peak_memory_mb", 0.0)
            if baseline_peak > 0:
                metrics_block["peak_memory_ratio"] = streaming_peak / baseline_peak
    if metrics_block:
        results["metrics"] = metrics_block
    
    print_results(results)
    save_results(results, args.output)
    
    if streaming_metrics and "speedup" in metrics_block:
        print(f"\n{'='*60}")
        print(f"总结")
        print(f"{'='*60}")
        print(f"加速比: {metrics_block['speedup']:.2f}x")
        print(f"压缩比: {compression_ratio:.2%}")
        print(f"PPL 增加: {metrics_block.get('ppl_increase_percent', 0):.2f}%")
        peak_memory_print = metrics_block.get("peak_memory_mb")
        if peak_memory_print is not None:
            print(f"峰值显存 (streaming): {peak_memory_print:.1f} MB")
        peak_ratio = metrics_block.get("peak_memory_ratio")
        if peak_ratio is not None:
            print(f"显存占比 (streaming/baseline): {peak_ratio:.2f}x")
        first_token_latency = metrics_block.get("first_token_latency_sec")
        if first_token_latency is not None:
            print(f"首个 token latency: {first_token_latency:.4f}s")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
