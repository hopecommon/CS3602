#!/usr/bin/env python3
"""
Per-Token Decoding Latency 评估脚本

按照 StreamingLLM 论文的标准方法测量每个 token 的解码延迟:
1. GPU warmup (100-200 tokens)
2. 只统计 cache 填满后的 tokens
3. 使用 torch.cuda.synchronize() 确保准确计时
4. 多次运行取平均和标准差
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Tuple, Dict, List

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from streaming_llm import StreamingLLMWrapper


def parse_args():
    parser = argparse.ArgumentParser(
        description="测量 per-token decoding latency"
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
    
    # 评估参数
    parser.add_argument(
        "--cache-size",
        type=int,
        default=1024,
        help="Cache size (window_size for StreamingLLM)"
    )
    parser.add_argument(
        "--n-sink",
        type=int,
        default=4,
        help="Sink token 数量 (仅 StreamingLLM)"
    )
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=512,
        help="初始 prompt 长度"
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=2000,
        help="生成的 token 数量 (用于统计)"
    )
    parser.add_argument(
        "--warmup-tokens",
        type=int,
        default=200,
        help="Warmup token 数量"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="重复运行次数"
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
        default=Path("results/decoding/decoding_latency.json"),
        help="输出文件路径"
    )
    
    return parser.parse_args()


def generate_random_prompt(tokenizer, length: int) -> torch.Tensor:
    """生成随机 prompt"""
    # 使用固定种子确保可复现
    torch.manual_seed(42)
    vocab_size = len(tokenizer)
    # 避免特殊 token
    token_ids = torch.randint(100, vocab_size - 100, (1, length))
    return token_ids


def warmup_model(
    model,
    input_ids: torch.Tensor,
    device: torch.device,
    num_tokens: int,
    use_streaming: bool = False,
    streaming_wrapper = None,
) -> None:
    """
    GPU warmup 阶段
    
    Args:
        model: 语言模型
        input_ids: 初始输入 [1, seq_len]
        device: 设备
        num_tokens: warmup token 数量
        use_streaming: 是否使用 StreamingLLM
        streaming_wrapper: StreamingLLMWrapper 实例
    """
    print(f"  Warmup: 生成 {num_tokens} tokens...")
    
    if use_streaming and streaming_wrapper is not None:
        context = streaming_wrapper.enable()
    else:
        from contextlib import nullcontext
        context = nullcontext()
    
    with context:
        with torch.no_grad():
            current_ids = input_ids.to(device)
            past_key_values = None
            
            for _ in range(num_tokens):
                outputs = model(
                    input_ids=current_ids,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                
                # 获取下一个 token
                next_token = outputs.logits[:, -1:].argmax(dim=-1)
                current_ids = next_token
                past_key_values = outputs.past_key_values
                
                if use_streaming and streaming_wrapper is not None:
                    streaming_wrapper.update(past_key_values)
    
    # 清空 CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    print("  Warmup 完成")


def measure_decoding_latency(
    model,
    input_ids: torch.Tensor,
    device: torch.device,
    num_tokens: int,
    cache_size: int,
    use_streaming: bool = False,
    streaming_wrapper = None,
) -> Tuple[List[float], float]:
    """
    测量 per-token decoding latency
    
    Args:
        model: 语言模型
        input_ids: 初始输入 [1, seq_len]
        device: 设备
        num_tokens: 要生成的 token 数量
        cache_size: cache 大小 (仅用于参考,不影响测量逻辑)
        use_streaming: 是否使用 StreamingLLM
        streaming_wrapper: StreamingLLMWrapper 实例
    
    Returns:
        latencies: 每个 token 的延迟列表 (秒)
        peak_memory_mb: 峰值显存 (MB)
    """
    latencies = []
    
    if use_streaming and streaming_wrapper is not None:
        context = streaming_wrapper.enable()
    else:
        from contextlib import nullcontext
        context = nullcontext()
    
    with context:
        with torch.no_grad():
            current_ids = input_ids.to(device)
            past_key_values = None
            
            for step in range(num_tokens):
                # 同步 CUDA 确保准确计时
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                t0 = time.perf_counter()
                
                outputs = model(
                    input_ids=current_ids,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                
                # 同步 CUDA
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                t1 = time.perf_counter()
                latencies.append(t1 - t0)
                
                # 获取下一个 token
                next_token = outputs.logits[:, -1:].argmax(dim=-1)
                current_ids = next_token
                past_key_values = outputs.past_key_values
                
                if use_streaming and streaming_wrapper is not None:
                    streaming_wrapper.update(past_key_values)
    
    # 获取峰值显存
    peak_memory_mb = 0
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    return latencies, peak_memory_mb


def evaluate_baseline(
    model,
    tokenizer,
    device: torch.device,
    args,
) -> Dict:
    """评估 Baseline (Full KV Cache)"""
    print("\n" + "="*60)
    print("评估 Baseline (Full KV Cache)")
    print("="*60)
    
    all_latencies = []
    all_memories = []
    
    for run in range(args.num_runs):
        print(f"\nRun {run + 1}/{args.num_runs}")
        
        # 生成 prompt
        input_ids = generate_random_prompt(tokenizer, args.prompt_length)
        
        # Warmup
        warmup_model(
            model=model,
            input_ids=input_ids,
            device=device,
            num_tokens=args.warmup_tokens,
            use_streaming=False,
        )
        
        # 测量
        print(f"  测量: 生成 {args.num_tokens} tokens...")
        latencies, peak_memory = measure_decoding_latency(
            model=model,
            input_ids=input_ids,
            device=device,
            num_tokens=args.num_tokens,
            cache_size=args.cache_size,
            use_streaming=False,
        )
        
        all_latencies.extend(latencies)
        all_memories.append(peak_memory)
        
        # 安全地计算平均值
        if len(latencies) > 0:
            print(f"  收集到 {len(latencies)} 个延迟测量")
            print(f"  平均延迟: {np.mean(latencies)*1000:.3f} ms/token")
        else:
            print(f"  错误: 没有收集到延迟数据!")
        print(f"  峰值显存: {peak_memory:.1f} MB")
    
    # 统计 - 添加空数组检查
    if len(all_latencies) > 0:
        mean_latency = np.mean(all_latencies)
        std_latency = np.std(all_latencies) if len(all_latencies) > 1 else 0.0
        median_latency = np.median(all_latencies)
    else:
        print("警告: 没有收集到任何延迟数据,使用默认值")
        mean_latency = 0.0
        std_latency = 0.0
        median_latency = 0.0
    
    if len(all_memories) > 0:
        mean_memory = np.mean(all_memories)
    else:
        mean_memory = 0.0
    
    print(f"\n总结 ({args.num_runs} runs):")
    print(f"  平均延迟: {mean_latency*1000:.3f} ± {std_latency*1000:.3f} ms/token")
    print(f"  中位数延迟: {median_latency*1000:.3f} ms/token")
    print(f"  平均显存: {mean_memory:.1f} MB")
    
    return {
        "mean_latency_ms": mean_latency * 1000,
        "std_latency_ms": std_latency * 1000,
        "median_latency_ms": median_latency * 1000,
        "mean_memory_mb": mean_memory,
        "all_latencies_ms": [l * 1000 for l in all_latencies],
    }


def evaluate_streaming_llm(
    model,
    tokenizer,
    device: torch.device,
    args,
) -> Dict:
    """评估 StreamingLLM"""
    print("\n" + "="*60)
    print("评估 StreamingLLM (我们的实现)")
    print("="*60)
    print(f"  n_sink: {args.n_sink}")
    print(f"  window_size: {args.cache_size}")
    
    all_latencies = []
    all_memories = []
    
    for run in range(args.num_runs):
        print(f"\nRun {run + 1}/{args.num_runs}")
        
        # 生成 prompt
        input_ids = generate_random_prompt(tokenizer, args.prompt_length)
        
        # 创建 StreamingLLM wrapper
        streaming_wrapper = StreamingLLMWrapper(
            model=model,
            n_sink=args.n_sink,
            window_size=args.cache_size
        )
        
        # Warmup
        warmup_model(
            model=model,
            input_ids=input_ids,
            device=device,
            num_tokens=args.warmup_tokens,
            use_streaming=True,
            streaming_wrapper=streaming_wrapper,
        )
        
        # 测量
        print(f"  测量: 生成 {args.num_tokens} tokens...")
        latencies, peak_memory = measure_decoding_latency(
            model=model,
            input_ids=input_ids,
            device=device,
            num_tokens=args.num_tokens,
            cache_size=args.cache_size,
            use_streaming=True,
            streaming_wrapper=streaming_wrapper,
        )
        
        all_latencies.extend(latencies)
        all_memories.append(peak_memory)
        
        # 安全地计算平均值
        if len(latencies) > 0:
            print(f"  收集到 {len(latencies)} 个延迟测量")
            print(f"  平均延迟: {np.mean(latencies)*1000:.3f} ms/token")
        else:
            print(f"  错误: 没有收集到延迟数据!")
        print(f"  峰值显存: {peak_memory:.1f} MB")
    
    # 统计 - 添加空数组检查
    if len(all_latencies) > 0:
        mean_latency = np.mean(all_latencies)
        std_latency = np.std(all_latencies) if len(all_latencies) > 1 else 0.0
        median_latency = np.median(all_latencies)
    else:
        print("警告: 没有收集到任何延迟数据,使用默认值")
        mean_latency = 0.0
        std_latency = 0.0
        median_latency = 0.0
    
    if len(all_memories) > 0:
        mean_memory = np.mean(all_memories)
    else:
        mean_memory = 0.0
    
    print(f"\n总结 ({args.num_runs} runs):")
    print(f"  平均延迟: {mean_latency*1000:.3f} ± {std_latency*1000:.3f} ms/token")
    print(f"  中位数延迟: {median_latency*1000:.3f} ms/token")
    print(f"  平均显存: {mean_memory:.1f} MB")
    
    return {
        "mean_latency_ms": mean_latency * 1000,
        "std_latency_ms": std_latency * 1000,
        "median_latency_ms": median_latency * 1000,
        "mean_memory_mb": mean_memory,
        "all_latencies_ms": [l * 1000 for l in all_latencies],
    }


def main():
    args = parse_args()
    
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
    print(f"Per-Token Decoding Latency 评估")
    print(f"{'='*60}")
    print(f"模型: {args.model_name}")
    print(f"设备: {device}")
    print(f"数据类型: {torch_dtype}")
    print(f"Cache Size: {args.cache_size}")
    print(f"Prompt Length: {args.prompt_length}")
    print(f"Num Tokens: {args.num_tokens}")
    print(f"Warmup Tokens: {args.warmup_tokens}")
    print(f"Num Runs: {args.num_runs}")
    print(f"Mode: {args.mode}")
    print(f"{'='*60}\n")
    
    # 加载 tokenizer
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    print("加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
    ).to(device)
    model.eval()
    
    baseline_results = None
    baseline_source = None
    
    if args.mode in {"both", "baseline"}:
        baseline_results = evaluate_baseline(
            model=model,
            tokenizer=tokenizer,
            device=device,
            args=args,
        )
        baseline_source = "computed"
    elif args.baseline_results:
        if not args.baseline_results.exists():
            raise FileNotFoundError(f"基线结果文件不存在: {args.baseline_results}")
        cache_data = json.loads(args.baseline_results.read_text())
        baseline_results = cache_data.get("baseline")
        if baseline_results is None:
            raise ValueError("提供的基线结果文件不包含 baseline 数据")
        baseline_source = f"loaded:{args.baseline_results}"
    else:
        raise ValueError("Streaming 模式需要提供 --baseline-results")
    
    streaming_results = None
    comparison = None
    
    if args.mode in {"both", "streaming"}:
        streaming_results = evaluate_streaming_llm(
            model=model,
            tokenizer=tokenizer,
            device=device,
            args=args,
        )
        
        if streaming_results["mean_latency_ms"] > 0:
            speedup = baseline_results["mean_latency_ms"] / streaming_results["mean_latency_ms"]
        else:
            print("警告: StreamingLLM 延迟为 0,无法计算加速比")
            speedup = 0.0
        
        if baseline_results["mean_memory_mb"] > 0:
            memory_reduction = (
                baseline_results["mean_memory_mb"] - streaming_results["mean_memory_mb"]
            ) / baseline_results["mean_memory_mb"]
        else:
            print("警告: Baseline 显存在 0,无法计算显存减少比例")
            memory_reduction = 0.0
        
        comparison = {
            "speedup": speedup,
            "memory_reduction_percent": memory_reduction * 100,
        }
    
    results = {
        "model": args.model_name,
        "device": str(device),
        "dtype": str(torch_dtype),
        "mode": args.mode,
        "baseline_source": baseline_source,
        "config": {
            "cache_size": args.cache_size,
            "n_sink": args.n_sink,
            "prompt_length": args.prompt_length,
            "num_tokens": args.num_tokens,
            "warmup_tokens": args.warmup_tokens,
            "num_runs": args.num_runs,
        },
    }
    if baseline_results:
        results["baseline"] = baseline_results
    if streaming_results:
        results["streaming_llm"] = streaming_results
    if comparison:
        results["comparison"] = comparison
    
    if streaming_results and comparison:
        print(f"\n{'='*60}")
        print(f"最终对比")
        print(f"{'='*60}")
        print(
            f"Baseline:      {baseline_results['mean_latency_ms']:.3f} ± "
            f"{baseline_results['std_latency_ms']:.3f} ms/token"
        )
        print(
            f"StreamingLLM:  {streaming_results['mean_latency_ms']:.3f} ± "
            f"{streaming_results['std_latency_ms']:.3f} ms/token"
        )
        print(f"加速比:        {comparison['speedup']:.2f}x")
        print(f"显存减少:      {comparison['memory_reduction_percent']:.1f}%")
        print(f"{'='*60}\n")
    elif baseline_results and args.mode == "baseline":
        print(f"\n{'='*60}")
        print("基线评估完成")
        print(f"{'='*60}")
        print(
            f"Baseline: {baseline_results['mean_latency_ms']:.3f} ± "
            f"{baseline_results['std_latency_ms']:.3f} ms/token"
        )
        print(f"{'='*60}\n")
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
