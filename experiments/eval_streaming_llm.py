#!/usr/bin/env python3
"""
StreamingLLM 评估脚本

评估我们从零实现的 StreamingLLM 在 Pythia-70M 上的性能
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from streaming_llm import StreamingLLMWrapper
from eval_utils import (
    load_tokenized_dataset,
    compute_perplexity,
    save_results,
    print_results
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="评估 StreamingLLM 在 Pythia-70M 上的性能"
    )
    
    # 模型参数
    parser.add_argument(
        "--model-name",
        type=str,
        default="EleutherAI/pythia-70m",
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
        default=4096,
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
    print(f"{'='*60}\n")
    
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
    
    # 评估基线
    print("\n" + "="*60)
    print("评估基线 (无压缩)")
    print("="*60)
    baseline_ppl, baseline_time, baseline_prefill = compute_perplexity(
        model=model,
        encoded_dataset=encoded_dataset,
        device=device,
        max_length=args.max_length,
        stride=args.stride,
        use_streaming=False,
    )
    print(f"基线 PPL: {baseline_ppl:.2f}")
    print(f"基线 Runtime: {baseline_time:.3f}s")
    print(f"基线 Prefill: {baseline_prefill:.3f}s")
    
    # 评估 StreamingLLM
    print("\n" + "="*60)
    print("评估 StreamingLLM (我们的实现)")
    print("="*60)
    
    streaming_wrapper = StreamingLLMWrapper(
        model=model,
        n_sink=args.n_sink,
        window_size=args.window_size
    )
    
    streaming_ppl, streaming_time, streaming_prefill = compute_perplexity(
        model=model,
        encoded_dataset=encoded_dataset,
        device=device,
        max_length=args.max_length,
        stride=args.stride,
        use_streaming=True,
        streaming_wrapper=streaming_wrapper,
    )
    print(f"StreamingLLM PPL: {streaming_ppl:.2f}")
    print(f"StreamingLLM Runtime: {streaming_time:.3f}s")
    print(f"StreamingLLM Prefill: {streaming_prefill:.3f}s")
    
    # 计算加速比和压缩比
    speedup = baseline_time / streaming_time if streaming_time > 0 else 0
    compression_ratio = streaming_wrapper.get_compression_ratio(
        encoded_dataset.shape[1]
    )
    ppl_increase = ((streaming_ppl - baseline_ppl) / baseline_ppl) * 100
    
    # 显存使用
    peak_memory_mb = 0
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    # 汇总结果
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
            "max_cache_size": args.n_sink + args.window_size,
        },
        "baseline": {
            "perplexity": baseline_ppl,
            "runtime_sec": baseline_time,
            "prefill_sec": baseline_prefill,
        },
        "streaming": {
            "perplexity": streaming_ppl,
            "runtime_sec": streaming_time,
            "prefill_sec": streaming_prefill,
        },
        "metrics": {
            "speedup": speedup,
            "compression_ratio": compression_ratio,
            "ppl_increase_percent": ppl_increase,
            "peak_memory_mb": peak_memory_mb,
        },
        "device": str(device),
        "dtype": str(torch_dtype),
    }
    
    # 打印和保存结果
    print_results(results)
    save_results(results, args.output)
    
    print(f"\n{'='*60}")
    print(f"总结")
    print(f"{'='*60}")
    print(f"加速比: {speedup:.2f}x")
    print(f"压缩比: {compression_ratio:.2%}")
    print(f"PPL 增加: {ppl_increase:.2f}%")
    print(f"峰值显存: {peak_memory_mb:.1f} MB")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()