#!/usr/bin/env python3
"""
消融实验脚本

研究 window_size 和 n_sink 对 StreamingLLM 性能的影响
"""

import argparse
import sys
from pathlib import Path
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from streaming_llm import StreamingLLMWrapper
from eval_utils import (
    load_tokenized_dataset,
    compute_perplexity,
)


def parse_args():
    parser = argparse.ArgumentParser(description="StreamingLLM 消融实验")
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="EleutherAI/pythia-70m",
        help="模型名称"
    )
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
    parser.add_argument(
        "--ablation-type",
        type=str,
        choices=["window_size", "n_sink"],
        required=True,
        help="消融类型"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/streaming_llm/ablation_results.json"),
        help="输出文件路径"
    )
    
    return parser.parse_args()


def run_ablation_window_size(
    model,
    encoded_dataset,
    device,
    max_length,
    stride,
    n_sink=4
):
    """
    消融实验: window_size 影响
    
    固定 n_sink=4, 变化 window_size
    """
    window_sizes = [128, 256, 512, 1024, 2048, 4096]
    results = []
    
    print(f"\n{'='*60}")
    print(f"消融实验: Window Size (n_sink={n_sink})")
    print(f"{'='*60}\n")
    
    for window_size in window_sizes:
        print(f"测试 window_size={window_size}...")
        
        wrapper = StreamingLLMWrapper(
            model=model,
            n_sink=n_sink,
            window_size=window_size
        )
        
        ppl, runtime, prefill = compute_perplexity(
            model=model,
            encoded_dataset=encoded_dataset,
            device=device,
            max_length=max_length,
            stride=stride,
            use_streaming=True,
            streaming_wrapper=wrapper,
        )
        
        compression_ratio = wrapper.get_compression_ratio(encoded_dataset.shape[1])
        
        result = {
            "window_size": window_size,
            "n_sink": n_sink,
            "max_cache_size": n_sink + window_size,
            "perplexity": ppl,
            "runtime_sec": runtime,
            "prefill_sec": prefill,
            "compression_ratio": compression_ratio,
        }
        results.append(result)
        
        print(f"  PPL: {ppl:.2f}, Runtime: {runtime:.3f}s, "
              f"Compression: {compression_ratio:.2%}")
    
    return results


def run_ablation_n_sink(
    model,
    encoded_dataset,
    device,
    max_length,
    stride,
    window_size=1024
):
    """
    消融实验: n_sink 影响
    
    固定 window_size=1024, 变化 n_sink
    """
    n_sinks = [0, 1, 2, 4, 8, 16]
    results = []
    
    print(f"\n{'='*60}")
    print(f"消融实验: N_sink (window_size={window_size})")
    print(f"{'='*60}\n")
    
    for n_sink in n_sinks:
        print(f"测试 n_sink={n_sink}...")
        
        wrapper = StreamingLLMWrapper(
            model=model,
            n_sink=n_sink,
            window_size=window_size
        )
        
        ppl, runtime, prefill = compute_perplexity(
            model=model,
            encoded_dataset=encoded_dataset,
            device=device,
            max_length=max_length,
            stride=stride,
            use_streaming=True,
            streaming_wrapper=wrapper,
        )
        
        compression_ratio = wrapper.get_compression_ratio(encoded_dataset.shape[1])
        
        result = {
            "n_sink": n_sink,
            "window_size": window_size,
            "max_cache_size": n_sink + window_size,
            "perplexity": ppl,
            "runtime_sec": runtime,
            "prefill_sec": prefill,
            "compression_ratio": compression_ratio,
        }
        results.append(result)
        
        print(f"  PPL: {ppl:.2f}, Runtime: {runtime:.3f}s, "
              f"Compression: {compression_ratio:.2%}")
    
    return results


def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    
    print(f"\n{'='*60}")
    print(f"StreamingLLM 消融实验")
    print(f"{'='*60}")
    print(f"模型: {args.model_name}")
    print(f"数据集: {args.dataset_name}:{args.dataset_config}")
    print(f"消融类型: {args.ablation_type}")
    print(f"设备: {device}")
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
        split="test",
        text_column="text",
        max_samples=args.max_samples,
        tokenizer=tokenizer,
        max_eval_tokens=args.max_eval_tokens,
    )
    print(f"数据集大小: {encoded_dataset.shape[1]} tokens")
    
    # 加载模型
    print("加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
    ).to(device)
    model.eval()
    
    # 运行消融实验
    if args.ablation_type == "window_size":
        results = run_ablation_window_size(
            model=model,
            encoded_dataset=encoded_dataset,
            device=device,
            max_length=args.max_length,
            stride=args.stride,
        )
    else:  # n_sink
        results = run_ablation_n_sink(
            model=model,
            encoded_dataset=encoded_dataset,
            device=device,
            max_length=args.max_length,
            stride=args.stride,
        )
    
    # 保存结果
    output_data = {
        "model": args.model_name,
        "dataset": f"{args.dataset_name}:{args.dataset_config}",
        "ablation_type": args.ablation_type,
        "total_tokens": encoded_dataset.shape[1],
        "results": results,
    }
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_data, indent=2))
    
    print(f"\n{'='*60}")
    print(f"消融实验完成")
    print(f"{'='*60}")
    print(f"结果已保存到: {args.output}")
    print(f"{'='*60}\n")
    
    # 打印总结
    print("结果总结:")
    print(json.dumps(output_data, indent=2))


if __name__ == "__main__":
    main()