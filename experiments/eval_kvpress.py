#!/usr/bin/env python3
"""
KVPress Official Implementation Evaluation Script

Evaluate kvpress's official StreamingLLM implementation on Pythia-70M
for comparison with our from-scratch implementation.
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import kvpress
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvpress import StreamingLLMPress, KeyRerotationPress

from eval_utils import (
    load_tokenized_dataset,
    save_results,
    print_results
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate kvpress StreamingLLM on Pythia-70M"
    )
    
    # Model parameters
    parser.add_argument(
        "--model-name",
        type=str,
        default="EleutherAI/pythia-70m",
        help="Model name"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type"
    )
    
    # Dataset parameters
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="wikitext",
        help="Dataset name"
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="wikitext-103-v1",
        help="Dataset config"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split"
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Text column name"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=64,
        help="Maximum number of samples"
    )
    parser.add_argument(
        "--max-eval-tokens",
        type=int,
        default=4096,
        help="Maximum evaluation tokens"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code"
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum length for single evaluation"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="Sliding stride"
    )
    
    # StreamingLLM parameters
    parser.add_argument(
        "--n-sink",
        type=int,
        default=4,
        help="Number of sink tokens"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=1024,
        help="Sliding window size"
    )
    
    # Output parameters
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/kvpress/wikitext_result.json"),
        help="Output file path"
    )
    
    return parser.parse_args()


def compute_perplexity_kvpress(
    model,
    encoded_dataset: torch.Tensor,
    device: torch.device,
    max_length: int,
    stride: int,
    press=None,
):
    """
    Compute perplexity using kvpress context manager
    """
    import time
    from contextlib import nullcontext
    
    nlls = []
    total_tokens = 0
    seq_len = encoded_dataset.size(1)
    
    context = press(model) if press is not None else nullcontext()
    
    total_start = time.perf_counter()
    prefill_start = time.perf_counter()
    
    with context:
        for start_idx in range(0, seq_len, stride):
            begin_loc = max(start_idx + stride - max_length, 0)
            end_loc = min(start_idx + stride, seq_len)
            trg_len = end_loc - start_idx
            input_ids = encoded_dataset[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            
            with torch.no_grad():
                outputs = model(input_ids=input_ids, labels=target_ids, use_cache=True)
            
            neg_log_likelihood = outputs.loss * trg_len
            nlls.append(neg_log_likelihood.detach().to("cpu"))
            total_tokens += trg_len
            
            if end_loc == seq_len:
                break
    
    ppl = torch.exp(torch.stack(nlls).sum() / total_tokens)
    total_time = time.perf_counter() - total_start
    prefill_time = time.perf_counter() - prefill_start
    
    return ppl.item(), total_time, prefill_time


def main():
    args = parse_args()
    target_cache = args.n_sink + args.window_size
    if args.max_length < target_cache:
        print(
            f"Warning: max_length({args.max_length}) < n_sink + window_size ({target_cache}), "
            f"adjusting max_length to {target_cache}"
        )
        args.max_length = target_cache
    
    # Setup device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_mapping[args.dtype]
    
    if device.type == "cpu" and torch_dtype != torch.float32:
        print("Warning: CPU does not support fp16/bf16, switching to fp32")
        torch_dtype = torch.float32
    
    print(f"\n{'='*60}")
    print(f"KVPress StreamingLLM Evaluation")
    print(f"{'='*60}")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset_name}:{args.dataset_config}")
    print(f"Device: {device}")
    print(f"Data type: {torch_dtype}")
    print(f"n_sink: {args.n_sink}")
    print(f"window_size: {args.window_size}")
    print(f"{'='*60}\n")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    print("Loading dataset...")
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
    print(f"Dataset size: {encoded_dataset.shape[1]} tokens")
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
    ).to(device)
    model.eval()
    
    if hasattr(model, "gpt_neox"):
        rotary_emb = getattr(model.gpt_neox, "rotary_emb", None)
        for layer in model.gpt_neox.layers:
            attn = getattr(layer, "attention", None)
            if attn is None:
                continue
            if rotary_emb is not None and not hasattr(attn, "rotary_emb"):
                attn.rotary_emb = rotary_emb
            if not hasattr(attn, "head_dim") and hasattr(attn, "head_size"):
                attn.head_dim = attn.head_size
    
    # Evaluate baseline
    print("\n" + "="*60)
    print("Evaluating Baseline (no compression)")
    print("="*60)
    baseline_ppl, baseline_time, baseline_prefill = compute_perplexity_kvpress(
        model=model,
        encoded_dataset=encoded_dataset,
        device=device,
        max_length=args.max_length,
        stride=args.stride,
        press=None,
    )
    print(f"Baseline PPL: {baseline_ppl:.2f}")
    print(f"Baseline Runtime: {baseline_time:.3f}s")
    print(f"Baseline Prefill: {baseline_prefill:.3f}s")
    
    max_cache_size = target_cache
    total_tokens = encoded_dataset.shape[1]
    keep_ratio = max_cache_size / args.max_length
    compression_ratio = max(0.0, 1.0 - keep_ratio)
    
    print(f"\nTarget cache size per layer: {max_cache_size} tokens")
    print(f"Compression ratio (kvpress): {compression_ratio:.4f} (max_length={args.max_length})")
    
    # Evaluate kvpress StreamingLLM
    print("\n" + "="*60)
    print("Evaluating kvpress StreamingLLM")
    print("="*60)
    
    base_press = StreamingLLMPress(
        compression_ratio=compression_ratio,
        n_sink=args.n_sink
    )

    press = KeyRerotationPress(
        press=base_press
    )
    
    streaming_ppl, streaming_time, streaming_prefill = compute_perplexity_kvpress(
        model=model,
        encoded_dataset=encoded_dataset,
        device=device,
        max_length=args.max_length,
        stride=args.stride,
        press=press,
    )
    print(f"kvpress StreamingLLM PPL: {streaming_ppl:.2f}")
    print(f"kvpress StreamingLLM Runtime: {streaming_time:.3f}s")
    print(f"kvpress StreamingLLM Prefill: {streaming_prefill:.3f}s")
    
    # Calculate metrics
    speedup = baseline_time / streaming_time if streaming_time > 0 else 0
    ppl_increase = ((streaming_ppl - baseline_ppl) / baseline_ppl) * 100
    
    # Memory usage
    peak_memory_mb = 0
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    # Aggregate results
    results = {
        "model": args.model_name,
        "dataset": f"{args.dataset_name}:{args.dataset_config}",
        "split": args.split,
        "max_length": args.max_length,
        "stride": args.stride,
        "max_samples": args.max_samples,
        "max_eval_tokens": args.max_eval_tokens,
        "total_tokens": total_tokens,
        "streaming_llm": {
            "n_sink": args.n_sink,
            "window_size": args.window_size,
            "max_cache_size": max_cache_size,
            "compression_ratio": compression_ratio,
        },
        "baseline": {
            "perplexity": baseline_ppl,
            "runtime_sec": baseline_time,
            "prefill_sec": baseline_prefill,
        },
        "kvpress": {
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
    
    # Print and save results
    print_results(results)
    save_results(results, args.output)
    
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Compression ratio: {compression_ratio:.2%}")
    print(f"PPL increase: {ppl_increase:.2f}%")
    print(f"Peak memory: {peak_memory_mb:.1f} MB")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
