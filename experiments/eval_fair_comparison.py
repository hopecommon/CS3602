#!/usr/bin/env python3
"""
Fair Comparison Evaluation Script

Provides accurate timing measurements for fair comparison between:
1. Baseline (full KV cache, no compression)
2. Our StreamingLLM implementation
3. kvpress StreamingLLM implementation

Key improvements:
- Separate model loading from evaluation timing
- Warmup runs to avoid cold start effects
- Multiple runs for statistical reliability
- Measure only the core inference loop (exclude data loading)
- Reset CUDA cache between runs
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from streaming_llm import StreamingLLMWrapper
from kvpress import StreamingLLMPress
from eval_utils import load_tokenized_dataset


def warmup_model(model, device, num_warmup=3):
    """Warmup model to avoid cold start effects"""
    print(f"Warming up model ({num_warmup} iterations)...")
    dummy_input = torch.randint(0, 1000, (1, 128)).to(device)
    
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input, use_cache=True)
    
    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    print("Warmup complete")


def evaluate_baseline(
    model,
    encoded_dataset: torch.Tensor,
    device: torch.device,
    max_length: int,
    stride: int,
    num_runs: int = 1,
) -> Tuple[float, float, float]:
    """
    Evaluate baseline (full KV cache, no compression)
    
    Returns:
        ppl: Perplexity
        avg_time: Average runtime over num_runs
        std_time: Standard deviation of runtime
    """
    print(f"\nEvaluating Baseline (full KV cache, {num_runs} runs)...")
    
    nlls = []
    total_tokens = 0
    seq_len = encoded_dataset.size(1)
    runtimes = []
    
    for run_idx in range(num_runs):
        # Reset for each run
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        run_start = time.perf_counter()
        run_nlls = []
        run_tokens = 0
        
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
            run_nlls.append(neg_log_likelihood.detach().to("cpu"))
            run_tokens += trg_len
            
            if end_loc == seq_len:
                break
        
        run_time = time.perf_counter() - run_start
        runtimes.append(run_time)
        
        # Only compute PPL once (should be same across runs)
        if run_idx == 0:
            nlls = run_nlls
            total_tokens = run_tokens
        
        print(f"  Run {run_idx + 1}/{num_runs}: {run_time:.3f}s")
    
    ppl = torch.exp(torch.stack(nlls).sum() / total_tokens).item()
    avg_time = sum(runtimes) / len(runtimes)
    std_time = (sum((t - avg_time) ** 2 for t in runtimes) / len(runtimes)) ** 0.5
    
    return ppl, avg_time, std_time


def evaluate_streaming_llm(
    model,
    encoded_dataset: torch.Tensor,
    device: torch.device,
    max_length: int,
    stride: int,
    n_sink: int,
    window_size: int,
    num_runs: int = 1,
) -> Tuple[float, float, float]:
    """
    Evaluate our StreamingLLM implementation
    
    Returns:
        ppl: Perplexity
        avg_time: Average runtime over num_runs
        std_time: Standard deviation of runtime
    """
    print(f"\nEvaluating Our StreamingLLM (n_sink={n_sink}, window={window_size}, {num_runs} runs)...")
    
    nlls = []
    total_tokens = 0
    seq_len = encoded_dataset.size(1)
    runtimes = []
    
    for run_idx in range(num_runs):
        # Reset for each run
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Create fresh wrapper for each run
        wrapper = StreamingLLMWrapper(
            model=model,
            n_sink=n_sink,
            window_size=window_size
        )
        
        run_start = time.perf_counter()
        run_nlls = []
        run_tokens = 0
        
        with wrapper.enable():
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
                run_nlls.append(neg_log_likelihood.detach().to("cpu"))
                run_tokens += trg_len
                
                if end_loc == seq_len:
                    break
        
        run_time = time.perf_counter() - run_start
        runtimes.append(run_time)
        
        # Only compute PPL once
        if run_idx == 0:
            nlls = run_nlls
            total_tokens = run_tokens
        
        print(f"  Run {run_idx + 1}/{num_runs}: {run_time:.3f}s")
    
    ppl = torch.exp(torch.stack(nlls).sum() / total_tokens).item()
    avg_time = sum(runtimes) / len(runtimes)
    std_time = (sum((t - avg_time) ** 2 for t in runtimes) / len(runtimes)) ** 0.5
    
    return ppl, avg_time, std_time


def evaluate_kvpress(
    model,
    encoded_dataset: torch.Tensor,
    device: torch.device,
    max_length: int,
    stride: int,
    n_sink: int,
    window_size: int,
    num_runs: int = 1,
) -> Tuple[float, float, float]:
    """
    Evaluate kvpress StreamingLLM implementation
    
    Returns:
        ppl: Perplexity
        avg_time: Average runtime over num_runs
        std_time: Standard deviation of runtime
    """
    print(f"\nEvaluating kvpress StreamingLLM (n_sink={n_sink}, window={window_size}, {num_runs} runs)...")
    
    # Calculate compression_ratio
    max_cache_size = n_sink + window_size
    total_tokens_dataset = encoded_dataset.shape[1]
    compression_ratio = max(0.0, (total_tokens_dataset - max_cache_size) / total_tokens_dataset)
    
    nlls = []
    total_tokens = 0
    seq_len = encoded_dataset.size(1)
    runtimes = []
    
    for run_idx in range(num_runs):
        # Reset for each run
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Create fresh press for each run
        press = StreamingLLMPress(
            compression_ratio=compression_ratio,
            n_sink=n_sink
        )
        
        run_start = time.perf_counter()
        run_nlls = []
        run_tokens = 0
        
        with press(model):
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
                run_nlls.append(neg_log_likelihood.detach().to("cpu"))
                run_tokens += trg_len
                
                if end_loc == seq_len:
                    break
        
        run_time = time.perf_counter() - run_start
        runtimes.append(run_time)
        
        # Only compute PPL once
        if run_idx == 0:
            nlls = run_nlls
            total_tokens = run_tokens
        
        print(f"  Run {run_idx + 1}/{num_runs}: {run_time:.3f}s")
    
    ppl = torch.exp(torch.stack(nlls).sum() / total_tokens).item()
    avg_time = sum(runtimes) / len(runtimes)
    std_time = (sum((t - avg_time) ** 2 for t in runtimes) / len(runtimes)) ** 0.5
    
    return ppl, avg_time, std_time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fair comparison evaluation with accurate timing"
    )
    
    # Model parameters
    parser.add_argument("--model-name", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    
    # Dataset parameters
    parser.add_argument("--dataset-name", type=str, default="wikitext")
    parser.add_argument("--dataset-config", type=str, default="wikitext-103-v1")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-eval-tokens", type=int, default=4096)
    parser.add_argument("--trust-remote-code", action="store_true")
    
    # Evaluation parameters
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--num-runs", type=int, default=3, help="Number of runs for averaging")
    parser.add_argument("--num-warmup", type=int, default=3, help="Number of warmup iterations")
    
    # StreamingLLM parameters
    parser.add_argument("--n-sink", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=1024)
    
    # Output
    parser.add_argument("--output", type=Path, default=Path("results/fair_comparison.json"))
    
    return parser.parse_args()


def main():
    args = parse_args()
    
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
    
    print(f"\n{'='*70}")
    print(f"Fair Comparison Evaluation")
    print(f"{'='*70}")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset_name}:{args.dataset_config}")
    print(f"Device: {device}")
    print(f"Data type: {torch_dtype}")
    print(f"n_sink: {args.n_sink}")
    print(f"window_size: {args.window_size}")
    print(f"Number of runs: {args.num_runs}")
    print(f"Warmup iterations: {args.num_warmup}")
    print(f"{'='*70}\n")
    
    # Load tokenizer and dataset (ONCE, before timing)
    print("Loading tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
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
    print(f"Dataset size: {encoded_dataset.shape[1]} tokens\n")
    
    # Load model (ONCE, before timing)
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
    ).to(device)
    model.eval()
    print("Model loaded\n")
    
    # Warmup
    warmup_model(model, device, args.num_warmup)
    
    # Run evaluations
    print(f"\n{'='*70}")
    print("Running Evaluations")
    print(f"{'='*70}")
    
    # 1. Baseline
    baseline_ppl, baseline_time, baseline_std = evaluate_baseline(
        model, encoded_dataset, device, args.max_length, args.stride, args.num_runs
    )
    print(f"Baseline: PPL={baseline_ppl:.2f}, Time={baseline_time:.3f}±{baseline_std:.3f}s")
    
    # 2. Our StreamingLLM
    our_ppl, our_time, our_std = evaluate_streaming_llm(
        model, encoded_dataset, device, args.max_length, args.stride,
        args.n_sink, args.window_size, args.num_runs
    )
    print(f"Our StreamingLLM: PPL={our_ppl:.2f}, Time={our_time:.3f}±{our_std:.3f}s")
    
    # 3. kvpress StreamingLLM
    kvpress_ppl, kvpress_time, kvpress_std = evaluate_kvpress(
        model, encoded_dataset, device, args.max_length, args.stride,
        args.n_sink, args.window_size, args.num_runs
    )
    print(f"kvpress StreamingLLM: PPL={kvpress_ppl:.2f}, Time={kvpress_time:.3f}±{kvpress_std:.3f}s")
    
    # Calculate metrics
    our_speedup = baseline_time / our_time if our_time > 0 else 0
    kvpress_speedup = baseline_time / kvpress_time if kvpress_time > 0 else 0
    max_cache_size = args.n_sink + args.window_size
    compression_ratio = 1 - max_cache_size / encoded_dataset.shape[1]
    
    # Save results
    results = {
        "model": args.model_name,
        "dataset": f"{args.dataset_name}:{args.dataset_config}",
        "config": {
            "n_sink": args.n_sink,
            "window_size": args.window_size,
            "max_cache_size": max_cache_size,
            "num_runs": args.num_runs,
            "num_warmup": args.num_warmup,
        },
        "baseline": {
            "perplexity": baseline_ppl,
            "runtime_sec": baseline_time,
            "runtime_std": baseline_std,
        },
        "our_streaming_llm": {
            "perplexity": our_ppl,
            "runtime_sec": our_time,
            "runtime_std": our_std,
            "speedup": our_speedup,
            "ppl_increase_percent": ((our_ppl - baseline_ppl) / baseline_ppl) * 100,
        },
        "kvpress_streaming_llm": {
            "perplexity": kvpress_ppl,
            "runtime_sec": kvpress_time,
            "runtime_std": kvpress_std,
            "speedup": kvpress_speedup,
            "ppl_increase_percent": ((kvpress_ppl - baseline_ppl) / baseline_ppl) * 100,
        },
        "compression_ratio": compression_ratio,
        "device": str(device),
        "dtype": str(torch_dtype),
    }
    
    # Save to file
    import json
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    
    # Print summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"Baseline:           PPL={baseline_ppl:.2f}, Time={baseline_time:.3f}±{baseline_std:.3f}s")
    print(f"Our StreamingLLM:   PPL={our_ppl:.2f}, Time={our_time:.3f}±{our_std:.3f}s, Speedup={our_speedup:.2f}x")
    print(f"kvpress StreamingLLM: PPL={kvpress_ppl:.2f}, Time={kvpress_time:.3f}±{kvpress_std:.3f}s, Speedup={kvpress_speedup:.2f}x")
    print(f"Compression ratio: {compression_ratio:.2%}")
    print(f"\nResults saved to: {args.output}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()