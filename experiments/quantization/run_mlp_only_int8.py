#!/usr/bin/env python3
"""
Evaluate MLP-only INT8 weight-only quantization for StreamingLLM.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "experiments"))

from streaming_llm import StartRecentKVCache, StreamingLLMWrapper
from eval_utils import load_tokenized_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MLP-only INT8 weight-only quantization evaluation"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=os.environ.get("MODEL_NAME", "EleutherAI/pythia-2.8b"),
        help="Model name"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code"
    )
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
        help="Text column"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=64,
        help="Max samples"
    )
    parser.add_argument(
        "--max-eval-tokens",
        type=int,
        default=int(os.environ.get("MAX_EVAL_TOKENS", "4096")),
        help="Max eval tokens"
    )
    parser.add_argument(
        "--n-sink",
        type=int,
        default=int(os.environ.get("N_SINK", "4")),
        help="Sink tokens"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=int(os.environ.get("WINDOW_SIZE", "2048")),
        help="Window size"
    )
    parser.add_argument(
        "--streaming-mode",
        type=str,
        choices=["ours", "mit"],
        default="ours",
        help="Streaming implementation"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Max length per window"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1024,
        help="Stride for sliding eval"
    )
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=512,
        help="Prompt length for latency eval"
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=2000,
        help="Generated tokens for latency eval"
    )
    parser.add_argument(
        "--warmup-tokens",
        type=int,
        default=200,
        help="Warmup tokens"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Latency eval repetitions"
    )
    parser.add_argument(
        "--print-quantized",
        action="store_true",
        help="Print which modules are quantized"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/quantization/mlp_only_int8.json"),
        help="Output JSON path"
    )
    return parser.parse_args()


def _mlp_filter(module: torch.nn.Module, fqn: str) -> bool:
    if not isinstance(module, torch.nn.Linear):
        return False
    return ("mlp.dense_h_to_4h" in fqn) or ("mlp.dense_4h_to_h" in fqn)


def _quantize_mlp_only(model: torch.nn.Module, print_modules: bool = False) -> int:
    try:
        from torchao.quantization import IntxWeightOnlyConfig, quantize_
    except Exception as exc:
        raise RuntimeError("torchao is required for MLP-only INT8 quantization.") from exc

    if print_modules:
        for name, module in model.named_modules():
            if _mlp_filter(module, name):
                print(f"quantize: {name}")

    config = IntxWeightOnlyConfig(weight_dtype=torch.int8, version=2)
    quantize_(model, config, filter_fn=_mlp_filter)
    return sum(1 for name, module in model.named_modules() if _mlp_filter(module, name))


def _compute_ppl_decode_loop(
    model,
    encoded_dataset: Tensor,
    device: torch.device,
    max_cache_size: int,
    streaming_wrapper=None,
    desc: str = "ppl",
) -> Dict[str, float]:
    seq_len = encoded_dataset.size(1)
    if seq_len < 2:
        raise ValueError("Dataset is too short to compute perplexity (seq_len < 2)")

    if encoded_dataset.device != device:
        encoded_dataset = encoded_dataset.to(device)

    max_cache_size = max(2, min(max_cache_size, seq_len))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    use_cuda_timing = device.type == "cuda" and torch.cuda.is_available()
    if use_cuda_timing:
        start_evt = torch.cuda.Event(enable_timing=True)
        prefill_end_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        first_start_evt = torch.cuda.Event(enable_timing=True)
        first_end_evt = torch.cuda.Event(enable_timing=True)
    else:
        total_start = time.perf_counter()
        prefill_start = time.perf_counter()

    total_nll = 0.0
    total_tokens = 0
    first_token_time = 0.0

    def _loss_from_logits(logits: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(logits.float(), target, reduction="sum")

    if streaming_wrapper is None:
        prefill_len = min(max_cache_size, seq_len)
        input_ids = encoded_dataset[:, :prefill_len]
        if use_cuda_timing:
            start_evt.record()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=False)
        if use_cuda_timing:
            prefill_end_evt.record()

        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        if labels.numel() > 0:
            total_nll += _loss_from_logits(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
            ).item()
            total_tokens += labels.numel()

        if use_cuda_timing:
            first_token_recorded = False
        else:
            prefill_time = time.perf_counter() - prefill_start
            first_token_recorded = False

        for pos in tqdm(range(prefill_len - 1, seq_len - 1), desc=desc):
            start_idx = max(0, pos + 1 - max_cache_size)
            context = encoded_dataset[:, start_idx:pos + 1]
            target = encoded_dataset[:, pos + 1]

            if use_cuda_timing and not first_token_recorded:
                first_start_evt.record()
            with torch.no_grad():
                outputs = model(input_ids=context, use_cache=False)
            if use_cuda_timing and not first_token_recorded:
                first_end_evt.record()

            logits = outputs.logits[:, -1, :]
            total_nll += _loss_from_logits(logits, target).item()
            total_tokens += target.numel()

            if not first_token_recorded:
                first_token_recorded = True
    else:
        prefill_len = min(max_cache_size, seq_len)
        past_key_values = None
        with streaming_wrapper.enable():
            input_ids = encoded_dataset[:, :prefill_len]
            if use_cuda_timing:
                start_evt.record()
            with torch.no_grad():
                outputs = model(input_ids=input_ids, use_cache=True)

            logits = outputs.logits[:, :-1, :]
            labels = input_ids[:, 1:]
            if labels.numel() > 0:
                total_nll += _loss_from_logits(
                    logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
                ).item()
                total_tokens += labels.numel()

            past_key_values = outputs.past_key_values
            streaming_wrapper.update(past_key_values)
            if use_cuda_timing:
                prefill_end_evt.record()

            if use_cuda_timing:
                first_token_recorded = False
            else:
                prefill_time = time.perf_counter() - prefill_start
                first_token_recorded = False

            for pos in tqdm(range(prefill_len - 1, seq_len - 1), desc=desc):
                current_input = encoded_dataset[:, pos:pos + 1]
                target = encoded_dataset[:, pos + 1]

                if use_cuda_timing and not first_token_recorded:
                    first_start_evt.record()
                with torch.no_grad():
                    outputs = model(
                        input_ids=current_input,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                if use_cuda_timing and not first_token_recorded:
                    first_end_evt.record()

                logits = outputs.logits[:, -1, :]
                total_nll += _loss_from_logits(logits, target).item()
                total_tokens += target.numel()
                past_key_values = outputs.past_key_values
                streaming_wrapper.update(past_key_values)

                if not first_token_recorded:
                    first_token_recorded = True

    if use_cuda_timing:
        end_evt.record()
        torch.cuda.synchronize()
        total_time = start_evt.elapsed_time(end_evt) / 1000.0
        prefill_time = start_evt.elapsed_time(prefill_end_evt) / 1000.0
        if total_tokens > 0:
            first_token_time = first_start_evt.elapsed_time(first_end_evt) / 1000.0
    else:
        total_time = time.perf_counter() - total_start

    ppl = torch.exp(torch.tensor(total_nll / total_tokens))
    return {
        "perplexity": ppl.item(),
        "runtime_sec": total_time,
        "prefill_sec": prefill_time,
        "first_token_latency_sec": first_token_time,
        "peak_memory_mb": (
            torch.cuda.max_memory_allocated() / 1024 / 1024
            if torch.cuda.is_available()
            else 0.0
        ),
    }


def _generate_random_prompt(tokenizer, length: int) -> torch.Tensor:
    torch.manual_seed(42)
    vocab_size = len(tokenizer)
    token_ids = torch.randint(100, vocab_size - 100, (1, length))
    return token_ids


def _warmup_model(
    model,
    input_ids: torch.Tensor,
    device: torch.device,
    num_tokens: int,
    use_streaming: bool = False,
    streaming_wrapper=None,
) -> None:
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
                    use_cache=True,
                )
                next_token = outputs.logits[:, -1:].argmax(dim=-1)
                current_ids = next_token
                past_key_values = outputs.past_key_values
                if use_streaming and streaming_wrapper is not None:
                    streaming_wrapper.update(past_key_values)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _measure_latency(
    model,
    input_ids: torch.Tensor,
    device: torch.device,
    num_tokens: int,
    use_streaming: bool = False,
    streaming_wrapper=None,
) -> Tuple[List[float], float]:
    latencies: List[float] = []
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
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                outputs = model(
                    input_ids=current_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies.append(t1 - t0)
                next_token = outputs.logits[:, -1:].argmax(dim=-1)
                current_ids = next_token
                past_key_values = outputs.past_key_values
                if use_streaming and streaming_wrapper is not None:
                    streaming_wrapper.update(past_key_values)

    peak_memory_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024
        if torch.cuda.is_available()
        else 0.0
    )
    return latencies, peak_memory_mb


def _eval_latency(
    model,
    tokenizer,
    device: torch.device,
    args: argparse.Namespace,
    use_streaming: bool,
) -> Dict[str, float]:
    all_latencies: List[float] = []
    all_memories: List[float] = []

    for _ in tqdm(range(args.num_runs), desc="latency runs"):
        input_ids = _generate_random_prompt(tokenizer, args.prompt_length)

        streaming_wrapper = None
        if use_streaming:
            streaming_wrapper = StreamingLLMWrapper(
                model=model,
                n_sink=args.n_sink,
                window_size=args.window_size,
            )

        _warmup_model(
            model=model,
            input_ids=input_ids,
            device=device,
            num_tokens=args.warmup_tokens,
            use_streaming=use_streaming,
            streaming_wrapper=streaming_wrapper,
        )

        latencies, peak_memory = _measure_latency(
            model=model,
            input_ids=input_ids,
            device=device,
            num_tokens=args.num_tokens,
            use_streaming=use_streaming,
            streaming_wrapper=streaming_wrapper,
        )

        all_latencies.extend(latencies)
        all_memories.append(peak_memory)

    mean_latency = float(np.mean(all_latencies)) if all_latencies else 0.0
    std_latency = float(np.std(all_latencies)) if len(all_latencies) > 1 else 0.0
    median_latency = float(np.median(all_latencies)) if all_latencies else 0.0
    mean_memory = float(np.mean(all_memories)) if all_memories else 0.0

    return {
        "mean_latency_ms": mean_latency * 1000,
        "std_latency_ms": std_latency * 1000,
        "median_latency_ms": median_latency * 1000,
        "mean_memory_mb": mean_memory,
    }


def _build_streaming_wrapper(model, args):
    cache_impl = None
    if args.streaming_mode == "mit":
        cache_impl = StartRecentKVCache(
            start_size=args.n_sink,
            recent_size=args.window_size,
            k_seq_dim=2,
            v_seq_dim=2,
        )
    return StreamingLLMWrapper(
        model=model,
        n_sink=args.n_sink,
        window_size=args.window_size,
        cache=cache_impl,
    )


def _run_ppl_and_latency(
    model,
    encoded_dataset: Tensor,
    tokenizer,
    device: torch.device,
    args: argparse.Namespace,
    use_streaming: bool,
) -> Dict[str, Dict[str, float]]:
    target_cache = args.n_sink + args.window_size
    streaming_wrapper = _build_streaming_wrapper(model, args) if use_streaming else None
    ppl_metrics = _compute_ppl_decode_loop(
        model=model,
        encoded_dataset=encoded_dataset,
        device=device,
        max_cache_size=target_cache,
        streaming_wrapper=streaming_wrapper,
        desc="ppl_streaming" if use_streaming else "ppl_baseline",
    )
    latency_metrics = _eval_latency(
        model=model,
        tokenizer=tokenizer,
        device=device,
        args=args,
        use_streaming=use_streaming,
    )
    return {"ppl": ppl_metrics, "latency": latency_metrics}


def main() -> None:
    args = parse_args()
    target_cache = args.n_sink + args.window_size
    if args.max_length < target_cache:
        args.max_length = target_cache

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_mapping[args.dtype]

    if device.type == "cpu" and torch_dtype != torch.float32:
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
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

    streaming_cache_name = (
        "StartRecentKVCache" if args.streaming_mode == "mit" else "StreamingKVCache"
    )

    # 1) Baseline (no quant)
    model_base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model_base.eval()
    baseline = _run_ppl_and_latency(
        model=model_base,
        encoded_dataset=encoded_dataset,
        tokenizer=tokenizer,
        device=device,
        args=args,
        use_streaming=False,
    )

    # 2) Streaming (no quant)
    streaming = _run_ppl_and_latency(
        model=model_base,
        encoded_dataset=encoded_dataset,
        tokenizer=tokenizer,
        device=device,
        args=args,
        use_streaming=True,
    )

    # 3) Streaming + MLP-only INT8 (v2)
    model_quant = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model_quant.eval()
    quantized_count = _quantize_mlp_only(model_quant, print_modules=args.print_quantized)
    streaming_quant = _run_ppl_and_latency(
        model=model_quant,
        encoded_dataset=encoded_dataset,
        tokenizer=tokenizer,
        device=device,
        args=args,
        use_streaming=True,
    )

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
        "quantization": {
            "type": "int8wo_mlp_only_v2",
            "quantized_modules": quantized_count,
        },
        "device": str(device),
        "dtype": str(torch_dtype),
        "baseline": baseline["ppl"],
        "streaming": streaming["ppl"],
        "streaming_quant": streaming_quant["ppl"],
        "latency_baseline": baseline["latency"],
        "latency_streaming": streaming["latency"],
        "latency_streaming_quant": streaming_quant["latency"],
    }

    if results["baseline"].get("runtime_sec", 0) > 0:
        results["metrics"] = {
            "ppl_increase_percent": (
                (results["streaming"]["perplexity"] - results["baseline"]["perplexity"])
                / results["baseline"]["perplexity"]
            )
            * 100,
            "speedup": results["baseline"]["runtime_sec"] / results["streaming"]["runtime_sec"],
        }
    if results["latency_baseline"].get("mean_latency_ms", 0) > 0:
        results.setdefault("metrics", {})
        results["metrics"]["latency_speedup"] = (
            results["latency_baseline"]["mean_latency_ms"]
            / results["latency_streaming"]["mean_latency_ms"]
            if results["latency_streaming"].get("mean_latency_ms")
            else 0.0
        )
        results["metrics"]["latency_speedup_quant"] = (
            results["latency_baseline"]["mean_latency_ms"]
            / results["latency_streaming_quant"]["mean_latency_ms"]
            if results["latency_streaming_quant"].get("mean_latency_ms")
            else 0.0
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
