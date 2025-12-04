#!/usr/bin/env python3
"""
Run the decode-by-token perplexity evaluation for multiple StreamingLLM variants.

Supports:
- baseline: sliding window recomputation
- ours: our StreamingLLMWrapper
- mit: StreamingLLMWrapper with MIT-style StartRecentKVCache slicing
- kvpress: kvpress StreamingLLMPress + KeyRerotationPress (filtered to decode loop)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Tuple

repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo))

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from streaming_llm import StartRecentKVCache, StreamingLLMWrapper

spec = importlib.util.spec_from_file_location(
    "eval_utils", repo / "experiments" / "eval_utils.py"
)
eval_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_utils)

compute_perplexity = eval_utils.compute_perplexity
load_tokenized_dataset = eval_utils.load_tokenized_dataset
from kvpress import StreamingLLMPress, KeyRerotationPress


def parse_args():
    parser = argparse.ArgumentParser(description="Decode-loop PPL comparison")
    parser.add_argument("--method", choices=["baseline", "ours", "mit", "kvpress"], required=True)
    parser.add_argument("--model-name", default="EleutherAI/pythia-70m")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--dataset-name", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-103-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-eval-tokens", type=int, default=4096)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--n-sink", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=Path("results/decode_comparison.json"))
    return parser.parse_args()


def load_model(model_name: str, dtype: str, device: torch.device):
    dtype_mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype_arg = dtype_mapping[dtype]
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype_arg).to(device)
    model.eval()
    return model


def decode_loop_kvpress(
    model,
    encoded,
    n_sink: int,
    window_size: int,
) -> Tuple[float, float, float]:
    press = KeyRerotationPress(StreamingLLMPress(compression_ratio=0.498, n_sink=n_sink))
    seq_len = encoded.size(1)
    max_cache = n_sink + window_size

    total_nll = 0.0
    total_tokens = 0
    prefill_len = min(max_cache, seq_len)

    start = time.perf_counter()
    with press(model):
        inputs = encoded[:, :prefill_len].to(model.device)
        with torch.no_grad():
            outputs = model(input_ids=inputs, use_cache=True)
        logits = outputs.logits[:, :-1, :]
        labels = inputs[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="sum")
        total_nll += loss.item()
        total_tokens += labels.numel()
        past = outputs.past_key_values
        prefill_end = time.perf_counter()

        for pos in range(prefill_len - 1, seq_len - 1):
            token = encoded[:, pos : pos + 1].to(model.device)
            target = encoded[:, pos + 1].to(model.device)
            with torch.no_grad():
                outputs = model(input_ids=token, past_key_values=past, use_cache=True)
            logits = outputs.logits[:, -1, :]
            loss = F.cross_entropy(logits, target, reduction="sum")
            total_nll += loss.item()
            total_tokens += target.numel()
            past = outputs.past_key_values
    end = time.perf_counter()
    ppl = torch.exp(torch.tensor(total_nll / total_tokens))
    return ppl.item(), end - start, prefill_end - start


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded = load_tokenized_dataset(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        text_column="text",
        max_samples=args.max_samples,
        tokenizer=tokenizer,
        max_eval_tokens=args.max_eval_tokens,
    )
    encoded = encoded.to(device)

    model = load_model(args.model_name, args.dtype, device)

    result = {"method": args.method, "dataset": f"{args.dataset_name}:{args.dataset_config}"}
    if args.method == "kvpress":
        ppl, total_time, prefill_time = decode_loop_kvpress(model, encoded, args.n_sink, args.window_size)
        result.update(
            {
                "perplexity": ppl,
                "total_time": total_time,
                "prefill_time": prefill_time,
                "mode": "kvpress",
            }
        )
    else:
        streaming = args.method in {"ours", "mit"}
        wrapper = None
        if streaming:
            cache = StartRecentKVCache(
                start_size=args.n_sink,
                recent_size=args.window_size,
            ) if args.method == "mit" else None
            wrapper = StreamingLLMWrapper(
                model=model,
                n_sink=args.n_sink,
                window_size=args.window_size,
                cache=cache,
            )
        max_cache = args.n_sink + args.window_size
        ppl, total_time, prefill_time = compute_perplexity(
            model=model,
            encoded_dataset=encoded,
            device=device,
            max_length=max_cache,
            stride=args.stride,
            use_streaming=streaming,
            streaming_wrapper=wrapper,
            max_cache_size=max_cache,
        )
        result.update(
            {
                "perplexity": ppl,
                "total_time": total_time,
                "prefill_time": prefill_time,
                "mode": args.method,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
