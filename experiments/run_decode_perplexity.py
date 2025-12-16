#!/usr/bin/env python3
"""
Run the decode-by-token perplexity evaluation for multiple StreamingLLM variants.

Supports:
- baseline: sliding window recomputation
- ours: our StreamingLLMWrapper
- mit: StreamingLLMWrapper with MIT-style StartRecentKVCache slicing
- kvpress: KVPress DecodingPress(StreamingLLMPress) (decode-loop comparable)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
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
DEFAULT_MAX_EVAL_TOKENS = int(os.environ.get("MAX_EVAL_TOKENS", "4096"))


def parse_args():
    parser = argparse.ArgumentParser(description="Decode-loop PPL comparison")
    parser.add_argument("--method", choices=["baseline", "ours", "mit", "kvpress"], required=True)
    parser.add_argument("--model-name", default=os.environ.get("MODEL_NAME", "EleutherAI/pythia-2.8b"))
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--dataset-name", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-103-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-eval-tokens", type=int, default=DEFAULT_MAX_EVAL_TOKENS)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--n-sink", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument(
        "--compress-every",
        type=int,
        default=4,
        help="For StreamingLLMWrapper/KVPress: compression interval (in decode steps).",
    )
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
    
    streaming = args.method in {"ours", "mit", "kvpress"}
    wrapper = None
    if args.method in {"ours", "mit"}:
        cache = StartRecentKVCache(
            start_size=args.n_sink,
            recent_size=args.window_size,
        ) if args.method == "mit" else None
        wrapper = StreamingLLMWrapper(
            model=model,
            n_sink=args.n_sink,
            window_size=args.window_size,
            cache=cache,
            compress_every=args.compress_every,
        )
    elif args.method == "kvpress":
        # Ensure vendored kvpress resolves to kvpress/kvpress/.
        sys.path.insert(0, str(repo / "kvpress"))
        from kvpress import DecodingPress, PrefillDecodingPress, StreamingLLMPress

        class _KVPressAdapter:
            def __init__(self, model, press):
                self._model = model
                self._press = press

            def enable(self):
                return self._press(self._model)

            def update(self, past_key_values):
                return None

        max_cache = args.n_sink + args.window_size
        decoding_press = DecodingPress(
            base_press=StreamingLLMPress(n_sink=args.n_sink),
            compression_interval=max(1, int(args.compress_every)),
            target_size=max_cache,
            hidden_states_buffer_size=0,
        )
        press = PrefillDecodingPress(prefilling_press=None, decoding_press=decoding_press)
        wrapper = _KVPressAdapter(model=model, press=press)
    max_cache = args.n_sink + args.window_size
    stats = compute_perplexity(
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
            "perplexity": stats.perplexity,
            "total_time": stats.runtime_sec,
            "prefill_time": stats.prefill_sec,
            "first_token_latency_sec": stats.first_token_latency_sec,
            "mode": args.method,
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
