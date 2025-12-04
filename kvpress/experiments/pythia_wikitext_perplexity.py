# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvpress import KnormPress


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate KVPress on Pythia-70m with WikiText.")
    parser.add_argument("--model-name", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--dataset-name", type=str, default="wikitext")
    parser.add_argument("--dataset-config", type=str, default="wikitext-103-v1")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--max-samples", type=int, default=128, help="Maximum number of text samples to read.")
    parser.add_argument("--max-length", type=int, default=1024, help="Sequence length evaluated at once.")
    parser.add_argument("--stride", type=int, default=512, help="Stride used when sliding across the evaluation text.")
    parser.add_argument("--press-name", type=str, default="knorm", help="Currently only 'knorm' is supported.")
    parser.add_argument("--compression-ratio", type=float, default=0.5)
    parser.add_argument("--dtype", type=str, default="float16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--output", type=Path, default=Path("experiments/results/pythia_wikitext.json"))
    parser.add_argument(
        "--max-eval-tokens",
        type=int,
        default=None,
        help="Cap the total number of tokens evaluated after concatenation (prevents huge runtimes).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow running custom dataset loading code (required for some datasets like PG19).",
    )
    return parser.parse_args()


def load_tokenized_dataset(
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    text_column: str,
    max_samples: int | None,
    tokenizer,
    *,
    split_expression: str | None = None,
    trust_remote_code: bool = False,
    max_eval_tokens: int | None = None,
    use_streaming: bool = False,
) -> torch.Tensor:
    split_value = split if use_streaming else (split_expression or split)
    dataset_kwargs = {"split": split_value, "trust_remote_code": trust_remote_code}
    if use_streaming:
        dataset_kwargs["streaming"] = True
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, **dataset_kwargs)
    else:
        dataset = load_dataset(dataset_name, **dataset_kwargs)
    if not use_streaming and max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    texts: list[str] = []
    for idx, row in enumerate(dataset):
        if use_streaming and max_samples and idx >= max_samples:
            break
        text = row.get(text_column, "")
        if text and not text.isspace():
            texts.append(text)

    if not texts:
        raise ValueError(f"No non-empty rows found for dataset {dataset_name} ({split_value}).")

    concatenated = "\n\n".join(texts)

    encodings = tokenizer(concatenated, return_tensors="pt")
    input_ids = encodings.input_ids
    if max_eval_tokens is not None and max_eval_tokens > 0:
        input_ids = input_ids[:, :max_eval_tokens]
    return input_ids


def create_press(name: str, compression_ratio: float):
    normalized = name.lower()
    if normalized != "knorm":
        raise ValueError(f"Unsupported press '{name}'. Only 'knorm' is currently wired in this script.")
    return KnormPress(compression_ratio=compression_ratio)


def compute_perplexity(
    model,
    encoded_dataset: torch.Tensor,
    *,
    device: torch.device,
    max_length: int,
    stride: int,
    press=None,
    track_time: bool = False,
) -> float | tuple[float, float, float]:
    nlls = []
    total_tokens = 0
    seq_len = encoded_dataset.size(1)
    context = press(model) if press is not None else nullcontext()
    total_start = time.perf_counter() if track_time else None
    prefill_start = None

    with context:
        if track_time:
            prefill_start = time.perf_counter()
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
    if track_time and total_start is not None and prefill_start is not None:
        prefill_time = time.perf_counter() - prefill_start
        total_time = time.perf_counter() - total_start
        return ppl.item(), total_time, prefill_time
    return ppl.item()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_mapping[args.dtype]
    if device.type == "cpu" and torch_dtype != torch.float32:
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded_dataset = load_tokenized_dataset(
        args.dataset_name,
        args.dataset_config,
        args.split,
        args.text_column,
        args.max_samples,
        tokenizer,
        trust_remote_code=args.trust_remote_code,
        max_eval_tokens=args.max_eval_tokens,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
    ).to(device)
    model.eval()

    baseline_start = time.perf_counter()
    baseline_ppl = compute_perplexity(
        model,
        encoded_dataset,
        device=device,
        max_length=args.max_length,
        stride=args.stride,
    )
    baseline_time = time.perf_counter() - baseline_start

    press = create_press(args.press_name, args.compression_ratio)
    press_start = time.perf_counter()
    press_ppl = compute_perplexity(
        model,
        encoded_dataset,
        device=device,
        max_length=args.max_length,
        stride=args.stride,
        press=press,
    )
    press_time = time.perf_counter() - press_start

    results = {
        "model": args.model_name,
        "dataset": f"{args.dataset_name}:{args.dataset_config}",
        "split": args.split,
        "max_length": args.max_length,
        "stride": args.stride,
        "max_samples": args.max_samples,
        "trust_remote_code": args.trust_remote_code,
        "press": {
            "name": args.press_name,
            "compression_ratio": args.compression_ratio,
        },
        "max_eval_tokens": args.max_eval_tokens,
        "baseline": {
            "perplexity": baseline_ppl,
            "runtime_sec": baseline_time,
        },
        "compressed": {
            "perplexity": press_ppl,
            "runtime_sec": press_time,
        },
        "device": str(device),
        "dtype": str(torch_dtype),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
