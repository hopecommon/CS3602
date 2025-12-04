# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pythia_wikitext_perplexity import create_press, compute_perplexity, load_tokenized_dataset

DEFAULT_DATASETS = {
    "wikitext": {
        "dataset_name": "wikitext",
        "dataset_config": "wikitext-103-v1",
        "split": "test",
        "text_column": "text",
        "max_samples": 32,
        "max_eval_tokens": None,
        "trust_remote_code": False,
    },
    "pg19": {
        "dataset_name": "pg19",
        "dataset_config": None,
        "split": "test",
        "split_expr": "test[:1]",
        "text_column": "text",
        "max_samples": 1,
        "max_eval_tokens": None,
        "trust_remote_code": True,
        "use_streaming": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep KVPress compression ratios on multiple datasets.")
    parser.add_argument("--model-name", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--dtype", type=str, default="float16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--press-name", type=str, default="knorm")
    parser.add_argument(
        "--compression-ratios",
        type=float,
        nargs="+",
        default=None,
        help="Custom list of compression ratios (fraction of KV tokens pruned). If omitted, "
        "a dense grid with step 0.1 plus {0.25,0.5,0.75} is used.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["wikitext", "pg19"],
        choices=sorted(DEFAULT_DATASETS.keys()),
        help="Datasets to include in the sweep.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Override the default number of text samples per dataset (applied to all datasets).",
    )
    parser.add_argument("--output", type=Path, default=Path("experiments/results/pythia_kvpress_sweep.json"))
    parser.add_argument("--figure", type=Path, default=Path("experiments/results/pythia_kvpress_tradeoff.png"))
    parser.add_argument(
        "--max-eval-tokens",
        type=int,
        default=None,
        help="Override the default token budget per dataset after concatenation (applied to all datasets).",
    )
    return parser.parse_args()


def prepare_model_and_tokenizer(model_name: str, dtype: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_mapping[dtype]
    if device.type == "cpu" and torch_dtype != torch.float32:
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype).to(device)
    model.eval()
    return model, tokenizer, device, torch_dtype


def measure_dataset(
    model,
    encoded_dataset: torch.Tensor,
    device: torch.device,
    *,
    max_length: int,
    stride: int,
    press_name: str,
    compression_ratios: list[float],
    dataset_label: str,
) -> list[dict]:
    dataset_results = []
    for idx, ratio in enumerate(compression_ratios):
        if not (0 <= ratio < 1):
            raise ValueError(f"Compression ratio must lie in [0,1): received {ratio}")
        press = None if ratio == 0 else create_press(press_name, ratio)
        print(
            f"[{datetime.now():%H:%M:%S}] Dataset={dataset_label} "
            f"({idx + 1}/{len(compression_ratios)} ratios) -> compression={ratio:.2f}"
        )
        start = time.perf_counter()
        ppl = compute_perplexity(
            model,
            encoded_dataset,
            device=device,
            max_length=max_length,
            stride=stride,
            press=press,
        )
        runtime = time.perf_counter() - start
        dataset_results.append({"compression_ratio": ratio, "perplexity": ppl, "runtime_sec": runtime})
    return dataset_results


def plot_results(summary: dict, figure_path: Path):
    plt.figure(figsize=(10, 4))
    axes = [plt.subplot(1, 2, 1), plt.subplot(1, 2, 2)]
    for dataset in summary["datasets"]:
        ratios = [entry["compression_ratio"] for entry in dataset["results"]]
        perplexities = [entry["perplexity"] for entry in dataset["results"]]
        runtimes = [entry["runtime_sec"] for entry in dataset["results"]]
        label = dataset["name"]
        axes[0].plot(ratios, perplexities, marker="o", label=label)
        axes[1].plot(ratios, runtimes, marker="o", label=label)

    axes[0].set_title("Perplexity vs. Compression")
    axes[0].set_xlabel("Compression ratio (fraction pruned)")
    axes[0].set_ylabel("Perplexity")
    axes[1].set_title("Prefill runtime vs. Compression")
    axes[1].set_xlabel("Compression ratio (fraction pruned)")
    axes[1].set_ylabel("Runtime per pass (s)")
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xticks(summary["compression_ratios"])
        ax.legend()

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    if args.compression_ratios:
        compression_ratios = sorted({round(r, 4) for r in args.compression_ratios if 0 <= r < 1})
    else:
        dense_grid = {round(i / 10, 2) for i in range(0, 10)}  # 0.0 to 0.9 inclusive
        specials = {0.25, 0.5, 0.75}
        compression_ratios = sorted({round(r, 4) for r in dense_grid.union(specials) if 0 <= r < 1})
    if not compression_ratios:
        raise ValueError("No valid compression ratios specified (must be in [0,1)).")

    model, tokenizer, device, torch_dtype = prepare_model_and_tokenizer(args.model_name, args.dtype)

    summary = {
        "model": args.model_name,
        "dtype": str(torch_dtype),
        "device": str(device),
        "max_length": args.max_length,
        "stride": args.stride,
        "press_name": args.press_name,
        "compression_ratios": compression_ratios,
        "datasets": [],
    }

    for dataset_key in args.datasets:
        cfg = DEFAULT_DATASETS[dataset_key]
        max_samples = args.max_samples or cfg["max_samples"]
        max_eval_tokens = args.max_eval_tokens or cfg.get("max_eval_tokens")
        print(
            f"[{datetime.now():%H:%M:%S}] Loading dataset '{dataset_key}' "
            f"with {max_samples} samples and token cap {max_eval_tokens}."
        )
        encoded_dataset = load_tokenized_dataset(
            cfg["dataset_name"],
            cfg.get("dataset_config"),
            cfg["split"],
            cfg["text_column"],
            max_samples,
            tokenizer,
            split_expression=cfg.get("split_expr"),
            trust_remote_code=cfg.get("trust_remote_code", False),
            max_eval_tokens=max_eval_tokens,
            use_streaming=cfg.get("use_streaming", False),
        )
        print(
            f"[{datetime.now():%H:%M:%S}] Dataset '{dataset_key}' tokenized: "
            f"{encoded_dataset.shape[1]} tokens total"
        )
        results = measure_dataset(
            model,
            encoded_dataset,
            device,
            max_length=args.max_length,
            stride=args.stride,
            press_name=args.press_name,
            compression_ratios=compression_ratios,
            dataset_label=dataset_key,
        )
        summary["datasets"].append(
            {
                "name": dataset_key,
                "dataset_name": cfg["dataset_name"],
                "dataset_config": cfg.get("dataset_config"),
                "split": cfg["split"],
                "text_column": cfg["text_column"],
                "max_samples": max_samples,
                "max_eval_tokens": max_eval_tokens,
                "trust_remote_code": cfg.get("trust_remote_code", False),
                "results": results,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    plot_results(summary, args.figure)
    print(json.dumps(summary, indent=2))
    print(f"Saved summary to {args.output}")
    print(f"Saved figure to {args.figure}")


if __name__ == "__main__":
    main()
