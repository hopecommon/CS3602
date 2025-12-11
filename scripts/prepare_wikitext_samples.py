#!/usr/bin/env python3
"""Generate long-context WikiText-103 samples for offline streaming eval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare WikiText long-context samples")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("wikitext"),
        help="Directory to store extracted samples",
    )
    parser.add_argument(
        "--model-name",
        default="EleutherAI/pythia-2.8b",
        help="Model tokenizer used to count tokens"
    )
    parser.add_argument(
        "--dataset-config",
        default="wikitext-103-v1",
        help="WikiText configuration"
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to sample"
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        required=True,
        help="Target token lengths for each sample"
    )
    parser.add_argument(
        "--samples-per-length",
        type=int,
        default=1,
        help="Number of samples to collect per target length"
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=500,
        help="Max paragraphs to inspect from the split"
    )
    parser.add_argument(
        "--filename-template",
        type=str,
        default="long_context_{length}.json",
        help="Filename template inside output-dir (supports {length} and {idx})"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(
        "wikitext",
        args.dataset_config,
        split=args.split,
        streaming=True,
    )

    targets = sorted(set(args.lengths))
    quota: Dict[int, int] = {t: args.samples_per_length for t in targets}
    per_length_written: Dict[int, int] = {t: 0 for t in targets}
    buffers: Dict[int, List[str]] = {t: [] for t in targets}
    samples: List[dict] = []

    processed = 0
    for example in dataset:
        processed += 1
        if processed > args.max_articles:
            break
        text = example.get("text", "").strip()
        if not text:
            continue
        for target in targets:
            if quota[target] <= 0:
                continue
            buffers[target].append(text)
            combined = "\n\n".join(buffers[target])
            token_ids = tokenizer(combined, add_special_tokens=False)["input_ids"]
            if len(token_ids) < target:
                continue
            truncated = tokenizer.decode(token_ids[:target], skip_special_tokens=False)
            samples.append(
                {
                    "source_idx": processed,
                    "split": args.split,
                    "dataset_config": args.dataset_config,
                    "text": truncated,
                    "target_tokens": target,
                    "available_tokens": len(token_ids),
                }
            )
            quota[target] -= 1
            buffers[target] = []
        if all(count <= 0 for count in quota.values()):
            break

    pending = {t: q for t, q in quota.items() if q > 0}
    if pending:
        raise RuntimeError(f"Unable to collect targets {pending}; increase --max-articles or lower lengths")

    for sample in samples:
        length = sample["target_tokens"]
        idx = per_length_written[length]
        filename = args.filename_template.format(length=length, idx=idx)
        output_path = args.output_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(sample, f, ensure_ascii=False, indent=2)
        per_length_written[length] += 1
        print(f"Wrote WikiText sample: {output_path}")


if __name__ == "__main__":
    main()
