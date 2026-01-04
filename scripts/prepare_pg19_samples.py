#!/usr/bin/env python3
"""Download PG19 samples and build multi-length cache for streaming eval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare PG19 cached samples")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/pg19"),
        help="Directory where extracted sample files are written",
    )
    parser.add_argument(
        "--model-name",
        default="EleutherAI/pythia-2.8b",
    )
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=None,
        help="Optional local PG19 parquet directory (e.g., pg19/dir/data). "
        "If omitted, falls back to huggingface datasets streaming download.",
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        required=True,
        help="Target token lengths for each sample (sorted ascending recommended)",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "validation", "test"],
    )
    parser.add_argument(
        "--samples-per-length",
        type=int,
        default=1,
        help="How many samples to collect for each target length",
    )
    parser.add_argument(
        "--filename-template",
        type=str,
        default="long_context_{length}.json",
        help="Filename template inside output-dir. Supports {length} and {idx}.",
    )
    parser.add_argument(
        "--max-books",
        type=int,
        default=500,
        help="Maximum books to scan before giving up",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.parquet_dir:
        pattern = str((args.parquet_dir / f"{args.split}-*.parquet").resolve())
        dataset = load_dataset(
            "parquet",
            data_files={"data": pattern},
            split="data",
            streaming=True,
        )
    else:
        dataset = load_dataset("pg19", split=args.split, streaming=True)

    targets = sorted(set(args.lengths))
    if not targets:
        raise ValueError("--lengths must be non-empty")
    max_target = max(targets)

    # Collect samples in "sets": each chosen book yields one sample per target length.
    # This avoids accidentally switching the underlying book when you change only the length.
    quota_sets = int(args.samples_per_length)
    samples: List[dict] = []
    per_length_written: dict[int, int] = {t: 0 for t in targets}

    processed = 0
    for example in dataset:
        processed += 1
        if processed > args.max_books:
            break
        text = example.get("text", "").strip()
        if not text:
            continue
        token_ids = tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        token_count = len(token_ids)
        if token_count == 0:
            continue
        # Only pick books that can satisfy the largest target length, so all lengths come
        # from the same underlying text for a given set.
        if token_count < max_target:
            continue

        for target in targets:
            truncated = tokenizer.decode(token_ids[:target], skip_special_tokens=False)
            samples.append(
                {
                    "book_id": example.get("_id", ""),
                    "short_book_title": example.get("short_book_title", ""),
                    "publication_date": example.get("publication_date"),
                    "target_tokens": target,
                    "available_tokens": token_count,
                    "split": args.split,
                    "text": truncated,
                }
            )
        quota_sets -= 1
        if quota_sets <= 0:
            break

    if quota_sets > 0:
        raise RuntimeError(
            f"Unable to collect {args.samples_per_length} sample set(s) that reach {max_target} tokens; "
            "increase --max-books or lower --lengths"
        )

    for sample in samples:
        length = sample["target_tokens"]
        idx = per_length_written[length]
        filename = args.filename_template.format(length=length, idx=idx)
        out_path = args.output_dir / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(sample, f, ensure_ascii=False)
        per_length_written[length] += 1
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
