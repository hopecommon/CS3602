#!/usr/bin/env python3
"""
Sweep StreamingLLM compress_every for lazy/periodic KV pruning.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep compress_every for lazy KV pruning"
    )
    parser.add_argument(
        "--compress-every",
        type=int,
        nargs="+",
        default=[4, 16, 32],
        help="Prune interval (tokens)"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=2048,
        help="Streaming window size"
    )
    parser.add_argument(
        "--n-sink",
        type=int,
        default=4,
        help="Sink size"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["wikitext", "pg19"],
        help="Datasets to test (wikitext, pg19)"
    )
    parser.add_argument(
        "--wikitext-config",
        type=str,
        default="wikitext-103-v1",
        help="Dataset config for wikitext"
    )
    parser.add_argument(
        "--wikitext-max-eval-tokens",
        type=int,
        default=4096,
        help="Max eval tokens for wikitext"
    )
    parser.add_argument(
        "--pg19-max-eval-tokens",
        type=int,
        default=20000,
        help="Max eval tokens for pg19"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=64,
        help="Max samples to load (wikitext only)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/lazy_prune_sweep"),
        help="Output directory"
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-run even if output JSON already exists"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype"
    )
    parser.add_argument(
        "--streaming-mode",
        type=str,
        choices=["ours", "mit"],
        default="ours",
        help="Streaming implementation"
    )
    return parser.parse_args()


def _dataset_args(dataset: str, args: argparse.Namespace) -> list[str]:
    if dataset == "wikitext":
        return [
            "--dataset-name", "wikitext",
            "--dataset-config", args.wikitext_config,
            "--max-eval-tokens", str(args.wikitext_max_eval_tokens),
            "--max-samples", str(args.max_samples),
        ]
    if dataset == "pg19":
        return [
            "--dataset-name", "pg19",
            "--dataset-config", "pg19",
            "--max-eval-tokens", str(args.pg19_max_eval_tokens),
            "--max-samples", "1",
        ]
    raise ValueError(f"Unsupported dataset: {dataset}")


def _run_baseline(
    dataset: str,
    args: argparse.Namespace,
    output_path: Path,
) -> dict[str, Any] | None:
    if output_path.exists() and not args.no_skip_existing:
        print(f"跳过基线 (已存在): {output_path}")
        return json.loads(output_path.read_text())

    max_length = args.window_size + args.n_sink
    stride = max(1, args.window_size // 2)

    cmd = [
        "python", "experiments/eval_streaming_llm.py",
        "--dtype", args.dtype,
        "--n-sink", str(args.n_sink),
        "--window-size", str(args.window_size),
        "--compress-every", str(min(args.compress_every)),
        "--max-length", str(max_length),
        "--stride", str(stride),
        "--streaming-mode", args.streaming_mode,
        "--mode", "baseline",
        "--output", str(output_path),
    ]
    cmd += _dataset_args(dataset, args)

    print("\n" + "=" * 72)
    print(f"Baseline | dataset={dataset} | cache={args.window_size}+{args.n_sink}")
    print("=" * 72)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"错误: 基线运行失败 (dataset={dataset})")
        return None
    if not output_path.exists():
        print(f"错误: 未生成基线输出文件 {output_path}")
        return None
    return json.loads(output_path.read_text())


def _run_eval(
    dataset: str,
    compress_every: int,
    args: argparse.Namespace,
    output_path: Path,
    baseline_results: Path,
) -> dict[str, Any] | None:
    if output_path.exists() and not args.no_skip_existing:
        print(f"跳过 (已存在): {output_path}")
        return json.loads(output_path.read_text())

    max_length = args.window_size + args.n_sink
    stride = max(1, args.window_size // 2)

    cmd = [
        "python", "experiments/eval_streaming_llm.py",
        "--dtype", args.dtype,
        "--n-sink", str(args.n_sink),
        "--window-size", str(args.window_size),
        "--compress-every", str(compress_every),
        "--max-length", str(max_length),
        "--stride", str(stride),
        "--streaming-mode", args.streaming_mode,
        "--mode", "streaming",
        "--baseline-results", str(baseline_results),
        "--output", str(output_path),
    ]
    cmd += _dataset_args(dataset, args)

    print("\n" + "=" * 72)
    print(f"Dataset: {dataset} | compress_every={compress_every}")
    print("=" * 72)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"错误: 运行失败 (compress_every={compress_every}, dataset={dataset})")
        return None
    if not output_path.exists():
        print(f"错误: 未生成输出文件 {output_path}")
        return None
    return json.loads(output_path.read_text())


def _flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    baseline = result.get("baseline", {})
    streaming = result.get("streaming", {})
    metrics = result.get("metrics", {})
    streaming_cfg = result.get("streaming_llm", {})
    dataset = result.get("dataset", "")

    return {
        "dataset": dataset,
        "window_size": streaming_cfg.get("window_size"),
        "n_sink": streaming_cfg.get("n_sink"),
        "compress_every": streaming_cfg.get("compress_every"),
        "baseline_ppl": baseline.get("perplexity"),
        "streaming_ppl": streaming.get("perplexity"),
        "ppl_increase_percent": metrics.get("ppl_increase_percent"),
        "baseline_runtime_sec": baseline.get("runtime_sec"),
        "streaming_runtime_sec": streaming.get("runtime_sec"),
        "speedup": metrics.get("speedup"),
        "baseline_tpot_ms": baseline.get("tpot_ms"),
        "streaming_tpot_ms": streaming.get("tpot_ms"),
        "baseline_peak_memory_mb": baseline.get("peak_memory_mb"),
        "streaming_peak_memory_mb": streaming.get("peak_memory_mb"),
    }


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_entries: list[dict[str, Any]] = []

    for dataset in args.datasets:
        baseline_path = args.output_dir / f"{dataset}_baseline.json"
        baseline_result = _run_baseline(dataset, args, baseline_path)
        if baseline_result is None:
            raise RuntimeError(f"Baseline failed for dataset {dataset}")

        for compress_every in args.compress_every:
            output_path = args.output_dir / f"{dataset}_c{compress_every}.json"
            result = _run_eval(
                dataset,
                compress_every,
                args,
                output_path,
                baseline_results=baseline_path,
            )
            if result is None:
                continue
            summary_entries.append(_flatten_result(result))

    summary = {
        "compress_every": args.compress_every,
        "datasets": args.datasets,
        "window_size": args.window_size,
        "n_sink": args.n_sink,
        "dtype": args.dtype,
        "streaming_mode": args.streaming_mode,
        "wikitext_max_eval_tokens": args.wikitext_max_eval_tokens,
        "pg19_max_eval_tokens": args.pg19_max_eval_tokens,
        "rows": summary_entries,
    }

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n已保存汇总: {summary_path}")

    csv_path = args.output_dir / "summary.csv"
    _write_csv(summary_entries, csv_path)
    print(f"已保存 CSV: {csv_path}")


if __name__ == "__main__":
    main()
