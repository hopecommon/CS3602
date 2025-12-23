#!/usr/bin/env python3
"""
Sweep overlap under auto window cap (streaming-only).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlap sweep with auto window cap")
    parser.add_argument("--dataset-name", type=str, default="wikitext")
    parser.add_argument("--dataset-config", type=str, default="wikitext-103-v1")
    parser.add_argument("--max-eval-tokens", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--n-sink", type=int, default=16)
    parser.add_argument("--auto-window-cap", type=int, default=2048)
    parser.add_argument("--overlaps", type=int, nargs="+", default=[0, 64, 128, 256, 512, 1024])
    parser.add_argument("--compress-every", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=Path("results/overlap_auto_sweep"))
    parser.add_argument("--no-skip-existing", action="store_true")
    return parser.parse_args()


def _run_eval(
    overlap: int,
    window_size: int,
    args: argparse.Namespace,
    output_path: Path,
    baseline_results: Path,
) -> dict[str, Any] | None:
    if output_path.exists() and not args.no_skip_existing:
        return json.loads(output_path.read_text())

    max_length = window_size + args.n_sink + overlap
    stride = max(1, window_size // 2)

    cmd = [
        "python", "experiments/eval_streaming_llm.py",
        "--dtype", args.dtype,
        "--dataset-name", args.dataset_name,
        "--dataset-config", args.dataset_config,
        "--max-eval-tokens", str(args.max_eval_tokens),
        "--max-samples", str(args.max_samples),
        "--mode", "streaming",
        "--baseline-results", str(baseline_results),
        "--n-sink", str(args.n_sink),
        "--window-size", str(window_size),
        "--overlap", str(overlap),
        "--refresh-budget", "0",
        "--refresh-policy", "none",
        "--compress-every", str(args.compress_every),
        "--max-length", str(max_length),
        "--stride", str(stride),
        "--output", str(output_path),
    ]

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0 or not output_path.exists():
        print(f"错误: overlap={overlap} window={window_size} 运行失败")
        return None
    return json.loads(output_path.read_text())


def _flatten(data: dict[str, Any], overlap: int, window_size: int, cap: int) -> dict[str, Any]:
    streaming = data.get("streaming", {})
    metrics = data.get("metrics", {})
    return {
        "dataset": data.get("dataset"),
        "overlap": overlap,
        "window_size": window_size,
        "n_sink": data.get("streaming_llm", {}).get("n_sink"),
        "auto_window_cap": cap,
        "streaming_ppl": streaming.get("perplexity"),
        "tpot_ms": streaming.get("tpot_ms"),
        "runtime_sec": streaming.get("runtime_sec"),
        "ppl_increase_percent": metrics.get("ppl_increase_percent"),
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

    baseline_path = args.output_dir / f"{args.dataset_name}_baseline.json"
    if args.no_skip_existing or not baseline_path.exists():
        baseline_window = args.auto_window_cap - args.n_sink
        baseline_cmd = [
            "python", "experiments/eval_streaming_llm.py",
            "--dtype", args.dtype,
            "--dataset-name", args.dataset_name,
            "--dataset-config", args.dataset_config,
            "--max-eval-tokens", str(args.max_eval_tokens),
            "--max-samples", str(args.max_samples),
            "--mode", "baseline",
            "--n-sink", str(args.n_sink),
            "--window-size", str(baseline_window),
            "--overlap", "0",
            "--refresh-budget", "0",
            "--refresh-policy", "none",
            "--compress-every", str(args.compress_every),
            "--max-length", str(baseline_window + args.n_sink),
            "--stride", str(max(1, baseline_window // 2)),
            "--output", str(baseline_path),
        ]
        result = subprocess.run(baseline_cmd, check=False)
        if result.returncode != 0 or not baseline_path.exists():
            raise RuntimeError("Baseline failed")

    rows: list[dict[str, Any]] = []
    for overlap in args.overlaps:
        window_size = args.auto_window_cap - args.n_sink - overlap
        if window_size < 1:
            print(f"跳过: overlap={overlap} 导致 window<1")
            continue
        output_path = args.output_dir / (
            f"{args.dataset_name}_w{window_size}_s{args.n_sink}_o{overlap}_c{args.compress_every}.json"
        )
        data = _run_eval(overlap, window_size, args, output_path, baseline_results=baseline_path)
        if data is None:
            continue
        rows.append(_flatten(data, overlap, window_size, args.auto_window_cap))

    summary = {
        "dataset": args.dataset_name,
        "dataset_config": args.dataset_config,
        "max_eval_tokens": args.max_eval_tokens,
        "n_sink": args.n_sink,
        "auto_window_cap": args.auto_window_cap,
        "overlaps": args.overlaps,
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_csv(rows, args.output_dir / "summary.csv")


if __name__ == "__main__":
    main()
