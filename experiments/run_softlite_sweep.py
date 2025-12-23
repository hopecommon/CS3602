#!/usr/bin/env python3
"""
Sweep soft-ring-lite settings (cache_slack + max_drop) for lazy pruning.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

from eval_utils import load_fixed_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep soft-ring-lite settings"
    )
    parser.add_argument("--dataset-name", type=str, default="pg19")
    parser.add_argument("--dataset-config", type=str, default="pg19")
    parser.add_argument("--max-eval-tokens", type=int, default=20000)
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--n-sink", type=int, default=32)
    parser.add_argument("--total-window", type=int, default=2048)
    parser.add_argument("--compress-every", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--cache-slack", type=int, nargs="+", default=[0, 16, 32])
    parser.add_argument("--max-drop", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--fixed-baseline-dir", type=Path, default=Path("results/baselines"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/softlite_sweep"))
    parser.add_argument("--no-skip-existing", action="store_true")
    return parser.parse_args()


def _run_eval(
    args: argparse.Namespace,
    compress_every: int,
    cache_slack: int,
    max_drop: int,
    output_path: Path,
    baseline_results: Path,
) -> dict[str, Any] | None:
    if output_path.exists() and not args.no_skip_existing:
        return json.loads(output_path.read_text())

    window_size = max(1, args.total_window - args.n_sink - cache_slack)
    max_length = args.total_window
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
        "--compress-every", str(compress_every),
        "--cache-slack", str(cache_slack),
        "--max-drop", str(max_drop),
        "--max-length", str(max_length),
        "--stride", str(stride),
        "--output", str(output_path),
    ]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0 or not output_path.exists():
        print(f"错误: 运行失败 c{compress_every} slack={cache_slack} max_drop={max_drop}")
        return None
    return json.loads(output_path.read_text())


def _flatten(result: dict[str, Any]) -> dict[str, Any]:
    baseline = result.get("baseline", {})
    streaming = result.get("streaming", {})
    metrics = result.get("metrics", {})
    streaming_cfg = result.get("streaming_llm", {})
    return {
        "dataset": result.get("dataset"),
        "n_sink": streaming_cfg.get("n_sink"),
        "window_size": streaming_cfg.get("window_size"),
        "cache_slack": streaming_cfg.get("cache_slack"),
        "max_drop": streaming_cfg.get("max_drop"),
        "compress_every": streaming_cfg.get("compress_every"),
        "baseline_ppl": baseline.get("perplexity"),
        "streaming_ppl": streaming.get("perplexity"),
        "ppl_increase_percent": metrics.get("ppl_increase_percent"),
        "baseline_tpot_ms": baseline.get("tpot_ms"),
        "streaming_tpot_ms": streaming.get("tpot_ms"),
        "speedup": metrics.get("speedup"),
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

    baseline = load_fixed_baseline(args.dataset_name, args.fixed_baseline_dir)
    if not baseline:
        raise RuntimeError("未找到固定 baseline，请先生成 results/baselines/*_baseline_avg.json")
    baseline_path = args.fixed_baseline_dir / f"{args.dataset_name.lower()}_baseline_avg.json"

    rows: list[dict[str, Any]] = []
    for compress_every in args.compress_every:
        for cache_slack in args.cache_slack:
            for max_drop in args.max_drop:
                label = f"c{compress_every}_slack{cache_slack}_md{max_drop}"
                output_path = args.output_dir / f"{label}.json"
                result = _run_eval(
                    args,
                    compress_every,
                    cache_slack,
                    max_drop,
                    output_path,
                    baseline_results=baseline_path,
                )
                if result is None:
                    continue
                rows.append(_flatten(result))

    (args.output_dir / "summary.json").write_text(json.dumps(rows, indent=2))
    _write_csv(rows, args.output_dir / "summary.csv")


if __name__ == "__main__":
    main()
