#!/usr/bin/env python3
"""
Phase B2 minimal matrix: strict vs lazy vs static cache.
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
    parser = argparse.ArgumentParser(description="Run Phase B2 matrix")
    parser.add_argument("--dataset-name", type=str, default="pg19")
    parser.add_argument("--dataset-config", type=str, default="pg19")
    parser.add_argument("--max-eval-tokens", type=int, default=20000)
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--window-size", type=int, default=2044)
    parser.add_argument("--n-sink", type=int, default=4)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--refresh-budget", type=int, default=0)
    parser.add_argument("--baseline-dynamic", type=Path, default=None)
    parser.add_argument("--baseline-static", type=Path, default=None)
    parser.add_argument("--fixed-baseline-dir", type=Path, default=Path("results/baselines"))
    parser.add_argument("--skip-static", action="store_true")
    parser.add_argument("--skip-no-prune", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase_b2"))
    parser.add_argument("--no-skip-existing", action="store_true")
    return parser.parse_args()


def _run_baseline(
    cache_impl: str | None,
    args: argparse.Namespace,
    output_path: Path,
    override_path: Path | None = None,
) -> Path:
    if override_path is not None:
        if not override_path.exists():
            raise FileNotFoundError(f"Baseline override not found: {override_path}")
        return override_path
    if output_path.exists() and not args.no_skip_existing:
        return output_path

    max_length = args.window_size + args.n_sink + args.overlap + args.refresh_budget
    stride = max(1, args.window_size // 2)

    cmd = [
        "python", "experiments/eval_streaming_llm.py",
        "--dtype", args.dtype,
        "--dataset-name", args.dataset_name,
        "--dataset-config", args.dataset_config,
        "--max-eval-tokens", str(args.max_eval_tokens),
        "--max-samples", str(args.max_samples),
        "--mode", "baseline",
        "--n-sink", str(args.n_sink),
        "--window-size", str(args.window_size),
        "--overlap", str(args.overlap),
        "--refresh-budget", str(args.refresh_budget),
        "--refresh-policy", "none",
        "--compress-every", "1",
        "--max-length", str(max_length),
        "--stride", str(stride),
        "--output", str(output_path),
    ]
    if cache_impl:
        cmd += ["--cache-implementation", cache_impl]

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0 or not output_path.exists():
        raise RuntimeError(f"Baseline failed ({cache_impl or 'dynamic'})")
    return output_path


def _run_eval(label: str, compress_every: int, cache_impl: str | None, args: argparse.Namespace, output_path: Path, baseline_results: Path) -> dict[str, Any] | None:
    if output_path.exists() and not args.no_skip_existing:
        return json.loads(output_path.read_text())

    max_length = args.window_size + args.n_sink + args.overlap + args.refresh_budget
    stride = max(1, args.window_size // 2)

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
        "--window-size", str(args.window_size),
        "--overlap", str(args.overlap),
        "--refresh-budget", str(args.refresh_budget),
        "--refresh-policy", "none",
        "--compress-every", str(compress_every),
        "--max-length", str(max_length),
        "--stride", str(stride),
        "--output", str(output_path),
    ]
    if cache_impl:
        cmd += ["--cache-implementation", cache_impl]

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0 or not output_path.exists():
        print(f"错误: {label} 运行失败")
        return None
    return json.loads(output_path.read_text())


def _flatten(label: str, data: dict[str, Any], cache_impl: str | None, compress_every: int) -> dict[str, Any]:
    baseline = data.get("baseline", {})
    streaming = data.get("streaming", {})
    metrics = data.get("metrics", {})
    return {
        "label": label,
        "cache_implementation": cache_impl,
        "compress_every": compress_every,
        "baseline_ppl": baseline.get("perplexity"),
        "streaming_ppl": streaming.get("perplexity"),
        "ppl_increase_percent": metrics.get("ppl_increase_percent"),
        "baseline_tpot_ms": baseline.get("tpot_ms"),
        "streaming_tpot_ms": streaming.get("tpot_ms"),
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

    fixed_baseline = load_fixed_baseline(args.dataset_name, args.fixed_baseline_dir)
    fixed_path = None
    if fixed_baseline:
        fixed_path = args.fixed_baseline_dir / f"{args.dataset_name.lower()}_baseline_avg.json"

    matrix = [
        ("lazy_c32_dynamic", 32, None),
        ("lazy_c16_dynamic", 16, None),
        ("no_prune_dynamic", 0, None),
        ("lazy_c32_static", 32, "static"),
        ("lazy_c16_static", 16, "static"),
        ("strict_dynamic", 1, None),
        ("strict_static", 1, "static"),
    ]
    if args.skip_no_prune:
        matrix = [item for item in matrix if item[1] != 0]

    rows: list[dict[str, Any]] = []
    baseline_dynamic = _run_baseline(
        None,
        args,
        args.output_dir / "baseline_dynamic.json",
        override_path=args.baseline_dynamic or fixed_path,
    )
    baseline_static = None
    if not args.skip_static:
        baseline_static = _run_baseline(
            "static",
            args,
            args.output_dir / "baseline_static.json",
            override_path=args.baseline_static,
        )
    for label, compress_every, cache_impl in matrix:
        if args.skip_static and cache_impl == "static":
            continue
        output_path = args.output_dir / f"{label}.json"
        baseline_results = baseline_static if cache_impl == "static" else baseline_dynamic
        data = _run_eval(label, compress_every, cache_impl, args, output_path, baseline_results)
        if data is None:
            continue
        rows.append(_flatten(label, data, cache_impl, compress_every))

    (args.output_dir / "summary.json").write_text(json.dumps(rows, indent=2))
    _write_csv(rows, args.output_dir / "summary.csv")


if __name__ == "__main__":
    main()
