#!/usr/bin/env python3
"""
Phase B1 sweep: overlap + refresh (uniform) under fixed window/sink/R.
Runs eval_streaming_llm (PPL/TPOT) and diagnose_prune_nll (spikes/latency P50/P90),
then writes a combined summary CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase B1 overlap/refresh sweep")
    parser.add_argument("--window-size", type=int, default=2048)
    parser.add_argument("--n-sink", type=int, default=4)
    parser.add_argument("--compress-every", type=int, default=32)
    parser.add_argument("--overlaps", type=int, nargs="+", default=[0, 128, 256])
    parser.add_argument("--refresh-budgets", type=int, nargs="+", default=[0, 8, 16])
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help="Use full Cartesian grid of overlaps x refresh budgets"
    )
    parser.add_argument("--refresh-policy", type=str, default="uniform", choices=["none", "uniform"])
    parser.add_argument("--datasets", type=str, nargs="+", default=["wikitext", "pg19"])
    parser.add_argument("--wikitext-window", type=int, default=None)
    parser.add_argument("--pg19-window", type=int, default=None)
    parser.add_argument(
        "--auto-window-cap",
        type=int,
        default=None,
        help="If set, clamp window so window + sink + overlap + refresh <= cap"
    )
    parser.add_argument("--wikitext-config", type=str, default="wikitext-103-v1")
    parser.add_argument("--wikitext-max-eval-tokens", type=int, default=4096)
    parser.add_argument("--pg19-max-eval-tokens", type=int, default=20000)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase_b1"))
    parser.add_argument("--no-skip-existing", action="store_true")
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


def _run_eval(
    dataset: str,
    window_size: int,
    overlap: int,
    refresh_budget: int,
    args: argparse.Namespace,
    output_path: Path,
    baseline_results: Path,
) -> dict[str, Any] | None:
    if output_path.exists() and not args.no_skip_existing:
        return json.loads(output_path.read_text())

    max_length = window_size + args.n_sink + overlap + refresh_budget
    stride = max(1, window_size // 2)

    cmd = [
        "python", "experiments/eval_streaming_llm.py",
        "--dtype", args.dtype,
        "--n-sink", str(args.n_sink),
        "--window-size", str(window_size),
        "--overlap", str(overlap),
        "--refresh-budget", str(refresh_budget),
        "--refresh-policy", args.refresh_policy if refresh_budget > 0 else "none",
        "--compress-every", str(args.compress_every),
        "--max-length", str(max_length),
        "--stride", str(stride),
        "--mode", "streaming",
        "--baseline-results", str(baseline_results),
        "--output", str(output_path),
    ]
    cmd += _dataset_args(dataset, args)

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0 or not output_path.exists():
        print(f"错误: eval 失败 {dataset} o{overlap} r{refresh_budget}")
        return None
    return json.loads(output_path.read_text())


def _run_diagnose(
    dataset: str,
    window_size: int,
    overlap: int,
    refresh_budget: int,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any] | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{dataset}_w{window_size}_s{args.n_sink}_o{overlap}_r{refresh_budget}_c{args.compress_every}"
    summary_path = output_dir / f"{prefix}_summary.json"
    if summary_path.exists() and not args.no_skip_existing:
        return json.loads(summary_path.read_text())

    cmd = [
        "python", "experiments/diagnose_prune_nll.py",
        "--dtype", args.dtype,
        "--dataset-name", dataset,
        "--dataset-config", args.wikitext_config if dataset == "wikitext" else "pg19",
        "--max-eval-tokens", str(args.wikitext_max_eval_tokens if dataset == "wikitext" else args.pg19_max_eval_tokens),
        "--max-samples", str(args.max_samples if dataset == "wikitext" else 1),
        "--window-size", str(window_size),
        "--n-sink", str(args.n_sink),
        "--overlap", str(overlap),
        "--refresh-budget", str(refresh_budget),
        "--refresh-policy", args.refresh_policy if refresh_budget > 0 else "none",
        "--compress-every", str(args.compress_every),
        "--output-dir", str(output_dir),
    ]

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0 or not summary_path.exists():
        print(f"错误: diagnose 失败 {dataset} o{overlap} r{refresh_budget}")
        return None
    return json.loads(summary_path.read_text())


def _flatten(eval_data: dict[str, Any], diag: dict[str, Any], window_size: int, auto_cap: int | None) -> dict[str, Any]:
    baseline = eval_data.get("baseline", {})
    streaming = eval_data.get("streaming", {})
    metrics = eval_data.get("metrics", {})
    streaming_cfg = eval_data.get("streaming_llm", {})

    row = {
        "dataset": eval_data.get("dataset"),
        "window_size": window_size,
        "n_sink": streaming_cfg.get("n_sink"),
        "overlap": streaming_cfg.get("overlap"),
        "refresh_budget": streaming_cfg.get("refresh_budget"),
        "refresh_policy": streaming_cfg.get("refresh_policy"),
        "compress_every": streaming_cfg.get("compress_every"),
        "auto_window_cap": auto_cap,
        "baseline_ppl": baseline.get("perplexity"),
        "streaming_ppl": streaming.get("perplexity"),
        "ppl_increase_percent": metrics.get("ppl_increase_percent"),
        "baseline_tpot_ms": baseline.get("tpot_ms"),
        "streaming_tpot_ms": streaming.get("tpot_ms"),
        "streaming_tpot_p50_ms": diag.get("latency_ms_p50"),
        "streaming_tpot_p90_ms": diag.get("latency_ms_p90"),
        "spike_peak_median": diag.get("spike_peak_median"),
        "spike_peak_p90": diag.get("spike_peak_p90"),
        "spike_area_median": diag.get("spike_area_median"),
        "spike_area_p90": diag.get("spike_area_p90"),
        "spike_width_median": diag.get("spike_width_median"),
        "spike_width_p90": diag.get("spike_width_p90"),
        "prune_events": diag.get("prune_events"),
        "drop_sizes": diag.get("drop_sizes"),
        "streaming_peak_memory_mb": streaming.get("peak_memory_mb"),
    }
    return row


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

    eval_dir = args.output_dir / "eval"
    diag_dir = args.output_dir / "diagnose"
    eval_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    if args.full_grid:
        pairs = [(o, r) for o in args.overlaps for r in args.refresh_budgets]
    else:
        pairs = [
            (0, 0),
            (128, 0),
            (256, 0),
            (0, 8),
            (0, 16),
            (128, 8),
            (128, 16),
            (256, 16),
        ]

    for dataset in args.datasets:
        base_window = args.window_size
        if dataset == "wikitext" and args.wikitext_window is not None:
            base_window = args.wikitext_window
        if dataset == "pg19" and args.pg19_window is not None:
            base_window = args.pg19_window
        baseline_path = eval_dir / f"{dataset}_baseline.json"
        if args.no_skip_existing or not baseline_path.exists():
            max_length = base_window + args.n_sink
            stride = max(1, base_window // 2)
            baseline_cmd = [
                "python", "experiments/eval_streaming_llm.py",
                "--dtype", args.dtype,
                "--n-sink", str(args.n_sink),
                "--window-size", str(base_window),
                "--overlap", "0",
                "--refresh-budget", "0",
                "--refresh-policy", "none",
                "--compress-every", str(args.compress_every),
                "--max-length", str(max_length),
                "--stride", str(stride),
                "--mode", "baseline",
                "--output", str(baseline_path),
            ]
            baseline_cmd += _dataset_args(dataset, args)
            result = subprocess.run(baseline_cmd, check=False)
            if result.returncode != 0 or not baseline_path.exists():
                raise RuntimeError(f"Baseline failed for {dataset}")
        for overlap, refresh_budget in pairs:
            window_size = base_window
            if args.auto_window_cap is not None:
                cap = args.auto_window_cap
                window_size = min(
                    window_size,
                    cap - args.n_sink - overlap - refresh_budget
                )
                if window_size < 1:
                    print(f"跳过: cap 太小 (dataset={dataset}, o={overlap}, r={refresh_budget})")
                    continue
            eval_path = eval_dir / (
                f"{dataset}_w{window_size}_s{args.n_sink}"
                f"_o{overlap}_r{refresh_budget}_c{args.compress_every}.json"
            )
            eval_data = _run_eval(
                dataset,
                window_size,
                overlap,
                refresh_budget,
                args,
                eval_path,
                baseline_results=baseline_path,
            )
            diag_data = _run_diagnose(dataset, window_size, overlap, refresh_budget, args, diag_dir)
            if eval_data is None or diag_data is None:
                continue
            rows.append(_flatten(eval_data, diag_data, window_size, args.auto_window_cap))

    summary = {
        "window_size": args.window_size,
        "wikitext_window": args.wikitext_window,
        "pg19_window": args.pg19_window,
        "auto_window_cap": args.auto_window_cap,
        "n_sink": args.n_sink,
        "compress_every": args.compress_every,
        "overlaps": args.overlaps,
        "refresh_budgets": args.refresh_budgets,
        "pairs": pairs,
        "refresh_policy": args.refresh_policy,
        "datasets": args.datasets,
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_csv(rows, args.output_dir / "summary.csv")


if __name__ == "__main__":
    main()
