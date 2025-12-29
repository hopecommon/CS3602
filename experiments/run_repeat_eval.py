#!/usr/bin/env python3
"""
Repeat a single eval config multiple times and report mean/std.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repeat eval config")
    parser.add_argument("--name", type=str, default="repeat_eval")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("results/repeat_eval"))
    parser.add_argument("--dataset-name", type=str, default="pg19")
    parser.add_argument("--dataset-config", type=str, default="pg19")
    parser.add_argument("--max-eval-tokens", type=int, default=20000)
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--n-sink", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=2016)
    parser.add_argument("--compress-every", type=int, default=32)
    parser.add_argument("--cache-slack", type=int, default=0)
    parser.add_argument("--max-drop", type=int, default=0)
    parser.add_argument("--refresh-policy", type=str, default="none")
    parser.add_argument("--refresh-budget", type=int, default=0)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--no-skip-existing", action="store_true")
    return parser.parse_args()


def _run_once(args: argparse.Namespace, idx: int, output_path: Path) -> dict:
    if output_path.exists() and not args.no_skip_existing:
        return json.loads(output_path.read_text())

    max_length = args.window_size + args.n_sink
    stride = max(1, args.window_size // 2)
    cmd = [
        "python", "experiments/eval_streaming_llm.py",
        "--dtype", args.dtype,
        "--dataset-name", args.dataset_name,
        "--dataset-config", args.dataset_config,
        "--max-eval-tokens", str(args.max_eval_tokens),
        "--max-samples", str(args.max_samples),
        "--mode", "streaming",
        "--baseline-results", str(args.baseline_results),
        "--n-sink", str(args.n_sink),
        "--window-size", str(args.window_size),
        "--compress-every", str(args.compress_every),
        "--cache-slack", str(args.cache_slack),
        "--max-drop", str(args.max_drop),
        "--refresh-policy", args.refresh_policy,
        "--refresh-budget", str(args.refresh_budget),
        "--max-length", str(max_length),
        "--stride", str(stride),
        "--output", str(output_path),
    ]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0 or not output_path.exists():
        raise RuntimeError(f"Run {idx} failed")
    return json.loads(output_path.read_text())


def _extract_metrics(result: dict) -> tuple[float, float, float]:
    streaming = result.get("streaming", {})
    metrics = result.get("metrics", {})
    tpot_ms = streaming.get("tpot_ms")
    ppl = streaming.get("perplexity")
    speedup = metrics.get("speedup")
    return tpot_ms, ppl, speedup


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    tpot = []
    ppl = []
    speedup = []
    for i in range(args.repeats):
        output_path = args.output_dir / f"{args.name}_run{i+1}.json"
        result = _run_once(args, i + 1, output_path)
        runs.append(result)
        tpot_ms, ppl_val, speedup_val = _extract_metrics(result)
        if tpot_ms is not None:
            tpot.append(tpot_ms)
        if ppl_val is not None:
            ppl.append(ppl_val)
        if speedup_val is not None:
            speedup.append(speedup_val)

    def _stats(values: list[float]) -> dict:
        if not values:
            return {"mean": None, "stdev": None}
        if len(values) == 1:
            return {"mean": values[0], "stdev": 0.0}
        return {"mean": statistics.mean(values), "stdev": statistics.stdev(values)}

    summary = {
        "config": {
            "dataset": f"{args.dataset_name}:{args.dataset_config}",
            "n_sink": args.n_sink,
            "window_size": args.window_size,
            "compress_every": args.compress_every,
            "cache_slack": args.cache_slack,
            "max_drop": args.max_drop,
            "refresh_policy": args.refresh_policy,
            "refresh_budget": args.refresh_budget,
        },
        "tpot_ms": _stats(tpot),
        "ppl": _stats(ppl),
        "speedup": _stats(speedup),
        "runs": [r.get("output_path", "") for r in runs],
    }
    summary_path = args.output_dir / f"{args.name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
