#!/usr/bin/env python3
"""
Run fixed baselines multiple times and average metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# Ensure repo root is on sys.path when invoked as a script (e.g., `python experiments/run_fixed_baseline.py`).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.paper.env_info import collect_env_info, env_compatible


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed baselines and average results")
    parser.add_argument("--model-name", type=str, default="EleutherAI/pythia-2.8b")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--n-sink", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=2044)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("results/baselines"))
    parser.add_argument("--no-skip-existing", action="store_true")
    return parser.parse_args()


def _run_baseline(
    dataset: str,
    dataset_config: str,
    max_eval_tokens: int,
    max_samples: int,
    args: argparse.Namespace,
    run_idx: int,
    output_path: Path,
) -> dict[str, Any] | None:
    if output_path.exists() and not args.no_skip_existing:
        return json.loads(output_path.read_text())

    max_length = args.window_size + args.n_sink
    stride = max(1, args.window_size // 2)

    cmd = [
        sys.executable,
        "experiments/eval_streaming_llm.py",
        "--model-name", args.model_name,
        "--dtype", args.dtype,
        "--dataset-name", dataset,
        "--dataset-config", dataset_config,
        "--max-eval-tokens", str(max_eval_tokens),
        "--max-samples", str(max_samples),
        "--mode", "baseline",
        "--n-sink", str(args.n_sink),
        "--window-size", str(args.window_size),
        "--overlap", "0",
        "--refresh-budget", "0",
        "--refresh-policy", "none",
        "--compress-every", "1",
        "--max-length", str(max_length),
        "--stride", str(stride),
        "--output", str(output_path),
    ]
    # Pin the presampled file selection to keep results comparable across machines:
    # - PG19: prefer the long_context_{max_eval_tokens}.json sample if available.
    # - WikiText: prefer the long_context_{max_eval_tokens}.json sample if available.
    env = os.environ.copy()
    if dataset.lower().startswith("pg19") and "PG19_SAMPLE_FILE" not in env and "PG19_SAMPLE_LENGTH" not in env:
        env["PG19_SAMPLE_LENGTH"] = str(max_eval_tokens)
    if dataset.lower().startswith("wikitext") and "WIKITEXT_SAMPLE_FILE" not in env and "WIKITEXT_SAMPLE_LENGTH" not in env:
        env["WIKITEXT_SAMPLE_LENGTH"] = str(max_eval_tokens)

    result = subprocess.run(cmd, check=False, env=env)
    if result.returncode != 0 or not output_path.exists():
        print(f"Baseline run failed: {dataset} #{run_idx}")
        return None
    return json.loads(output_path.read_text())


def _average_metrics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "perplexity",
        "runtime_sec",
        "prefill_sec",
        "first_token_latency_sec",
        "decode_tokens",
        "decode_time_sec",
        "tpot_ms",
        "peak_memory_mb",
    ]
    avg = {}
    for key in keys:
        vals = [r["baseline"].get(key) for r in runs if r.get("baseline") is not None]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        if key == "decode_tokens":
            avg[key] = int(round(sum(vals) / len(vals)))
        else:
            avg[key] = float(sum(vals) / len(vals))
    return avg


def _write_avg(
    dataset: str,
    dataset_config: str,
    runs: list[dict[str, Any]],
    output_path: Path,
    args: argparse.Namespace,
) -> None:
    env_now = collect_env_info(repo_root=str(Path(__file__).resolve().parents[1])).to_dict()
    prev_env = None
    if output_path.exists():
        try:
            prev_env = json.loads(output_path.read_text()).get("env")
        except Exception:
            prev_env = None

    avg_metrics = _average_metrics(runs)
    template = runs[0]
    results = {
        "model": template.get("model"),
        "dataset": f"{dataset}:{dataset_config}",
        "split": template.get("split"),
        "max_length": template.get("max_length"),
        "stride": template.get("stride"),
        "max_samples": template.get("max_samples"),
        "max_eval_tokens": template.get("max_eval_tokens"),
        "total_tokens": template.get("total_tokens"),
        "streaming_llm": template.get("streaming_llm"),
        "baseline": avg_metrics,
        "baseline_source": f"averaged:{len(runs)}",
        "device": template.get("device"),
        "dtype": template.get("dtype"),
        "env": env_now,
        "runs": [r.get("_run_path") for r in runs],
    }
    output_path.write_text(json.dumps(results, indent=2))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        ("wikitext", "wikitext-103-v1", 4096, 64),
        ("pg19", "pg19", 20000, 1),
    ]

    for dataset, dataset_config, max_eval_tokens, max_samples in datasets:
        run_dir = args.output_dir / dataset
        run_dir.mkdir(parents=True, exist_ok=True)
        runs = []
        for run_idx in range(args.runs):
            output_path = run_dir / f"run_{run_idx+1}.json"
            data = _run_baseline(
                dataset,
                dataset_config,
                max_eval_tokens,
                max_samples,
                args,
                run_idx + 1,
                output_path,
            )
            if data is None:
                return
            data["_run_path"] = str(output_path)
            runs.append(data)

        avg_path = args.output_dir / f"{dataset}_baseline_avg.json"
        _write_avg(dataset, dataset_config, runs, avg_path, args)
        print(f"Saved averaged baseline: {avg_path}")


if __name__ == "__main__":
    main()
