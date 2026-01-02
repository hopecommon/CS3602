#!/usr/bin/env python3
"""
Run `experiments/eval_streaming_llm.py` repeatedly (warmup + runs) and aggregate metrics.

This is used to make paper experiments more reliable (mean/std) without duplicating
logic in bash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

# Ensure repo root is on sys.path when invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, var ** 0.5


def _aggregate_block(blocks: list[dict[str, Any]]) -> dict[str, Any]:
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
    out: dict[str, Any] = {}
    for k in keys:
        vals = [b.get(k) for b in blocks if isinstance(b.get(k), (int, float))]
        vals_f = [float(v) for v in vals]
        if not vals_f:
            continue
        mean, std = _mean_std(vals_f)
        if k == "decode_tokens":
            out[k] = int(round(mean))
            out[f"{k}_std"] = float(std)
        else:
            out[k] = float(mean)
            out[f"{k}_std"] = float(std)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3, help="Number of timed runs to aggregate.")
    ap.add_argument("--warmup", type=int, default=1, help="Warmup runs (discarded).")
    ap.add_argument("--out", type=Path, required=True, help="Aggregated JSON output path.")
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Directory to store per-run JSONs (default: out.parent/out.stem_runs/).",
    )
    ap.add_argument("--skip-existing", action="store_true", help="Skip if out exists.")
    ap.add_argument("eval_args", nargs=argparse.REMAINDER, help="Arguments for eval_streaming_llm.py after --")
    return ap.parse_args()

def _config_hash(eval_args: list[str], warmup: int, runs: int) -> str:
    payload = "\n".join(
        [
            f"warmup={warmup}",
            f"runs={runs}",
            "args=" + " ".join(eval_args),
        ]
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]

def main() -> int:
    args = parse_args()
    if args.eval_args and args.eval_args[0] == "--":
        eval_args = args.eval_args[1:]
    else:
        eval_args = args.eval_args

    if not eval_args:
        raise SystemExit("Missing eval args; pass them after '--'.")

    cfg_hash = _config_hash(eval_args, warmup=args.warmup, runs=args.runs)
    if args.skip_existing and args.out.exists():
        try:
            existing = _load_json(args.out)
            if existing.get("config_hash") == cfg_hash:
                print(f"Skip existing (config match): {args.out}")
                return 0
            print(f"Existing output config mismatch; rerunning: {args.out}")
        except Exception:
            print(f"Existing output unreadable; rerunning: {args.out}")

    out_dir = args.run_dir or (args.out.parent / f"{args.out.stem}_runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    total = args.warmup + args.runs
    run_paths: list[Path] = []
    for i in range(total):
        run_path = out_dir / f"run_{i+1}.json"
        cmd = [sys.executable, "experiments/eval_streaming_llm.py", *eval_args, "--output", str(run_path)]
        print("cmd:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        run_paths.append(run_path)

    # Aggregate (exclude warmup)
    timed_paths = run_paths[args.warmup :]
    timed = [_load_json(p) for p in timed_paths]

    baseline_blocks = [t.get("baseline", {}) for t in timed if isinstance(t.get("baseline"), dict)]
    streaming_blocks = [t.get("streaming", {}) for t in timed if isinstance(t.get("streaming"), dict)]

    agg: dict[str, Any] = {}
    template = timed[0]
    for k in (
        "model",
        "dataset",
        "split",
        "max_length",
        "stride",
        "max_samples",
        "max_eval_tokens",
        "total_tokens",
        "streaming_llm",
        "device",
        "dtype",
        "baseline_source",
        "env",
    ):
        if k in template:
            agg[k] = template[k]

    if baseline_blocks:
        agg["baseline"] = _aggregate_block(baseline_blocks)
    if streaming_blocks:
        agg["streaming"] = _aggregate_block(streaming_blocks)

    # Derived metrics (use aggregated means)
    metrics: dict[str, Any] = {}
    if agg.get("baseline") and agg.get("streaming"):
        b = agg["baseline"]
        s = agg["streaming"]
        if s.get("runtime_sec", 0.0) > 0:
            metrics["speedup"] = float(b.get("runtime_sec", 0.0)) / float(s.get("runtime_sec", 1.0))
        if b.get("perplexity") and s.get("perplexity"):
            metrics["ppl_increase_percent"] = (float(s["perplexity"]) - float(b["perplexity"])) / float(b["perplexity"]) * 100.0
        metrics["tpot_ms"] = s.get("tpot_ms")
        metrics["peak_memory_mb"] = s.get("peak_memory_mb")
    if metrics:
        agg["metrics"] = metrics

    agg["config_hash"] = cfg_hash
    agg["repeats"] = {"warmup": args.warmup, "runs": args.runs, "run_paths": [str(p) for p in timed_paths]}
    args.out.write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(f"Wrote aggregated results: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
