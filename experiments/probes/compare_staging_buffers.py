#!/usr/bin/env python3
"""
Compare StreamingLLM staging-buffer on/off for peak memory and TPOT.

This is a targeted probe for the question:
  "Why does our implementation show higher peak memory than another implementation?"

We keep everything identical except `--(no-)staging-buffers`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunResult:
    name: str
    tpot_ms: float
    peak_memory_mb: float
    perplexity: float
    path: str


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_streaming_metrics(path: Path) -> RunResult:
    data = _load_json(path)
    streaming = data.get("streaming") or {}
    return RunResult(
        name=data.get("streaming_llm", {}).get("cache_type", "streaming"),
        tpot_ms=float(streaming.get("tpot_ms", 0.0)),
        peak_memory_mb=float(streaming.get("peak_memory_mb", 0.0)),
        perplexity=float(streaming.get("perplexity", 0.0)),
        path=str(path),
    )


def _run_eval(eval_args: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "experiments/eval_streaming_llm.py", *eval_args, "--output", str(out_path)]
    print("cmd:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", type=str, default="EleutherAI/pythia-2.8b")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--dataset-name", type=str, default="pg19")
    ap.add_argument("--dataset-config", type=str, default="pg19")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--max-samples", type=int, default=1)
    ap.add_argument("--max-eval-tokens", type=int, default=20000)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=1022)
    ap.add_argument("--n-sink", type=int, default=32)
    ap.add_argument("--window-size", type=int, default=2016)
    ap.add_argument("--compress-every", type=int, default=64)
    ap.add_argument("--cache-slack", type=int, default=0)
    ap.add_argument("--max-drop", type=int, default=0)
    ap.add_argument("--fp32-loss", action="store_true", default=True)
    ap.add_argument("--no-fp32-loss", action="store_false", dest="fp32_loss")
    ap.add_argument("--out-dir", type=Path, default=Path("results/probes/staging_buffers"))
    ap.add_argument("--tag", type=str, default="run")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    base = [
        "--model-name",
        args.model_name,
        "--dtype",
        args.dtype,
        "--dataset-name",
        args.dataset_name,
        "--dataset-config",
        args.dataset_config,
        "--split",
        args.split,
        "--max-samples",
        str(args.max_samples),
        "--max-eval-tokens",
        str(args.max_eval_tokens),
        "--max-length",
        str(args.max_length),
        "--stride",
        str(args.stride),
        "--n-sink",
        str(args.n_sink),
        "--window-size",
        str(args.window_size),
        "--compress-every",
        str(args.compress_every),
        "--cache-slack",
        str(args.cache_slack),
        "--max-drop",
        str(args.max_drop),
        "--overlap",
        "0",
        "--refresh-budget",
        "0",
        "--refresh-policy",
        "none",
        "--streaming-mode",
        "ours",
        "--cache-backend",
        "ours",
        "--mode",
        "both",
    ]
    if args.fp32_loss:
        base.append("--fp32-loss")
    else:
        base.append("--no-fp32-loss")

    out_on = args.out_dir / f"{args.dataset_name}_{args.tag}_staging_on.json"
    out_off = args.out_dir / f"{args.dataset_name}_{args.tag}_staging_off.json"
    _run_eval([*base, "--staging-buffers"], out_on)
    _run_eval([*base, "--no-staging-buffers"], out_off)

    on = _extract_streaming_metrics(out_on)
    off = _extract_streaming_metrics(out_off)

    compare = {
        "staging_on": on.__dict__,
        "staging_off": off.__dict__,
        "delta": {
            "tpot_ms": off.tpot_ms - on.tpot_ms,
            "peak_memory_mb": off.peak_memory_mb - on.peak_memory_mb,
            "perplexity": off.perplexity - on.perplexity,
        },
    }
    out_compare = args.out_dir / f"{args.dataset_name}_{args.tag}_compare.json"
    out_compare.write_text(json.dumps(compare, indent=2), encoding="utf-8")
    print(f"Wrote: {out_compare}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

