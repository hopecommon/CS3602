#!/usr/bin/env python3
"""
Run MIT official StreamingLLM benchmark (throughput + peak VRAM) from the vendored
`mit-streaming-llm` repo and save results as JSON.

This benchmark is NOT perplexity evaluation; it measures decode tokens/s and peak memory.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import os
from os import environ
from pathlib import Path
from typing import Any, Dict, Optional


def _parse_stdout(stdout: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"raw_stdout": stdout}
    patterns = {
        "mode": r"^mode:\s*(.+)$",
        "model_type": r"^model_type:\s*(.+)$",
        "prefix_tokens": r"^prefix_tokens:\s*(\d+)$",
        "gen_tokens_requested": r"^gen_tokens_requested:\s*(\d+)$",
        "gen_tokens_produced": r"^gen_tokens_produced:\s*(\d+)$",
        "prefill_seconds": r"^prefill_seconds:\s*([0-9.]+)$",
        "decode_seconds": r"^decode_seconds:\s*([0-9.]+)$",
        "decode_tokens_per_second": r"^decode_tokens_per_second:\s*([0-9.]+)$",
        "kv_len_after_prefill": r"^kv_len_after_prefill:\s*(\d+)$",
        "kv_len_after_decode": r"^kv_len_after_decode:\s*(\d+)$",
    }
    int_keys = {
        "prefix_tokens",
        "gen_tokens_requested",
        "gen_tokens_produced",
        "kv_len_after_prefill",
        "kv_len_after_decode",
    }
    float_keys = {
        "prefill_seconds",
        "decode_seconds",
        "decode_tokens_per_second",
    }
    for key, pat in patterns.items():
        m = re.search(pat, stdout, flags=re.MULTILINE)
        if not m:
            continue
        val = m.group(1)
        if key in int_keys:
            out[key] = int(val)
        elif key in float_keys:
            out[key] = float(val)
        else:
            out[key] = val.strip()

    # Parse CUDA peaks (prefer decode_peak, fall back to peak_*).
    cuda = {}
    for m in re.finditer(
        r"^cuda:(\d+)\s+prefill_peak_allocated_mb=([0-9.]+)\s+prefill_peak_reserved_mb=([0-9.]+)$",
        stdout,
        flags=re.MULTILINE,
    ):
        cuda[int(m.group(1))] = {
            "prefill_peak_allocated_mb": float(m.group(2)),
            "prefill_peak_reserved_mb": float(m.group(3)),
        }
    for m in re.finditer(
        r"^cuda:(\d+)\s+decode_peak_allocated_mb=([0-9.]+)\s+decode_peak_reserved_mb=([0-9.]+)$",
        stdout,
        flags=re.MULTILINE,
    ):
        entry = cuda.setdefault(int(m.group(1)), {})
        entry.update(
            {
                "decode_peak_allocated_mb": float(m.group(2)),
                "decode_peak_reserved_mb": float(m.group(3)),
            }
        )
    if not cuda:
        for m in re.finditer(
            r"^cuda:(\d+)\s+peak_allocated_mb=([0-9.]+)\s+peak_reserved_mb=([0-9.]+)$",
            stdout,
            flags=re.MULTILINE,
        ):
            cuda[int(m.group(1))] = {
                "peak_allocated_mb": float(m.group(2)),
                "peak_reserved_mb": float(m.group(3)),
            }
    if cuda:
        out["cuda"] = cuda
    return out


def _run_one(
    *,
    python: str,
    script: Path,
    cwd: Path,
    model_name_or_path: str,
    data_json: Path,
    mode: str,
    prefix_tokens: int,
    gen_tokens: int,
    start_size: int,
    recent_size: int,
    prefill_chunk_size: int,
    recompute_window_tokens: int,
    recompute_keep_start: bool,
) -> Dict[str, Any]:
    cmd = [
        python,
        str(script),
        "--model_name_or_path",
        model_name_or_path,
        "--mode",
        mode,
        "--data_json",
        str(data_json),
        "--prefix_tokens",
        str(prefix_tokens),
        "--gen_tokens",
        str(gen_tokens),
        "--start_size",
        str(start_size),
        "--recent_size",
        str(recent_size),
        "--prefill_chunk_size",
        str(prefill_chunk_size),
        "--recompute_window_tokens",
        str(recompute_window_tokens),
    ]
    if recompute_keep_start:
        cmd.append("--recompute_keep_start")

    env = dict(environ)
    # Ensure `mit-streaming-llm/` is importable as a package root for `streaming_llm.*`
    pythonpath = str(cwd)
    env["PYTHONPATH"] = pythonpath + (
        (os.pathsep + env["PYTHONPATH"]) if env.get("PYTHONPATH") else ""
    )

    proc = subprocess.run(cmd, text=True, capture_output=True, cwd=str(cwd), env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Benchmark failed (mode={mode}).\n"
            f"cmd: {' '.join(cmd)}\n"
            f"cwd: {cwd}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}\n"
        )
    parsed = _parse_stdout(proc.stdout)
    parsed["cmd"] = cmd
    parsed["stderr"] = proc.stderr
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MIT official benchmark and save JSON.")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--data-json", type=Path, required=True)
    parser.add_argument("--prefix-tokens", type=int, default=20000)
    parser.add_argument("--gen-tokens", type=int, default=512)
    parser.add_argument("--start-size", type=int, default=4)
    parser.add_argument("--recent-size", type=int, default=2048)
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument("--recompute-window-tokens", type=int, default=2048)
    parser.add_argument("--recompute-keep-start", action="store_true")
    parser.add_argument(
        "--modes",
        type=str,
        default="streaming,recompute",
        help="Comma-separated modes to run: streaming,recompute,full",
    )
    parser.add_argument("--output", type=Path, default=Path("results/mit_official/benchmark.json"))
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    mit_repo = repo / "mit-streaming-llm"
    script = mit_repo / "examples" / "benchmark_streaming.py"
    if not mit_repo.exists() or not script.exists():
        raise FileNotFoundError(f"MIT benchmark script not found: {script}")

    python_path = Path(args.python)
    if not python_path.is_absolute():
        candidate = (repo / python_path).resolve()
        python_path = candidate if candidate.exists() else python_path.resolve()

    results: Dict[str, Any] = {
        "model_name_or_path": args.model_name_or_path,
        "data_json": str(args.data_json),
        "params": {
            "prefix_tokens": args.prefix_tokens,
            "gen_tokens": args.gen_tokens,
            "start_size": args.start_size,
            "recent_size": args.recent_size,
            "prefill_chunk_size": args.prefill_chunk_size,
            "recompute_window_tokens": args.recompute_window_tokens,
            "recompute_keep_start": bool(args.recompute_keep_start),
        },
        "runs": {},
    }

    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        results["runs"][mode] = _run_one(
            python=str(python_path),
            script=script,
            cwd=mit_repo,
            model_name_or_path=args.model_name_or_path,
            data_json=args.data_json,
            mode=mode,
            prefix_tokens=args.prefix_tokens,
            gen_tokens=args.gen_tokens,
            start_size=args.start_size,
            recent_size=args.recent_size,
            prefill_chunk_size=args.prefill_chunk_size,
            recompute_window_tokens=args.recompute_window_tokens,
            recompute_keep_start=args.recompute_keep_start,
        )

    # Convenience speedup numbers when possible.
    streaming = results["runs"].get("streaming")
    recompute = results["runs"].get("recompute")
    if streaming and recompute:
        s_tps = streaming.get("decode_tokens_per_second")
        r_tps = recompute.get("decode_tokens_per_second")
        if isinstance(s_tps, float) and isinstance(r_tps, float) and r_tps > 0:
            results["speedup_streaming_vs_recompute"] = s_tps / r_tps

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
