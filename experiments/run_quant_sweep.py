#!/usr/bin/env python3
"""
Run quantization + compile sweeps and log TPOT/PPL into a CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List
import importlib.util


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep quantization/compile configs and log TPOT/PPL."
    )
    parser.add_argument(
        "--quantizations",
        type=str,
        default="none,int8wo,int8da,int4wo",
        help="Comma-separated quantization configs"
    )
    parser.add_argument(
        "--compile-modes",
        type=str,
        default="off,reduce-overhead",
        help="Comma-separated compile modes (use 'off' to disable)"
    )
    parser.add_argument(
        "--compile-cudagraphs",
        action="store_true",
        help="Allow cudagraphs when compiling"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/quant_sweep"),
        help="Directory to store JSON outputs"
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("results/quant_sweep/summary.csv"),
        help="CSV summary path"
    )
    parser.add_argument(
        "--overwrite-csv",
        action="store_true",
        help="Overwrite existing CSV instead of appending"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip runs when output JSON already exists (default: enabled)"
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Disable skipping existing outputs"
    )
    parser.add_argument(
        "--skip-ppl",
        action="store_true",
        help="Skip perplexity evaluation"
    )
    parser.add_argument(
        "--skip-latency",
        action="store_true",
        help="Skip decoding latency evaluation"
    )

    # Shared model/data args
    parser.add_argument(
        "--model-name",
        type=str,
        default=os.environ.get("MODEL_NAME", "EleutherAI/pythia-2.8b"),
        help="Model name"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code"
    )

    # PPL eval args
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="wikitext",
        help="Dataset name"
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="wikitext-103-v1",
        help="Dataset config"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split"
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Dataset text column"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=64,
        help="Max samples"
    )
    parser.add_argument(
        "--max-eval-tokens",
        type=int,
        default=int(os.environ.get("MAX_EVAL_TOKENS", "4096")),
        help="Max eval tokens"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Max length per window"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1024,
        help="Stride for sliding evaluation"
    )
    parser.add_argument(
        "--streaming-mode",
        type=str,
        choices=["ours", "mit"],
        default="ours",
        help="StreamingLLM implementation"
    )

    # Streaming parameters
    parser.add_argument(
        "--n-sink",
        type=int,
        default=int(os.environ.get("N_SINK", "4")),
        help="Sink tokens"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=int(os.environ.get("WINDOW_SIZE", "2048")),
        help="Window size"
    )

    # Latency eval args
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=512,
        help="Prompt length for latency eval"
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=2000,
        help="Generated tokens for latency eval"
    )
    parser.add_argument(
        "--warmup-tokens",
        type=int,
        default=200,
        help="Warmup tokens"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Latency eval repetitions"
    )

    return parser.parse_args()


def _split_values(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _run_command(args: List[str]) -> None:
    subprocess.run(args, check=True)


def _torchao_available() -> bool:
    try:
        import torchao  # noqa: F401
    except Exception:
        return False
    return True


def _int4_available() -> bool:
    return (
        importlib.util.find_spec("fbgemm_gpu_genai") is not None
        or importlib.util.find_spec("fbgemm_gpu.experimental.genai") is not None
    )


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def _write_csv_row(csv_path: Path, row: Dict[str, object], overwrite: bool) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    write_header = overwrite or not csv_path.exists()
    mode = "w" if overwrite else "a"
    with csv_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    quantizations = _split_values(args.quantizations)
    compile_modes = _split_values(args.compile_modes)

    if not _torchao_available():
        filtered = [q for q in quantizations if q in {"none", ""}]
        if not filtered:
            print("未检测到 torchao, 将仅保留 quantization=none")
            filtered = ["none"]
        else:
            print("未检测到 torchao, 已跳过量化配置, 仅运行 none")
        quantizations = filtered
    else:
        if "int4wo" in quantizations and not _int4_available():
            quantizations = [q for q in quantizations if q != "int4wo"]
            print("未检测到 fbgemm-gpu-genai, 已跳过 int4wo")

    args.results_dir.mkdir(parents=True, exist_ok=True)

    eval_streaming = Path("experiments/eval_streaming_llm.py")
    eval_latency = Path("experiments/eval_decoding_latency.py")

    for quant in quantizations:
        for compile_mode in compile_modes:
            use_compile = compile_mode != "off"
            tag = f"q-{quant}_compile-{compile_mode}"
            print(f"\n{'='*72}")
            print(f"Running sweep config: {tag}")
            print(f"{'='*72}")

            ppl_path = args.results_dir / f"ppl_{tag}.json"
            latency_path = args.results_dir / f"latency_{tag}.json"

            if not args.skip_ppl:
                if args.skip_existing and ppl_path.exists():
                    print(f"跳过 PPL (已存在): {ppl_path}")
                else:
                    cmd = [
                        sys.executable,
                        str(eval_streaming),
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
                        "--text-column",
                        args.text_column,
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
                        "--streaming-mode",
                        args.streaming_mode,
                        "--mode",
                        "both",
                        "--output",
                        str(ppl_path),
                        "--quantization",
                        quant,
                    ]
                    if args.trust_remote_code:
                        cmd.append("--trust-remote-code")
                    if use_compile:
                        cmd.extend(["--compile", "--compile-mode", compile_mode])
                        if args.compile_cudagraphs:
                            cmd.append("--compile-cudagraphs")
                    _run_command(cmd)

            if not args.skip_latency:
                if args.skip_existing and latency_path.exists():
                    print(f"跳过 latency (已存在): {latency_path}")
                else:
                    cmd = [
                        sys.executable,
                        str(eval_latency),
                        "--model-name",
                        args.model_name,
                        "--dtype",
                        args.dtype,
                        "--cache-size",
                        str(args.window_size),
                        "--n-sink",
                        str(args.n_sink),
                        "--prompt-length",
                        str(args.prompt_length),
                        "--num-tokens",
                        str(args.num_tokens),
                        "--warmup-tokens",
                        str(args.warmup_tokens),
                        "--num-runs",
                        str(args.num_runs),
                        "--mode",
                        "both",
                        "--output",
                        str(latency_path),
                        "--quantization",
                        quant,
                    ]
                    if args.trust_remote_code:
                        cmd.append("--trust-remote-code")
                    if use_compile:
                        cmd.extend(["--compile", "--compile-mode", compile_mode])
                        if args.compile_cudagraphs:
                            cmd.append("--compile-cudagraphs")
                    _run_command(cmd)

            row = {
                "tag": tag,
                "quantization": quant,
                "compile_mode": compile_mode,
                "compile_cudagraphs": args.compile_cudagraphs,
                "n_sink": args.n_sink,
                "window_size": args.window_size,
                "streaming_mode": args.streaming_mode,
            }

            if not args.skip_ppl and ppl_path.exists():
                ppl_data = _load_json(ppl_path)
                baseline = ppl_data.get("baseline", {})
                streaming = ppl_data.get("streaming", {})
                metrics = ppl_data.get("metrics", {})
                row.update(
                    {
                        "ppl_baseline": baseline.get("perplexity"),
                        "ppl_streaming": streaming.get("perplexity"),
                        "ppl_increase_percent": metrics.get("ppl_increase_percent"),
                        "runtime_baseline_sec": baseline.get("runtime_sec"),
                        "runtime_streaming_sec": streaming.get("runtime_sec"),
                        "runtime_speedup": metrics.get("speedup"),
                        "peak_mem_baseline_mb": baseline.get("peak_memory_mb"),
                        "peak_mem_streaming_mb": streaming.get("peak_memory_mb"),
                    }
                )

            if not args.skip_latency and latency_path.exists():
                latency_data = _load_json(latency_path)
                baseline_lat = latency_data.get("baseline", {})
                streaming_lat = latency_data.get("streaming_llm", {})
                comparison = latency_data.get("comparison", {})
                row.update(
                    {
                        "latency_baseline_ms": baseline_lat.get("mean_latency_ms"),
                        "latency_streaming_ms": streaming_lat.get("mean_latency_ms"),
                        "latency_speedup": comparison.get("speedup"),
                        "latency_mem_baseline_mb": baseline_lat.get("mean_memory_mb"),
                        "latency_mem_streaming_mb": streaming_lat.get("mean_memory_mb"),
                    }
                )

            _write_csv_row(args.csv_path, row, args.overwrite_csv)
            args.overwrite_csv = False


if __name__ == "__main__":
    main()
