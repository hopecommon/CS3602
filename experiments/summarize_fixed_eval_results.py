#!/usr/bin/env python3
"""
Summarize results/fixed_eval/*.json into a Markdown table.

Decode-loop methods (comparable): baseline / ours / kvpress / mit
Legacy KVPress results (non-comparable) may exist from older runs.

MIT official benchmark results (throughput/VRAM, non-PPL) are summarized separately.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class DecodeLoopRow:
    dataset: str
    method: str
    ppl: float
    runtime_s: float
    prefill_s: float
    first_token_s: float


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _decode_loop_row(path: Path) -> Optional[DecodeLoopRow]:
    data = _load_json(path)
    if "total_time" not in data:
        return None
    return DecodeLoopRow(
        dataset=path.stem.rsplit("_", 1)[0],
        method=data.get("mode", data.get("method", path.stem.rsplit("_", 1)[-1])),
        ppl=float(data["perplexity"]),
        runtime_s=float(data["total_time"]),
        prefill_s=float(data.get("prefill_time", 0.0)),
        first_token_s=float(data.get("first_token_latency_sec", 0.0)),
    )


def _kvpress_row(path: Path) -> Dict[str, Any]:
    data = _load_json(path)
    dataset = path.stem.rsplit("_", 1)[0]
    baseline = data["baseline"]
    kvpress = data["kvpress"]
    metrics = data.get("metrics", {})
    return {
        "dataset": dataset,
        "baseline_ppl": float(baseline["perplexity"]),
        "kvpress_ppl": float(kvpress["perplexity"]),
        "baseline_runtime_s": float(baseline["runtime_sec"]),
        "kvpress_runtime_s": float(kvpress["runtime_sec"]),
        "speedup": float(metrics.get("speedup", baseline["runtime_sec"] / kvpress["runtime_sec"])),
        "baseline_first_token_s": float(baseline.get("first_token_latency_sec", 0.0)),
        "kvpress_first_token_s": float(kvpress.get("first_token_latency_sec", 0.0)),
        "kvpress_peak_mem_mb": float(kvpress.get("peak_memory_mb", 0.0)),
    }


def _fmt(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}"

def _parse_mit_official(results_dir: Path) -> list[dict[str, Any]]:
    """
    Parse `experiments/run_mit_official_benchmark.py` JSON outputs under results/mit_official/.
    This is NOT perplexity; it reports decode tokens/s and peak VRAM.
    """
    rows: list[dict[str, Any]] = []
    if not results_dir.exists():
        return rows

    def pick_peak(cuda: Any) -> Tuple[float, float]:
        if not isinstance(cuda, dict) or not cuda:
            return 0.0, 0.0
        # Prefer cuda:0; otherwise first entry.
        entry = cuda.get("0") or next(iter(cuda.values()))
        if not isinstance(entry, dict):
            return 0.0, 0.0
        for prefix in ("decode_peak_", "peak_", "prefill_peak_"):
            a = entry.get(f"{prefix}allocated_mb")
            r = entry.get(f"{prefix}reserved_mb")
            if isinstance(a, (int, float)) and isinstance(r, (int, float)):
                return float(a), float(r)
        return 0.0, 0.0

    for path in sorted(results_dir.glob("*.json")):
        try:
            data = _load_json(path)
        except Exception:
            continue
        runs = data.get("runs", {})
        if not isinstance(runs, dict) or not runs:
            continue

        for mode, run in runs.items():
            if not isinstance(run, dict):
                continue
            cuda = run.get("cuda")
            peak_alloc, peak_reserved = pick_peak(cuda)
            rows.append(
                {
                    "file": path.name,
                    "mode": str(mode),
                    "prefix_tokens": int(run.get("prefix_tokens") or data.get("params", {}).get("prefix_tokens") or 0),
                    "gen_tokens": int(data.get("params", {}).get("gen_tokens") or 0),
                    "prefill_seconds": float(run.get("prefill_seconds") or 0.0),
                    "decode_seconds": float(run.get("decode_seconds") or 0.0),
                    "decode_tokens_per_second": float(run.get("decode_tokens_per_second") or 0.0),
                    "peak_allocated_mb": peak_alloc,
                    "peak_reserved_mb": peak_reserved,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize fixed_eval JSON results.")
    parser.add_argument("--results-dir", type=Path, default=Path("results/fixed_eval"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--mit-official-dir",
        type=Path,
        default=Path("results/mit_official"),
        help="Directory containing MIT official benchmark JSONs (non-PPL).",
    )
    args = parser.parse_args()

    results_dir: Path = args.results_dir
    output_md: Path = args.output or (results_dir / "summary.md")

    decode_rows: list[DecodeLoopRow] = []
    kvpress_rows: list[dict[str, Any]] = []

    for path in sorted(results_dir.glob("*.json")):
        row = _decode_loop_row(path)
        if row is not None:
            decode_rows.append(row)
            continue
        data = _load_json(path)
        if "baseline" in data and "kvpress" in data:
            kvpress_rows.append(_kvpress_row(path))

    by_dataset: dict[str, dict[str, DecodeLoopRow]] = {}
    for row in decode_rows:
        by_dataset.setdefault(row.dataset, {})[row.method] = row

    lines: list[str] = []
    lines.append("# Fixed Eval Summary\n")
    lines.append("## Decode-Loop (comparable)\n")
    lines.append("| Dataset | Method | PPL | Runtime (s) | Speedup vs baseline | First-token (s) |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for dataset in sorted(by_dataset.keys()):
        baseline = by_dataset[dataset].get("baseline")
        if baseline is None:
            continue
        # Prefer the minimal comparable set (baseline/ours). Include extras if present.
        method_order = ["baseline", "ours"]
        if "kvpress" in by_dataset[dataset]:
            method_order.append("kvpress")
        if "mit" in by_dataset[dataset]:
            method_order.append("mit")
        for method in method_order:
            row = by_dataset[dataset].get(method)
            if row is None:
                continue
            speedup = baseline.runtime_s / row.runtime_s if row.runtime_s > 0 else 0.0
            lines.append(
                f"| {dataset} | {method} | {_fmt(row.ppl, 4)} | {_fmt(row.runtime_s, 2)} | {_fmt(speedup, 2)}× | {_fmt(row.first_token_s, 3)} |"
            )
    lines.append("")

    if kvpress_rows:
        lines.append("## Legacy KVPress (non-comparable)\n")
        lines.append("| Dataset | KVPress PPL | KVPress Runtime (s) | KVPress Speedup (vs its baseline) | KVPress First-token (s) |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in sorted(kvpress_rows, key=lambda r: r["dataset"]):
            lines.append(
                f"| {row['dataset']} | {_fmt(row['kvpress_ppl'], 4)} | {_fmt(row['kvpress_runtime_s'], 3)} | {_fmt(row['speedup'], 2)}× | {_fmt(row['kvpress_first_token_s'], 3)} |"
            )
        lines.append("")

    mit_rows = _parse_mit_official(args.mit_official_dir)
    if mit_rows:
        lines.append("## MIT Official Benchmark (non-PPL)\n")
        lines.append("Measures decode throughput (tokens/s) and peak VRAM; not directly comparable to the decode-loop PPL table.\n")
        lines.append("| File | Mode | Prefix | Gen | Decode tok/s | Decode (s) | Prefill (s) | Peak alloc (MB) | Peak reserved (MB) |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in mit_rows:
            lines.append(
                f"| {row['file']} | {row['mode']} | {row['prefix_tokens']} | {row['gen_tokens']} |"
                f" {_fmt(row['decode_tokens_per_second'], 2)} | {_fmt(row['decode_seconds'], 2)} | {_fmt(row['prefill_seconds'], 2)} |"
                f" {_fmt(row['peak_allocated_mb'], 1)} | {_fmt(row['peak_reserved_mb'], 1)} |"
            )
        lines.append("")

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output_md}")


if __name__ == "__main__":
    main()
