#!/usr/bin/env python3
"""
Generate LaTeX tables for the NeurIPS paper from JSON result files.

This script is intentionally conservative:
- If a result file is missing or malformed, it outputs [INSERT ...] placeholders.
- It never fabricates numbers.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


PLACEHOLDER = "[INSERT DATA]"


def _load_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _get(d: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _fmt_num(x: Any, fmt: str) -> str:
    try:
        return format(float(x), fmt)
    except Exception:
        return PLACEHOLDER


@dataclass(frozen=True)
class Row:
    name: str
    tpot_ms: str
    speedup: str
    ppl: str
    peak_mem_mb: str


def _extract_row(name: str, data: Optional[dict[str, Any]], assume_speedup_one: bool = False) -> Row:
    if not data:
        return Row(name, PLACEHOLDER, "1.00$\\times$" if assume_speedup_one else PLACEHOLDER, PLACEHOLDER, PLACEHOLDER)

    baseline = data.get("baseline", {}) if isinstance(data, dict) else {}
    streaming = data.get("streaming", {}) if isinstance(data, dict) else {}
    metrics = data.get("metrics", {}) if isinstance(data, dict) else {}

    # Prefer "streaming" block when present (streaming eval output),
    # otherwise fall back to "baseline" (baseline-only JSON).
    block = streaming if streaming else baseline

    tpot = _fmt_num(block.get("tpot_ms"), ".2f")
    ppl = _fmt_num(block.get("perplexity"), ".3f")
    mem = _fmt_num(block.get("peak_memory_mb"), ".0f")

    if assume_speedup_one:
        speedup = "1.00$\\times$"
    else:
        speedup_val = metrics.get("speedup")
        speedup = _fmt_num(speedup_val, ".2f")
        if speedup != PLACEHOLDER:
            speedup = f"{speedup}$\\times$"

    return Row(name=name, tpot_ms=tpot, speedup=speedup, ppl=ppl, peak_mem_mb=mem)


def _table_main_pg19(rows: list[Row]) -> str:
    lines: list[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Main results on PG19 (long-context, \\emph{fill values via script output}).}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Method & TPOT$\\downarrow$ & Speedup$\\uparrow$ & PPL$\\downarrow$ & Peak Mem (MB)$\\downarrow$ \\\\")
    lines.append("\\midrule")
    for r in rows:
        lines.append(f"{r.name} & {r.tpot_ms} & {r.speedup} & {r.ppl} & {r.peak_mem_mb} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def _table_main_wikitext(rows: list[Row]) -> str:
    lines: list[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Sanity-check results on WikiText-103 (short-context).}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append("Method & TPOT$\\downarrow$ & Speedup$\\uparrow$ & PPL$\\downarrow$ \\\\")
    lines.append("\\midrule")
    for r in rows:
        lines.append(f"{r.name} & {r.tpot_ms} & {r.speedup} & {r.ppl} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results/paper_experiments"))
    ap.add_argument("--baseline-dir", type=Path, default=Path("results/baselines"))
    ap.add_argument("--out", type=Path, default=Path("NeurIPS/generated/tables.tex"))
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Resolve inputs (prefer paper_experiments outputs; fall back to fixed baselines).
    pg19_baseline = _load_json(args.results_dir / "pg19_baseline.json") or _load_json(
        args.baseline_dir / "pg19_baseline_avg.json"
    )
    wikitext_baseline = _load_json(args.results_dir / "wikitext_baseline.json") or _load_json(
        args.baseline_dir / "wikitext_baseline_avg.json"
    )

    pg19_mit = _load_json(args.results_dir / "pg19_mit.json")
    pg19_ours = _load_json(args.results_dir / "pg19_ours.json")

    wikitext_mit = _load_json(args.results_dir / "wikitext_mit.json")
    wikitext_ours = _load_json(args.results_dir / "wikitext_ours.json")

    table_pg19 = _table_main_pg19(
        [
            _extract_row("Baseline (Full KV)", pg19_baseline, assume_speedup_one=True),
            _extract_row("StreamingLLM (MIT)", pg19_mit),
            _extract_row("Ours (Lazy/Slack/Max\\_Drop)", pg19_ours),
        ]
    )
    table_wiki = _table_main_wikitext(
        [
            _extract_row("Baseline (Full KV)", wikitext_baseline, assume_speedup_one=True),
            _extract_row("StreamingLLM (MIT)", wikitext_mit),
            _extract_row("Ours (Lazy/Slack/Max\\_Drop)", wikitext_ours),
        ]
    )

    content = "\n\n".join([table_pg19, table_wiki]) + "\n"
    args.out.write_text(content, encoding="utf-8")
    print(f"Wrote LaTeX tables to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

