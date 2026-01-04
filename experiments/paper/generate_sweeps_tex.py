#!/usr/bin/env python3
"""
Generate LaTeX sweep tables (R / sigma / delta) from JSON files.

These tables are intended for supplementary material or appendix (not
necessarily included in the 4-page compressed paper). Missing inputs are
rendered as placeholders.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# Ensure repo root is on sys.path when invoked as a script.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


PLACEHOLDER = "[INSERT DATA]"


def _load_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _fmt_num(x: Any, fmt: str) -> str:
    try:
        return format(float(x), fmt)
    except Exception:
        return PLACEHOLDER


@dataclass(frozen=True)
class SweepRow:
    key: str
    tpot_ms: str
    speedup: str
    ppl: str


def _extract_row(key: str, data: Optional[dict[str, Any]]) -> SweepRow:
    if not data:
        return SweepRow(key, PLACEHOLDER, PLACEHOLDER, PLACEHOLDER)
    block = data.get("streaming") or data.get("baseline") or {}
    metrics = data.get("metrics") or {}
    tpot = _fmt_num(block.get("tpot_ms"), ".2f")
    ppl = _fmt_num(block.get("perplexity"), ".3f")
    speedup = _fmt_num(metrics.get("speedup"), ".2f")
    if speedup != PLACEHOLDER:
        speedup = f"{speedup}$\\times$"
    return SweepRow(key, tpot, speedup, ppl)


def _render_table(title: str, colname: str, rows: list[SweepRow]) -> str:
    lines: list[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{title}}}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append(f"{colname} & TPOT$\\downarrow$ & Speedup$\\uparrow$ & PPL$\\downarrow$ \\\\")
    lines.append("\\midrule")
    for r in rows:
        lines.append(f"{r.key} & {r.tpot_ms} & {r.speedup} & {r.ppl} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results/paper_experiments"))
    ap.add_argument("--out", type=Path, default=Path("NeurIPS/generated/sweeps.tex"))
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    sweep_dir = args.results_dir / "sweeps"
    r_dir = sweep_dir / "R"
    s_dir = sweep_dir / "sigma"
    d_dir = sweep_dir / "delta"

    # Filenames are produced by run_paper_experiments.sh when RUN_SWEEPS=1.
    r_values = ["1", "4", "16", "32", "64", "128"]
    sigma_values = ["0", "8", "16", "32", "64"]
    delta_values = ["0", "8", "16", "32", "64"]

    tex_parts: list[str] = []
    tex_parts.append(
        _render_table(
            "Sweep over Lazy Pruning interval $R$ (PG19).",
            "$R$",
            [_extract_row(v, _load_json(r_dir / f"pg19_R{v}.json")) for v in r_values],
        )
    )
    if any((s_dir / f"pg19_sigma{v}.json").exists() for v in sigma_values):
        tex_parts.append(
            _render_table(
                "Sweep over Slack $\\sigma$ (PG19).",
                "$\\sigma$",
                [_extract_row(v, _load_json(s_dir / f"pg19_sigma{v}.json")) for v in sigma_values],
            )
        )
    if any((d_dir / f"pg19_delta{v}.json").exists() for v in delta_values):
        tex_parts.append(
            _render_table(
                "Sweep over Max\\_Drop $\\delta$ (PG19).",
                "$\\delta$",
                [_extract_row(v, _load_json(d_dir / f"pg19_delta{v}.json")) for v in delta_values],
            )
        )

    args.out.write_text("\n".join(tex_parts), encoding="utf-8")
    print(f"Wrote sweep tables to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
