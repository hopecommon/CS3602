#!/usr/bin/env python3
"""
Generate compact ablation tables (LaTeX) from JSON result files.

Design goals:
- Compact enough to fit in the 4-page paper (single small table by default).
- Never fabricate numbers: missing inputs -> [INSERT DATA].
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
class Row:
    name: str
    tpot_ms: str
    speedup: str
    ppl: str


def _extract_row(name: str, data: Optional[dict[str, Any]]) -> Row:
    if not data:
        return Row(name, PLACEHOLDER, PLACEHOLDER, PLACEHOLDER)

    block = data.get("streaming") or data.get("baseline") or {}
    metrics = data.get("metrics") or {}
    tpot = _fmt_num(block.get("tpot_ms"), ".2f")
    ppl = _fmt_num(block.get("perplexity"), ".3f")
    speedup = _fmt_num(metrics.get("speedup"), ".2f")
    if speedup != PLACEHOLDER:
        speedup = f"{speedup}$\\times$"
    return Row(name, tpot, speedup, ppl)


def _render_table(rows: list[Row]) -> str:
    lines: list[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Ablation ladder on PG19 (aligned $S,W$): impact of Lazy, Slack, and Max\\_Drop.}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append("Setting & TPOT$\\downarrow$ & Speedup$\\uparrow$ & PPL$\\downarrow$ \\\\")
    lines.append("\\midrule")
    for r in rows:
        lines.append(f"{r.name} & {r.tpot_ms} & {r.speedup} & {r.ppl} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results/paper_experiments"))
    ap.add_argument("--out", type=Path, default=Path("NeurIPS/generated/ablations.tex"))
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    abl_dir = args.results_dir / "ablations"
    a0_mit = _load_json(abl_dir / "pg19_A0_mit.json")
    a_neg_lazy = _load_json(abl_dir / "pg19_Aneg_lazy_strict.json")
    a1_lazy = _load_json(abl_dir / "pg19_A1_lazy.json")
    a2_lazy_slack = _load_json(abl_dir / "pg19_A2_lazy_slack.json")
    a3_full = _load_json(abl_dir / "pg19_A3_full.json")

    tex = _render_table(
        [
            _extract_row("MIT (Start+Recent)", a0_mit),
            _extract_row("Ours w/o Lazy ($R{=}1,\\sigma{=}0,\\delta{=}0$)", a_neg_lazy),
            _extract_row("+ Lazy", a1_lazy),
            _extract_row("+ Lazy + Slack", a2_lazy_slack),
            _extract_row("+ Lazy + Slack + Max\\_Drop", a3_full),
        ]
    )
    args.out.write_text(tex, encoding="utf-8")
    print(f"Wrote ablation table to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
