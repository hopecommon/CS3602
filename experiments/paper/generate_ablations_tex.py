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
    # MIT reference JSON stores metrics directly under `metrics`.
    if not block and isinstance(metrics, dict) and "perplexity" in metrics:
        block = metrics
    tpot = _fmt_num(block.get("tpot_ms"), ".2f")
    ppl = _fmt_num(block.get("perplexity"), ".3f")
    speedup = _fmt_num(metrics.get("speedup"), ".2f")
    if speedup != PLACEHOLDER:
        speedup = f"{speedup}$\\times$"
    return Row(name, tpot, speedup, ppl)


def _render_table(rows: list[Row]) -> str:
    lines: list[str] = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{Ablation ladder on PG19 (aligned $S,W$). Differences below \\textasciitilde1\\% may fall within run-to-run noise unless stated otherwise.}")
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
    a0_start_recent = _load_json(args.results_dir / "pg19_start_recent.json")
    a_neg_lazy = _load_json(abl_dir / "pg19_Aneg_lazy_strict.json")
    a1_lazy = _load_json(abl_dir / "pg19_A1_lazy.json")
    a2_lazy_slack = _load_json(abl_dir / "pg19_A2_lazy_slack.json")
    a3_full = _load_json(abl_dir / "pg19_A3_full.json")

    def cfg_suffix(d: Optional[dict[str, Any]]) -> str:
        if not d:
            return ""
        cfg = d.get("streaming_llm") or {}
        parts = []
        R = cfg.get("compress_every")
        if R is not None:
            parts.append(f"$R{{=}}{int(R)}$")
        sigma = cfg.get("cache_slack")
        if sigma:
            parts.append(f"$\\sigma{{=}}{int(sigma)}$")
        delta = cfg.get("max_drop")
        if delta:
            parts.append(f"$\\delta{{=}}{int(delta)}$")
        if not parts:
            return ""
        return " (" + ", ".join(parts) + ")"

    rows: list[Row] = [
        _extract_row("Start+Recent (strict prune)" + cfg_suffix(a0_start_recent), a0_start_recent),
        _extract_row("Start+Recent (strict prune; framework-only)" + cfg_suffix(a_neg_lazy), a_neg_lazy),
        _extract_row("+ Lazy Pruning" + cfg_suffix(a1_lazy), a1_lazy),
    ]
    # Optional exploratory rows (only include if present to keep the paper compact).
    if a2_lazy_slack is not None:
        rows.append(_extract_row("+ (expl.) Slack" + cfg_suffix(a2_lazy_slack), a2_lazy_slack))
    if a3_full is not None:
        rows.append(_extract_row("+ (expl.) Slack + Max\\_Drop" + cfg_suffix(a3_full), a3_full))

    tex = _render_table(rows)
    args.out.write_text(tex, encoding="utf-8")
    print(f"Wrote ablation table to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
