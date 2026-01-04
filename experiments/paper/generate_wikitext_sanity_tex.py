#!/usr/bin/env python3
"""
Generate a LaTeX sanity-check table for WikiText-103 from paper JSON outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(x: Any, nd: int) -> str:
    return f"{float(x):.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results/paper_experiments"))
    ap.add_argument("--out", type=Path, default=Path("NeurIPS/generated/wikitext_sanity.tex"))
    args = ap.parse_args()

    base = _load(args.results_dir / "wikitext_baseline.json")["baseline"]
    strict = _load(args.results_dir / "wikitext_start_recent.json")["streaming"]
    lazy = _load(args.results_dir / "wikitext_ours.json")["streaming"]

    base_tpot = float(base["tpot_ms"])
    strict_speedup = base_tpot / float(strict["tpot_ms"])
    lazy_speedup = base_tpot / float(lazy["tpot_ms"])

    tex = rf"""
\begin{{table}}[H]
\centering
\caption{{Sanity-check results on WikiText-103 (short-context).}}
\small
\begin{{tabular}}{{lccc}}
\toprule
Method & TPOT$\downarrow$ & Speedup$\uparrow$ & PPL$\downarrow$ \\
\midrule
Baseline (Sliding Window, no KV) & {_fmt(base['tpot_ms'],2)} & 1.00$\times$ & {_fmt(base['perplexity'],3)} \\
StreamingLLM (Start+Recent; strict prune) & {_fmt(strict['tpot_ms'],2)} & {_fmt(strict_speedup,2)}$\times$ & {_fmt(strict['perplexity'],3)} \\
Ours (Lazy Pruning) & {_fmt(lazy['tpot_ms'],2)} & {_fmt(lazy_speedup,2)}$\times$ & {_fmt(lazy['perplexity'],3)} \\
\bottomrule
\end{{tabular}}
\end{{table}}
""".lstrip()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(tex, encoding="utf-8")
    print(f"Wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

