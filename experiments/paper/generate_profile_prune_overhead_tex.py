#!/usr/bin/env python3
"""
Generate a LaTeX table summarizing prune/update overhead vs R (compress_every).

This is evidence for the Lazy Pruning "amortization" claim: forward time stays
nearly constant while cache-management overhead decreases as prune frequency
drops.

Inputs are taken from existing probe outputs under:
  results/probes/profile_prune_overhead/*summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    header: list[str] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if header is None:
            header = parts
            continue
        rows.append(dict(zip(header, parts)))
    return rows


def _fmt(x: str, nd: int) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-csv",
        type=Path,
        default=Path(
            "results/probes/profile_prune_overhead/pg19_S32_cap2048_sigma16_delta0_summary.csv"
        ),
    )
    ap.add_argument("--out", type=Path, default=Path("NeurIPS/generated/profile_prune_overhead.tex"))
    ap.add_argument("--r-small", type=int, default=1)
    ap.add_argument("--r-large", type=int, default=64)
    args = ap.parse_args()

    rows = _read_csv(args.in_csv)
    by_r = {int(r["compress_every"]): r for r in rows if "compress_every" in r}
    if args.r_small not in by_r or args.r_large not in by_r:
        raise SystemExit(
            f"Missing rows for R={args.r_small} or R={args.r_large} in {args.in_csv}"
        )

    def row(r: int) -> dict[str, str]:
        d = by_r[r]
        return {
            "R": str(r),
            "prunes": d.get("prune_events", "?"),
            "forward": _fmt(d.get("forward_ms_mean", "?"), 2),
            "update": _fmt(d.get("update_ms_mean", "?"), 2),
            "total": _fmt(d.get("total_ms_mean", "?"), 2),
        }

    a = row(args.r_small)
    b = row(args.r_large)

    # NOTE: we use str.format() below, so all LaTeX braces must be doubled.
    tex = r"""
\begin{{table}}[H]
\centering
\caption{{Profiling evidence for amortization (PG19 probe; fixed $C_{{\text{{cap}}}}=2048$, $S=32$, $\sigma=16$): forward time stays nearly constant while cache-update overhead is amortized by increasing $R$.}}
\label{{tab:profile_prune}}
\small
\begin{{tabular}}{{lcccc}}
\toprule
Setting & Prune events & Forward (ms) & Update (ms) & Total (ms) \\
\midrule
Strict ($R={r_small}$) & {prunes_a} & {forward_a} & {update_a} & {total_a} \\
Lazy ($R={r_large}$) & {prunes_b} & {forward_b} & {update_b} & {total_b} \\
\bottomrule
\end{{tabular}}
\end{{table}}
""".lstrip().format(
        r_small=a["R"],
        r_large=b["R"],
        prunes_a=a["prunes"],
        forward_a=a["forward"],
        update_a=a["update"],
        total_a=a["total"],
        prunes_b=b["prunes"],
        forward_b=b["forward"],
        update_b=b["update"],
        total_b=b["total"],
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(tex, encoding="utf-8")
    print(f"Wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
