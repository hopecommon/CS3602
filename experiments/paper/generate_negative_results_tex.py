#!/usr/bin/env python3
"""
Generate a compact LaTeX table summarizing negative results.

This is a qualitative summary to avoid fabricating quantitative numbers.
It is grounded in repository logs (docs/) rather than external claims.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("NeurIPS/generated/negative_results.tex"))
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Keep this table compact to fit a course paper; detailed logs live in docs/.
    tex = r"""
\begin{table}[t]
\centering
\caption{Summary of investigated but ineffective optimization routes in our batch-1 streaming setting.}
\small
\begin{tabular}{p{0.34\linewidth}p{0.18\linewidth}p{0.40\linewidth}}
\toprule
Method & Outcome & Notes (see project logs) \\
\midrule
FlashAttention / FlashDecoding & No gain / hard to integrate & Streaming already bounds attention length; remaining bottlenecks are MLP and launch/framework overhead; RoPE re-alignment complicates clean integration. \\
Speculative decoding & No gain / unreliable & Long-form generation yields low acceptance; cache consistency with pruning is fragile. \\
Quantization (TorchAO INT8/INT4) & Slower / unstable & INT8 WO v1 produced NaNs; v2 is stable but slower for batch-1 decode in our stack; INT4 backend dependencies were problematic. \\
\texttt{torch.compile} / CUDA Graphs & Unstable & Repeated-run CUDA graph overwrite errors observed in rotary-embedding path; shape/cache semantics hinder capture. \\
HF StaticCache & Incompatible & StaticCache assumes fixed cache updates; pruning can trigger device-side asserts (index out of bounds). \\
CUDA fusion (residual/LN) & No gain & Amdahl's law: residual/LN is a small fraction; custom kernel launch overhead dominated, leading to slowdown. \\
\bottomrule
\end{tabular}
\end{table}
""".lstrip()

    args.out.write_text(tex, encoding="utf-8")
    print(f"Wrote negative-results table to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

