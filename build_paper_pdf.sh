#!/bin/bash
################################################################################
# Build the NeurIPS paper PDF (multi-pass LaTeX + BibTeX).
#
# Runs correctly from any working directory (auto-cd to repo root).
#
# Usage:
#   chmod +x build_paper_pdf.sh
#   ./build_paper_pdf.sh
################################################################################

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT/NeurIPS"

TEX="neurips_2025_compressed.tex"
JOB="neurips_2025_compressed"

if [[ ! -f "$TEX" ]]; then
  echo "Missing TeX file: $REPO_ROOT/NeurIPS/$TEX" >&2
  exit 1
fi

if ! command -v pdflatex >/dev/null 2>&1; then
  echo "pdflatex not found in PATH" >&2
  exit 1
fi

echo "Building: $REPO_ROOT/NeurIPS/$TEX"

if command -v python >/dev/null 2>&1; then
  # Regenerate lightweight assets from existing JSON results (no GPU required).
  python "$REPO_ROOT/experiments/paper/generate_fig1_kv_length.py" \
    --results-dir "$REPO_ROOT/results/paper_experiments" \
    --steps 256 \
    --out-tex "$REPO_ROOT/NeurIPS/generated/fig1_kv_length.tex" >/dev/null || true
  python "$REPO_ROOT/experiments/paper/generate_ablations_tex.py" \
    --results-dir "$REPO_ROOT/results/paper_experiments" \
    --out "$REPO_ROOT/NeurIPS/generated/ablations.tex" >/dev/null || true
  python "$REPO_ROOT/experiments/paper/generate_negative_results_tex.py" \
    --out "$REPO_ROOT/NeurIPS/generated/negative_results.tex" >/dev/null || true
  python "$REPO_ROOT/experiments/paper/generate_profile_prune_overhead_tex.py" \
    --in-csv "$REPO_ROOT/results/probes/profile_prune_overhead/pg19_S32_cap2048_sigma16_delta0_summary.csv" \
    --out "$REPO_ROOT/NeurIPS/generated/profile_prune_overhead.tex" >/dev/null || true
  python "$REPO_ROOT/experiments/paper/generate_wikitext_sanity_tex.py" \
    --results-dir "$REPO_ROOT/results/paper_experiments" \
    --out "$REPO_ROOT/NeurIPS/generated/wikitext_sanity.tex" >/dev/null || true
fi

pdflatex -interaction=nonstopmode -halt-on-error "$TEX" >/dev/null

if command -v bibtex >/dev/null 2>&1; then
  # BibTeX is only needed if the .aux requests it; running it is harmless otherwise.
  bibtex "$JOB" >/dev/null || true
fi

pdflatex -interaction=nonstopmode -halt-on-error "$TEX" >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error "$TEX" >/dev/null

echo "OK: $REPO_ROOT/NeurIPS/$JOB.pdf"
