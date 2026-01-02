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

pdflatex -interaction=nonstopmode -halt-on-error "$TEX" >/dev/null

if command -v bibtex >/dev/null 2>&1; then
  # BibTeX is only needed if the .aux requests it; running it is harmless otherwise.
  bibtex "$JOB" >/dev/null || true
fi

pdflatex -interaction=nonstopmode -halt-on-error "$TEX" >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error "$TEX" >/dev/null

echo "OK: $REPO_ROOT/NeurIPS/$JOB.pdf"

