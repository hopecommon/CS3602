#!/bin/bash
################################################################################
# Quick paper sanity check (small runs) + table generation.
#
# This script is for fast validation only (not for final numbers).
# It reuses fixed baselines and runs a smaller sample count.
################################################################################

set -euo pipefail

ENV_FILE=".env"
if [[ -f "$ENV_FILE" ]]; then
  set -o allexport
  source "$ENV_FILE"
  set +o allexport
fi

export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

PYTHON="${PYTHON_BIN:-kvpress/.venv/bin/python}"
MODEL_NAME="${MODEL_NAME:-EleutherAI/pythia-2.8b}"

RESULTS_DIR="${RESULTS_DIR:-results/paper_quick_check}"
BASELINE_DIR="${BASELINE_DIR:-results/baselines}"
mkdir -p "$RESULTS_DIR"

WIKITEXT_TOKENS="${WIKITEXT_TOKENS:-4096}"
PG19_TOKENS="${PG19_TOKENS:-20000}"

MAX_SAMPLES_WIKI="${MAX_SAMPLES_WIKI:-8}"
MAX_SAMPLES_PG19="${MAX_SAMPLES_PG19:-2}"

OURS_SINK="${OURS_SINK:-32}"
OURS_WINDOW="${OURS_WINDOW:-2016}"
OURS_COMPRESS_EVERY="${OURS_COMPRESS_EVERY:-64}"
OURS_CACHE_SLACK="${OURS_CACHE_SLACK:-16}"
OURS_MAX_DROP="${OURS_MAX_DROP:-32}"

MIT_SINK="${MIT_SINK:-4}"
MIT_WINDOW="${MIT_WINDOW:-2044}"

# Disable ablations by default in quick mode
RUN_ABLATIONS="${RUN_ABLATIONS:-0}"

run() {
  local name="$1"
  shift
  echo ""
  echo "================================================================"
  echo "$name"
  echo "================================================================"
  printf 'cmd: '
  printf '%q ' "$@"
  echo
  "$@"
}

ensure_baseline_link() {
  local dataset="$1"
  local src="$2"
  local dst="$3"

  if [[ -f "$dst" && -s "$dst" ]]; then
    echo "✓ baseline exists: $dst"
    return 0
  fi
  if [[ -f "$src" && -s "$src" ]]; then
    # Avoid `realpath` for portability (e.g., macOS may not ship it by default).
    ln -sf "$src" "$dst"
    echo "✓ baseline linked: $dst -> $src"
    return 0
  fi
  echo "✗ baseline missing for $dataset: expected $src"
  exit 1
}

ensure_baseline_link "wikitext" "$BASELINE_DIR/wikitext_baseline_avg.json" "$RESULTS_DIR/wikitext_baseline.json"
ensure_baseline_link "pg19" "$BASELINE_DIR/pg19_baseline_avg.json" "$RESULTS_DIR/pg19_baseline.json"

run "MIT StreamingLLM (PG19, quick)" \
  "$PYTHON" experiments/eval_streaming_llm.py \
    --model-name "$MODEL_NAME" \
    --dataset-name pg19 \
    --dataset-config pg19 \
    --split test \
    --max-samples "$MAX_SAMPLES_PG19" \
    --max-eval-tokens "$PG19_TOKENS" \
    --max-length 2048 \
    --stride 1022 \
    --n-sink "$MIT_SINK" \
    --window-size "$MIT_WINDOW" \
    --compress-every 1 \
    --streaming-mode mit \
    --mode streaming \
    --baseline-results "$RESULTS_DIR/pg19_baseline.json" \
    --output "$RESULTS_DIR/pg19_mit.json"

run "Ours (PG19, quick)" \
  "$PYTHON" experiments/eval_streaming_llm.py \
    --model-name "$MODEL_NAME" \
    --dataset-name pg19 \
    --dataset-config pg19 \
    --split test \
    --max-samples "$MAX_SAMPLES_PG19" \
    --max-eval-tokens "$PG19_TOKENS" \
    --max-length 2048 \
    --stride 1022 \
    --n-sink "$OURS_SINK" \
    --window-size "$OURS_WINDOW" \
    --compress-every "$OURS_COMPRESS_EVERY" \
    --cache-slack "$OURS_CACHE_SLACK" \
    --max-drop "$OURS_MAX_DROP" \
    --streaming-mode ours \
    --mode streaming \
    --baseline-results "$RESULTS_DIR/pg19_baseline.json" \
    --output "$RESULTS_DIR/pg19_ours.json"

run "Generate LaTeX tables (quick)" \
  "$PYTHON" experiments/paper/generate_tables_tex.py \
    --results-dir "$RESULTS_DIR" \
    --baseline-dir "$BASELINE_DIR" \
    --out "NeurIPS/generated/tables.tex"

if [[ "$RUN_ABLATIONS" == "1" ]]; then
  "$PYTHON" experiments/paper/generate_ablations_tex.py \
    --results-dir "$RESULTS_DIR" \
    --out "NeurIPS/generated/ablations.tex"
fi

"$PYTHON" experiments/paper/generate_negative_results_tex.py \
  --out "NeurIPS/generated/negative_results.tex"

echo ""
echo "Done."
echo "- Results: $RESULTS_DIR/"
echo "- Paper tables: NeurIPS/generated/tables.tex"
