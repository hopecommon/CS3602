#!/bin/bash
################################################################################
# Paper experiments (one-click) + auto-generated LaTeX tables.
#
# - Runs a minimal, paper-oriented set of evaluations for:
#   * Baseline (full KV)
#   * MIT StreamingLLM (Start+Recent)
#   * Ours (StreamingLLM + Lazy Prune + Slack + Max_Drop)
# - Reuses a fixed baseline by default (to avoid repeatedly running baseline).
# - Generates: `NeurIPS/generated/tables.tex` (used by `neurips_2025_compressed.tex`).
#
# Usage:
#   chmod +x run_paper_experiments.sh
#   ./run_paper_experiments.sh
################################################################################

set -euo pipefail

# ------------------------------------------------------------------------------
# Load .env if present
# ------------------------------------------------------------------------------
ENV_FILE=".env"
if [[ -f "$ENV_FILE" ]]; then
  set -o allexport
  source "$ENV_FILE"
  set +o allexport
fi

# ------------------------------------------------------------------------------
# Environment defaults (offline-friendly)
# ------------------------------------------------------------------------------
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

PYTHON="${PYTHON_BIN:-kvpress/.venv/bin/python}"
MODEL_NAME="${MODEL_NAME:-EleutherAI/pythia-2.8b}"

RESULTS_DIR="${RESULTS_DIR:-results/paper_experiments}"
BASELINE_DIR="${BASELINE_DIR:-results/baselines}"
mkdir -p "$RESULTS_DIR"

WIKITEXT_TOKENS="${WIKITEXT_TOKENS:-4096}"
PG19_TOKENS="${PG19_TOKENS:-20000}"

# How many independent samples for dataset evaluation (mean across samples).
MAX_SAMPLES_WIKI="${MAX_SAMPLES_WIKI:-64}"
MAX_SAMPLES_PG19="${MAX_SAMPLES_PG19:-10}"

# Ours default config (override via env if needed).
OURS_SINK="${OURS_SINK:-32}"
OURS_WINDOW="${OURS_WINDOW:-2016}"
OURS_COMPRESS_EVERY="${OURS_COMPRESS_EVERY:-64}"
OURS_CACHE_SLACK="${OURS_CACHE_SLACK:-16}"
OURS_MAX_DROP="${OURS_MAX_DROP:-32}"

# MIT baseline config (Start+Recent)
MIT_SINK="${MIT_SINK:-4}"
MIT_WINDOW="${MIT_WINDOW:-2044}"

# Run knobs (1=run, 0=skip)
RUN_MAIN="${RUN_MAIN:-1}"
RUN_ABLATIONS="${RUN_ABLATIONS:-1}"
RUN_SWEEPS="${RUN_SWEEPS:-1}"

SWEEP_R_VALUES="${SWEEP_R_VALUES:-1 4 16 32 64 128}"
SWEEP_SIGMA_VALUES="${SWEEP_SIGMA_VALUES:-0 8 16 32 64}"
SWEEP_DELTA_VALUES="${SWEEP_DELTA_VALUES:-0 8 16 32 64}"

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
  local dataset="$1"      # wikitext|pg19
  local src="$2"          # path to baseline avg json
  local dst="$3"          # destination in paper dir

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
  echo "  Please generate it first (once), then re-run this script."
  echo "  Example:"
  echo "    $PYTHON experiments/eval_streaming_llm.py --mode baseline --dataset-name $dataset ..."
  exit 1
}

###############################################################################
# 0) Reuse fixed baselines (do not rerun baseline repeatedly)
###############################################################################
ensure_baseline_link "wikitext" "$BASELINE_DIR/wikitext_baseline_avg.json" "$RESULTS_DIR/wikitext_baseline.json"
ensure_baseline_link "pg19" "$BASELINE_DIR/pg19_baseline_avg.json" "$RESULTS_DIR/pg19_baseline.json"

if [[ "$RUN_MAIN" == "1" ]]; then
  ###############################################################################
  # 1) MIT StreamingLLM (Start+Recent)
  ###############################################################################
  run "MIT StreamingLLM (PG19)" \
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

  run "MIT StreamingLLM (WikiText)" \
    "$PYTHON" experiments/eval_streaming_llm.py \
      --model-name "$MODEL_NAME" \
      --dataset-name wikitext \
      --dataset-config wikitext-103-v1 \
      --split test \
      --max-samples "$MAX_SAMPLES_WIKI" \
      --max-eval-tokens "$WIKITEXT_TOKENS" \
      --max-length 2048 \
      --stride 1022 \
      --n-sink "$MIT_SINK" \
      --window-size "$MIT_WINDOW" \
      --compress-every 1 \
      --streaming-mode mit \
      --mode streaming \
      --baseline-results "$RESULTS_DIR/wikitext_baseline.json" \
      --output "$RESULTS_DIR/wikitext_mit.json"

  ###############################################################################
  # 2) Ours (Lazy + Slack + Max_Drop)
  ###############################################################################
  run "Ours (PG19)" \
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

  run "Ours (WikiText)" \
    "$PYTHON" experiments/eval_streaming_llm.py \
      --model-name "$MODEL_NAME" \
      --dataset-name wikitext \
      --dataset-config wikitext-103-v1 \
      --split test \
      --max-samples "$MAX_SAMPLES_WIKI" \
      --max-eval-tokens "$WIKITEXT_TOKENS" \
      --max-length 2048 \
      --stride 1022 \
      --n-sink "$OURS_SINK" \
      --window-size "$OURS_WINDOW" \
      --compress-every "$OURS_COMPRESS_EVERY" \
      --cache-slack "$OURS_CACHE_SLACK" \
      --max-drop "$OURS_MAX_DROP" \
      --streaming-mode ours \
      --mode streaming \
      --baseline-results "$RESULTS_DIR/wikitext_baseline.json" \
      --output "$RESULTS_DIR/wikitext_ours.json"
fi

###############################################################################
# 2.5) Ablations (compact: Slack / Max_Drop)
###############################################################################
if [[ "$RUN_ABLATIONS" == "1" ]]; then
  ABL_DIR="$RESULTS_DIR/ablations"
  mkdir -p "$ABL_DIR"

  run "Ablation (PG19) w/o Slack" \
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
      --cache-slack 0 \
      --max-drop "$OURS_MAX_DROP" \
      --streaming-mode ours \
      --mode streaming \
      --baseline-results "$RESULTS_DIR/pg19_baseline.json" \
      --output "$ABL_DIR/pg19_no_slack.json"

  run "Ablation (PG19) w/o Max_Drop" \
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
      --max-drop 0 \
      --streaming-mode ours \
      --mode streaming \
      --baseline-results "$RESULTS_DIR/pg19_baseline.json" \
      --output "$ABL_DIR/pg19_no_maxdrop.json"

  run "Ablation (PG19) full" \
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
      --output "$ABL_DIR/pg19_full.json"
fi

###############################################################################
# 2.6) Sweeps (R / sigma / delta) for PG19
###############################################################################
if [[ "$RUN_SWEEPS" == "1" ]]; then
  SWEEP_DIR="$RESULTS_DIR/sweeps"
  mkdir -p "$SWEEP_DIR/R" "$SWEEP_DIR/sigma" "$SWEEP_DIR/delta"

  for R in $SWEEP_R_VALUES; do
    run "Sweep R=$R (PG19)" \
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
        --compress-every "$R" \
        --cache-slack "$OURS_CACHE_SLACK" \
        --max-drop "$OURS_MAX_DROP" \
        --streaming-mode ours \
        --mode streaming \
        --baseline-results "$RESULTS_DIR/pg19_baseline.json" \
        --output "$SWEEP_DIR/R/pg19_R${R}.json"
  done

  for SIGMA in $SWEEP_SIGMA_VALUES; do
    run "Sweep sigma=$SIGMA (PG19)" \
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
        --cache-slack "$SIGMA" \
        --max-drop "$OURS_MAX_DROP" \
        --streaming-mode ours \
        --mode streaming \
        --baseline-results "$RESULTS_DIR/pg19_baseline.json" \
        --output "$SWEEP_DIR/sigma/pg19_sigma${SIGMA}.json"
  done

  for DELTA in $SWEEP_DELTA_VALUES; do
    run "Sweep delta=$DELTA (PG19)" \
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
        --max-drop "$DELTA" \
        --streaming-mode ours \
        --mode streaming \
        --baseline-results "$RESULTS_DIR/pg19_baseline.json" \
        --output "$SWEEP_DIR/delta/pg19_delta${DELTA}.json"
  done
fi

###############################################################################
# 3) Generate LaTeX tables for the paper
###############################################################################
run "Generate LaTeX tables" \
  "$PYTHON" experiments/paper/generate_tables_tex.py \
    --results-dir "$RESULTS_DIR" \
    --baseline-dir "$BASELINE_DIR" \
    --out "NeurIPS/generated/tables.tex"

run "Generate ablation tables" \
  "$PYTHON" experiments/paper/generate_ablations_tex.py \
    --results-dir "$RESULTS_DIR" \
    --out "NeurIPS/generated/ablations.tex"

run "Generate sweep tables (supplementary)" \
  "$PYTHON" experiments/paper/generate_sweeps_tex.py \
    --results-dir "$RESULTS_DIR" \
    --out "NeurIPS/generated/sweeps.tex"

run "Generate negative-results table (qualitative)" \
  "$PYTHON" experiments/paper/generate_negative_results_tex.py \
    --out "NeurIPS/generated/negative_results.tex"

echo ""
echo "Done."
echo "- Results: $RESULTS_DIR/"
echo "- Paper tables: NeurIPS/generated/tables.tex"
echo "- Compile paper: (cd NeurIPS && pdflatex neurips_2025_compressed.tex)"
