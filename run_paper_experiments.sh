#!/bin/bash
################################################################################
# Paper experiments (one-click) + auto-generated LaTeX tables.
#
# - Runs a minimal, paper-oriented set of evaluations for:
#   * Baseline (sliding window, no KV cache; same context cap)
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

# Ensure we run from the repo root even when invoked from elsewhere.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ------------------------------------------------------------------------------
# Flags
# ------------------------------------------------------------------------------
# -f / --force: rerun even if outputs already exist
FORCE=0
for arg in "$@"; do
  case "$arg" in
    -f|--force)
      FORCE=1
      ;;
  esac
done

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

# Pin presampled files for reproducibility across machines (override via env).
# These variables are consumed by experiments/eval_utils.py.
if [[ -z "${WIKITEXT_SAMPLE_LENGTH:-}" && -z "${WIKITEXT_SAMPLE_FILE:-}" ]]; then
  export WIKITEXT_SAMPLE_LENGTH="$WIKITEXT_TOKENS"
fi
if [[ -z "${PG19_SAMPLE_LENGTH:-}" && -z "${PG19_SAMPLE_FILE:-}" ]]; then
  export PG19_SAMPLE_LENGTH="$PG19_TOKENS"
fi

# How many independent samples for dataset evaluation (mean across samples).
MAX_SAMPLES_WIKI="${MAX_SAMPLES_WIKI:-64}"
MAX_SAMPLES_PG19="${MAX_SAMPLES_PG19:-10}"

# ---------------------------------------------------------------------------
# Main-result fairness: align (S, W) between MIT and Ours.
# ---------------------------------------------------------------------------
# By default, we set (S, W) to match the canonical StreamingLLM setting.
MAIN_SINK="${MAIN_SINK:-4}"
MAIN_WINDOW="${MAIN_WINDOW:-2044}" # ensures MAIN_SINK+MAIN_WINDOW=2048

# Ours default hyperparameters (with aligned S/W unless explicitly overridden).
OURS_SINK="${OURS_SINK:-$MAIN_SINK}"
OURS_WINDOW="${OURS_WINDOW:-$MAIN_WINDOW}"
OURS_COMPRESS_EVERY="${OURS_COMPRESS_EVERY:-64}"
OURS_CACHE_SLACK="${OURS_CACHE_SLACK:-16}"
OURS_MAX_DROP="${OURS_MAX_DROP:-32}"

# MIT baseline config (Start+Recent) - aligned to MAIN_* by default.
MIT_SINK="${MIT_SINK:-$MAIN_SINK}"
MIT_WINDOW="${MIT_WINDOW:-$MAIN_WINDOW}"

# Optional additional (S, W) pair to measure sink-size confound explicitly.
EXTRA_SINK="${EXTRA_SINK:-32}"
EXTRA_WINDOW="${EXTRA_WINDOW:-2016}" # ensures EXTRA_SINK+EXTRA_WINDOW=2048
RUN_SINK_CONFOUND="${RUN_SINK_CONFOUND:-1}"

# Repeatability protocol
# Default to a single-pass run (fast iteration). Increase these for final numbers.
WARMUP_RUNS="${WARMUP_RUNS:-0}"
REPEAT_RUNS="${REPEAT_RUNS:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
if [[ "$FORCE" == "1" ]]; then
  SKIP_EXISTING=0
fi

# Run knobs (1=run, 0=skip)
RUN_MAIN="${RUN_MAIN:-1}"
RUN_ABLATIONS="${RUN_ABLATIONS:-1}"
RUN_SWEEPS="${RUN_SWEEPS:-1}"

SWEEP_R_VALUES="${SWEEP_R_VALUES:-1 4 16 32 64 128}"
SWEEP_SIGMA_VALUES="${SWEEP_SIGMA_VALUES:-0 8 16 32 64}"
SWEEP_DELTA_VALUES="${SWEEP_DELTA_VALUES:-0 8 16 32 64}"

# Baseline generation (run once if missing)
AUTO_BASELINE="${AUTO_BASELINE:-1}"
BASELINE_RUNS="${BASELINE_RUNS:-1}"
BASELINE_SINK="${BASELINE_SINK:-4}"
BASELINE_WINDOW="${BASELINE_WINDOW:-2044}"
BASELINE_WIKITEXT="${BASELINE_WIKITEXT:-$BASELINE_DIR/wikitext_baseline_avg.json}"
BASELINE_PG19="${BASELINE_PG19:-$BASELINE_DIR/pg19_baseline_avg.json}"

# Baseline compatibility check:
# - STRICT_BASELINE_CHECK=0 (default): warn on mismatch but keep going (do not block runs).
# - STRICT_BASELINE_CHECK=1: mismatch triggers baseline regeneration to avoid speedup drift.
STRICT_BASELINE_CHECK="${STRICT_BASELINE_CHECK:-0}"

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

run_repeated() {
  local name="$1"
  local out_path="$2"
  shift 2
  echo ""
  echo "================================================================"
  echo "$name"
  echo "================================================================"
  "$PYTHON" experiments/paper/run_repeated_eval.py \
    --warmup "$WARMUP_RUNS" \
    --runs "$REPEAT_RUNS" \
    $( [[ "$SKIP_EXISTING" == "1" ]] && echo "--skip-existing" ) \
    --out "$out_path" \
    -- "$@"
}

abs_path() {
  local p="$1"
  if [[ "$p" = /* ]]; then
    echo "$p"
  else
    echo "$REPO_ROOT/$p"
  fi
}

ensure_fixed_baselines() {
  if [[ "$AUTO_BASELINE" != "1" ]]; then
    return 0
  fi
  if [[ "$FORCE" != "1" && -f "$BASELINE_WIKITEXT" && -s "$BASELINE_WIKITEXT" && -f "$BASELINE_PG19" && -s "$BASELINE_PG19" ]]; then
    return 0
  fi
  echo "Fixed baselines missing; generating them now (runs=$BASELINE_RUNS)..."
  run "Generate fixed baselines" \
    "$PYTHON" experiments/run_fixed_baseline.py \
      --model-name "$MODEL_NAME" \
      --dtype float16 \
      --n-sink "$BASELINE_SINK" \
      --window-size "$BASELINE_WINDOW" \
      --runs "$BASELINE_RUNS" \
      $( [[ "$FORCE" == "1" ]] && echo "--no-skip-existing" ) \
      --output-dir "$BASELINE_DIR"
}

ensure_baseline_link() {
  local dataset="$1"      # wikitext|pg19
  local src="$2"          # path to baseline avg json
  local dst="$3"          # destination in paper dir

  local expected_max_length
  expected_max_length=$((BASELINE_SINK + BASELINE_WINDOW))
  local expected_stride
  expected_stride=$((BASELINE_WINDOW / 2))

  if [[ "$FORCE" != "1" && -f "$dst" && -s "$dst" ]]; then
    if "$PYTHON" experiments/paper/check_baseline_compat.py \
      --baseline "$dst" \
      --model-name "$MODEL_NAME" \
      --dtype "float16" \
      --max-length "$expected_max_length" \
      --stride "$expected_stride" \
      --max-eval-tokens "$( [[ "$dataset" == "pg19" ]] && echo "$PG19_TOKENS" || echo "$WIKITEXT_TOKENS" )" \
      --n-sink "$BASELINE_SINK" \
      --window-size "$BASELINE_WINDOW" >/dev/null 2>&1; then
      echo "✓ baseline exists (compatible): $dst"
      return 0
    fi
    echo "⚠ baseline exists but fingerprint/config mismatch: $dst"
    if [[ "$STRICT_BASELINE_CHECK" != "1" ]]; then
      echo "  Continuing with the existing baseline (STRICT_BASELINE_CHECK=0)."
      return 0
    fi
  fi

  # If the fixed baselines are missing (or broken symlinks), try to generate them once.
  if [[ ! -f "$src" || ! -s "$src" ]]; then
    ensure_fixed_baselines
  fi

  # If baseline exists but is incompatible, regenerate fixed baselines (overwrite).
  if [[ -f "$src" && -s "$src" ]]; then
    if ! "$PYTHON" experiments/paper/check_baseline_compat.py \
      --baseline "$src" \
      --model-name "$MODEL_NAME" \
      --dtype "float16" \
      --max-length "$expected_max_length" \
      --stride "$expected_stride" \
      --max-eval-tokens "$( [[ "$dataset" == "pg19" ]] && echo "$PG19_TOKENS" || echo "$WIKITEXT_TOKENS" )" \
      --n-sink "$BASELINE_SINK" \
      --window-size "$BASELINE_WINDOW" >/dev/null 2>&1; then
      echo "⚠ fixed baseline fingerprint/config mismatch: $src"
      if [[ "$STRICT_BASELINE_CHECK" == "1" ]]; then
        echo "  STRICT_BASELINE_CHECK=1 → regenerating baselines (overwrite, runs=$BASELINE_RUNS)..."
        run "Regenerate fixed baselines" \
          "$PYTHON" experiments/run_fixed_baseline.py \
            --model-name "$MODEL_NAME" \
            --dtype float16 \
            --n-sink "$BASELINE_SINK" \
            --window-size "$BASELINE_WINDOW" \
            --runs "$BASELINE_RUNS" \
            --no-skip-existing \
            --output-dir "$BASELINE_DIR"
      else
        echo "  Continuing with the existing baseline (STRICT_BASELINE_CHECK=0)."
      fi
    fi
  fi

  if [[ -f "$src" && -s "$src" ]]; then
    # Copy instead of symlink to avoid broken-link issues across machines/filesystems.
    cp -f "$src" "$dst"
    echo "✓ baseline copied: $dst <- $src"
    return 0
  fi

  echo "✗ baseline missing for $dataset: expected $src"
  echo "  Tried AUTO_BASELINE=$AUTO_BASELINE via experiments/run_fixed_baseline.py but still missing."
  exit 1
}

###############################################################################
# 0) Reuse fixed baselines (do not rerun baseline repeatedly)
###############################################################################
ensure_fixed_baselines
ensure_baseline_link "wikitext" "$BASELINE_WIKITEXT" "$RESULTS_DIR/wikitext_baseline.json"
ensure_baseline_link "pg19" "$BASELINE_PG19" "$RESULTS_DIR/pg19_baseline.json"

if [[ "$RUN_MAIN" == "1" ]]; then
  ###############################################################################
  # 1) MIT StreamingLLM (Start+Recent) - MAIN (S,W) aligned
  ###############################################################################
  run_repeated "MIT StreamingLLM (PG19)" "$RESULTS_DIR/pg19_mit.json" \
      --model-name "$MODEL_NAME" \
      --dataset-name pg19 \
      --dataset-config pg19 \
      --split test \
      --max-samples "$MAX_SAMPLES_PG19" \
      --max-eval-tokens "$PG19_TOKENS" \
      --max-length 2048 \
      --stride 1022 \
      --n-sink "$MAIN_SINK" \
      --window-size "$MAIN_WINDOW" \
      --compress-every 1 \
      --streaming-mode mit \
      --cache-backend mit \
      --mode streaming \
      --baseline-results "$RESULTS_DIR/pg19_baseline.json"

  run_repeated "MIT StreamingLLM (WikiText)" "$RESULTS_DIR/wikitext_mit.json" \
      --model-name "$MODEL_NAME" \
      --dataset-name wikitext \
      --dataset-config wikitext-103-v1 \
      --split test \
      --max-samples "$MAX_SAMPLES_WIKI" \
      --max-eval-tokens "$WIKITEXT_TOKENS" \
      --max-length 2048 \
      --stride 1022 \
      --n-sink "$MAIN_SINK" \
      --window-size "$MAIN_WINDOW" \
      --compress-every 1 \
      --streaming-mode mit \
      --cache-backend mit \
      --mode streaming \
      --baseline-results "$RESULTS_DIR/wikitext_baseline.json"

  ###############################################################################
  # 1.1) Fairness check: same semantics, different implementation
  # Ours-framework-only: Start+Recent semantics using our wrapper codepath.
  ###############################################################################
  run_repeated "Ours-framework-only (PG19; Start+Recent semantics)" "$RESULTS_DIR/pg19_ours_framework_only.json" \
      --model-name "$MODEL_NAME" \
      --dataset-name pg19 \
      --dataset-config pg19 \
      --split test \
      --max-samples "$MAX_SAMPLES_PG19" \
      --max-eval-tokens "$PG19_TOKENS" \
      --max-length 2048 \
      --stride 1022 \
      --n-sink "$MAIN_SINK" \
      --window-size "$MAIN_WINDOW" \
      --compress-every 1 \
      --cache-slack 0 \
      --max-drop 0 \
      --streaming-mode ours \
      --cache-backend mit \
      --mode streaming \
      --baseline-results "$RESULTS_DIR/pg19_baseline.json"

  run_repeated "Ours-framework-only (WikiText; Start+Recent semantics)" "$RESULTS_DIR/wikitext_ours_framework_only.json" \
      --model-name "$MODEL_NAME" \
      --dataset-name wikitext \
      --dataset-config wikitext-103-v1 \
      --split test \
      --max-samples "$MAX_SAMPLES_WIKI" \
      --max-eval-tokens "$WIKITEXT_TOKENS" \
      --max-length 2048 \
      --stride 1022 \
      --n-sink "$MAIN_SINK" \
      --window-size "$MAIN_WINDOW" \
      --compress-every 1 \
      --cache-slack 0 \
      --max-drop 0 \
      --streaming-mode ours \
      --cache-backend mit \
      --mode streaming \
      --baseline-results "$RESULTS_DIR/wikitext_baseline.json"

  ###############################################################################
  # 2) Ours (Lazy + Slack + Max_Drop) - MAIN (S,W) aligned
  ###############################################################################
  run_repeated "Ours (PG19)" "$RESULTS_DIR/pg19_ours.json" \
      --model-name "$MODEL_NAME" \
      --dataset-name pg19 \
      --dataset-config pg19 \
      --split test \
      --max-samples "$MAX_SAMPLES_PG19" \
      --max-eval-tokens "$PG19_TOKENS" \
      --max-length 2048 \
      --stride 1022 \
      --n-sink "$MAIN_SINK" \
      --window-size "$MAIN_WINDOW" \
      --compress-every "$OURS_COMPRESS_EVERY" \
      --cache-slack "$OURS_CACHE_SLACK" \
      --max-drop "$OURS_MAX_DROP" \
      --streaming-mode ours \
      --cache-backend ours \
      --mode streaming \
      --baseline-results "$RESULTS_DIR/pg19_baseline.json"

  run_repeated "Ours (WikiText)" "$RESULTS_DIR/wikitext_ours.json" \
      --model-name "$MODEL_NAME" \
      --dataset-name wikitext \
      --dataset-config wikitext-103-v1 \
      --split test \
      --max-samples "$MAX_SAMPLES_WIKI" \
      --max-eval-tokens "$WIKITEXT_TOKENS" \
      --max-length 2048 \
      --stride 1022 \
      --n-sink "$MAIN_SINK" \
      --window-size "$MAIN_WINDOW" \
      --compress-every "$OURS_COMPRESS_EVERY" \
      --cache-slack "$OURS_CACHE_SLACK" \
      --max-drop "$OURS_MAX_DROP" \
      --streaming-mode ours \
      --cache-backend ours \
      --mode streaming \
      --baseline-results "$RESULTS_DIR/wikitext_baseline.json"

  ###############################################################################
  # 2.1) Sink-size confound check (optional): run both MIT and Ours at (S,W)=EXTRA_*
  ###############################################################################
  if [[ "$RUN_SINK_CONFOUND" == "1" ]]; then
    run_repeated "MIT StreamingLLM (PG19, sink confound S=$EXTRA_SINK)" "$RESULTS_DIR/pg19_mit_s${EXTRA_SINK}.json" \
        --model-name "$MODEL_NAME" \
        --dataset-name pg19 \
        --dataset-config pg19 \
        --split test \
        --max-samples "$MAX_SAMPLES_PG19" \
        --max-eval-tokens "$PG19_TOKENS" \
        --max-length 2048 \
        --stride 1022 \
        --n-sink "$EXTRA_SINK" \
        --window-size "$EXTRA_WINDOW" \
        --compress-every 1 \
        --streaming-mode mit \
        --mode streaming \
        --baseline-results "$RESULTS_DIR/pg19_baseline.json"

    run_repeated "Ours (PG19, sink confound S=$EXTRA_SINK)" "$RESULTS_DIR/pg19_ours_s${EXTRA_SINK}.json" \
        --model-name "$MODEL_NAME" \
        --dataset-name pg19 \
        --dataset-config pg19 \
        --split test \
        --max-samples "$MAX_SAMPLES_PG19" \
        --max-eval-tokens "$PG19_TOKENS" \
        --max-length 2048 \
        --stride 1022 \
        --n-sink "$EXTRA_SINK" \
        --window-size "$EXTRA_WINDOW" \
        --compress-every "$OURS_COMPRESS_EVERY" \
        --cache-slack "$OURS_CACHE_SLACK" \
        --max-drop "$OURS_MAX_DROP" \
        --streaming-mode ours \
        --mode streaming \
        --baseline-results "$RESULTS_DIR/pg19_baseline.json"
  fi
fi

###############################################################################
# 2.5) Ablations (ladder): MIT -> +Lazy -> +Slack -> +Max_Drop
###############################################################################
if [[ "$RUN_ABLATIONS" == "1" ]]; then
  ABL_DIR="$RESULTS_DIR/ablations"
  mkdir -p "$ABL_DIR"

  # A0: MIT at aligned (S,W) (for ablation ladder reference).
  run_repeated "Ablation A0 (PG19) MIT (aligned S,W)" "$ABL_DIR/pg19_A0_mit.json" \
      --model-name "$MODEL_NAME" \
      --dataset-name pg19 \
      --dataset-config pg19 \
      --split test \
      --max-samples "$MAX_SAMPLES_PG19" \
      --max-eval-tokens "$PG19_TOKENS" \
      --max-length 2048 \
      --stride 1022 \
      --n-sink "$MAIN_SINK" \
      --window-size "$MAIN_WINDOW" \
      --compress-every 1 \
      --streaming-mode mit \
      --mode streaming \
      --baseline-results "$RESULTS_DIR/pg19_baseline.json"

  # A1: +Lazy only (ours impl, no slack/max_drop).
  run_repeated "Ablation A1 (PG19) +Lazy (ours, no slack/max_drop)" "$ABL_DIR/pg19_A1_lazy.json" \
      --model-name "$MODEL_NAME" \
      --dataset-name pg19 \
      --dataset-config pg19 \
      --split test \
      --max-samples "$MAX_SAMPLES_PG19" \
      --max-eval-tokens "$PG19_TOKENS" \
      --max-length 2048 \
      --stride 1022 \
      --n-sink "$MAIN_SINK" \
      --window-size "$MAIN_WINDOW" \
      --compress-every "$OURS_COMPRESS_EVERY" \
      --cache-slack 0 \
      --max-drop 0 \
      --streaming-mode ours \
      --mode streaming \
      --baseline-results "$RESULTS_DIR/pg19_baseline.json"

  # A2: +Lazy +Slack (no max_drop).
  run_repeated "Ablation A2 (PG19) +Lazy+Slack (ours, no max_drop)" "$ABL_DIR/pg19_A2_lazy_slack.json" \
      --model-name "$MODEL_NAME" \
      --dataset-name pg19 \
      --dataset-config pg19 \
      --split test \
      --max-samples "$MAX_SAMPLES_PG19" \
      --max-eval-tokens "$PG19_TOKENS" \
      --max-length 2048 \
      --stride 1022 \
      --n-sink "$MAIN_SINK" \
      --window-size "$MAIN_WINDOW" \
      --compress-every "$OURS_COMPRESS_EVERY" \
      --cache-slack "$OURS_CACHE_SLACK" \
      --max-drop 0 \
      --streaming-mode ours \
      --mode streaming \
      --baseline-results "$RESULTS_DIR/pg19_baseline.json"

  # A3: +Lazy +Slack +Max_Drop (full).
  run_repeated "Ablation A3 (PG19) +Lazy+Slack+Max_Drop (full)" "$ABL_DIR/pg19_A3_full.json" \
      --model-name "$MODEL_NAME" \
      --dataset-name pg19 \
      --dataset-config pg19 \
      --split test \
      --max-samples "$MAX_SAMPLES_PG19" \
      --max-eval-tokens "$PG19_TOKENS" \
      --max-length 2048 \
      --stride 1022 \
      --n-sink "$MAIN_SINK" \
      --window-size "$MAIN_WINDOW" \
      --compress-every "$OURS_COMPRESS_EVERY" \
      --cache-slack "$OURS_CACHE_SLACK" \
      --max-drop "$OURS_MAX_DROP" \
      --streaming-mode ours \
      --mode streaming \
      --baseline-results "$RESULTS_DIR/pg19_baseline.json"

  # A(-Lazy): strict pruning (ours impl, immediate prune semantics).
  # This is the "w/o Lazy" control: R=1, slack=0, max_drop=0.
  run_repeated "Ablation A(-Lazy) (PG19) w/o Lazy (ours: R=1, no slack/max_drop)" "$ABL_DIR/pg19_Aneg_lazy_strict.json" \
      --model-name "$MODEL_NAME" \
      --dataset-name pg19 \
      --dataset-config pg19 \
      --split test \
      --max-samples "$MAX_SAMPLES_PG19" \
      --max-eval-tokens "$PG19_TOKENS" \
      --max-length 2048 \
      --stride 1022 \
      --n-sink "$MAIN_SINK" \
      --window-size "$MAIN_WINDOW" \
      --compress-every 1 \
      --cache-slack 0 \
      --max-drop 0 \
      --streaming-mode ours \
      --mode streaming \
      --baseline-results "$RESULTS_DIR/pg19_baseline.json"
fi

###############################################################################
# 2.6) Sweeps (R / sigma / delta) for PG19
###############################################################################
if [[ "$RUN_SWEEPS" == "1" ]]; then
  SWEEP_DIR="$RESULTS_DIR/sweeps"
  mkdir -p "$SWEEP_DIR/R" "$SWEEP_DIR/sigma" "$SWEEP_DIR/delta"

  for R in $SWEEP_R_VALUES; do
    run_repeated "Sweep R=$R (PG19)" "$SWEEP_DIR/R/pg19_R${R}.json" \
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
        --baseline-results "$RESULTS_DIR/pg19_baseline.json"
  done

  for SIGMA in $SWEEP_SIGMA_VALUES; do
    run_repeated "Sweep sigma=$SIGMA (PG19)" "$SWEEP_DIR/sigma/pg19_sigma${SIGMA}.json" \
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
        --baseline-results "$RESULTS_DIR/pg19_baseline.json"
  done

  for DELTA in $SWEEP_DELTA_VALUES; do
    run_repeated "Sweep delta=$DELTA (PG19)" "$SWEEP_DIR/delta/pg19_delta${DELTA}.json" \
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
        --baseline-results "$RESULTS_DIR/pg19_baseline.json"
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
