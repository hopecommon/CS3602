#!/bin/bash
################################################################################
# Paper experiments (one-click) + auto-generated LaTeX tables.
#
# - Runs a minimal, paper-oriented set of evaluations for:
#   * Baseline (sliding window, no KV cache; same context cap)
#   * StreamingLLM (Start+Recent; strict prune)
#   * Ours (Start+Recent + Lazy Pruning)
# - Optional (exploratory): Slack/Max_Drop and other quality heuristics (Appendix).
# - Reuses a fixed baseline by default (to avoid repeatedly running baseline).
# - Generates: `NeurIPS/generated/tables.tex` (used by `neurips_2025_compressed.tex`).
#
# Usage:
#   chmod +x run_paper_experiments.sh
#   ./run_paper_experiments.sh        # run
#   ./run_paper_experiments.sh -n     # dry-run (print commands only)
#   ./run_paper_experiments.sh -f     # force rerun (ignore existing outputs)
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
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    -f|--force)
      FORCE=1
      ;;
    -n|--dry-run)
      DRY_RUN=1
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

# Result reuse policy:
# - By default, allow reusing stale aggregated results even if config_hash mismatches.
#   This avoids surprise reruns during iteration; set ALLOW_STALE_RESULTS=0 to enforce reruns.
export ALLOW_STALE_RESULTS="${ALLOW_STALE_RESULTS:-1}"

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
  # Default to a single pinned long book to avoid accidental "different book per length"
  # causing large PPL shifts. Evaluated length is still controlled by --max-eval-tokens.
  export PG19_SAMPLE_FILE="data/pg19/long_context_50000.json"
fi

# How many independent samples for dataset evaluation (mean across samples).
MAX_SAMPLES_WIKI="${MAX_SAMPLES_WIKI:-64}"
# NOTE: Our PG19 evaluation is pinned to a single presampled long-context file by default,
# so `--max-samples` is effectively ignored. Keep the metadata consistent (and runtime short).
MAX_SAMPLES_PG19="${MAX_SAMPLES_PG19:-1}"

# ---------------------------------------------------------------------------
# Main-result fairness: align (S, W) between MIT and Ours.
# ---------------------------------------------------------------------------
# We use an auto-cap design: keep a fixed total KV budget (hard cap) and derive
# `window_size` from (sink, slack, overlap, refresh_budget).
# This avoids manual (sink, window) tuning and makes Slack/Max_Drop effects measurable
# under a constant total budget.
CAP_TOTAL="${CAP_TOTAL:-2048}"
MAIN_OVERLAP="${MAIN_OVERLAP:-0}"
MAIN_REFRESH_BUDGET="${MAIN_REFRESH_BUDGET:-0}"
MAIN_REFRESH_POLICY="${MAIN_REFRESH_POLICY:-none}"

MAIN_SINK="${MAIN_SINK:-32}"

# Ours default hyperparameters (with aligned S/W unless explicitly overridden).
OURS_SINK="${OURS_SINK:-$MAIN_SINK}"
# Lazy Pruning (main): amortize cache compaction + RoPE re-alignment.
OURS_COMPRESS_EVERY="${OURS_COMPRESS_EVERY:-64}"
# Slack/Max_Drop are exploratory and default to OFF in the main paper path.
OURS_CACHE_SLACK="${OURS_CACHE_SLACK:-0}"
OURS_MAX_DROP="${OURS_MAX_DROP:-0}"

# MIT baseline config (Start+Recent) - aligned to MAIN_* by default.
MIT_SINK="${MIT_SINK:-$MAIN_SINK}"

# Optional additional (S, W) pair to measure sink-size confound explicitly.
EXTRA_SINK="${EXTRA_SINK:-4}"
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
# Optional: run MIT's official repo scripts for sanity/benchmarking.
# NOTE: MIT's StreamingLLM repo relies on attention-level pos-shift patches that are
# currently incompatible with our default Transformers version for GPTNeoX/Pythia.
# Enable these only if you know they run correctly in your environment.
RUN_MIT_REFERENCE="${RUN_MIT_REFERENCE:-0}"
RUN_MIT_BENCH="${RUN_MIT_BENCH:-0}"

# Optional: quality-oriented heuristics (Slack/Max_Drop) are not part of the main story.
RUN_QUALITY_HEURISTICS="${RUN_QUALITY_HEURISTICS:-0}"

MIT_REF_DIR="${MIT_REF_DIR:-results/mit_bench}"
MIT_REF_WIKITEXT="${MIT_REF_WIKITEXT:-$MIT_REF_DIR/wikitext_mit_reference.json}"
MIT_REF_PG19="${MIT_REF_PG19:-$MIT_REF_DIR/pg19_mit_reference.json}"
MIT_REF_WIKITEXT_WINDOWMATCH="${MIT_REF_WIKITEXT_WINDOWMATCH:-$MIT_REF_DIR/wikitext_mit_reference_windowmatch.json}"
MIT_REF_PG19_WINDOWMATCH="${MIT_REF_PG19_WINDOWMATCH:-$MIT_REF_DIR/pg19_mit_reference_windowmatch.json}"
MIT_REF_POS_SHIFT="${MIT_REF_POS_SHIFT:-0}"

MIT_BENCH_DIR="${MIT_BENCH_DIR:-results/mit_bench}"
MIT_BENCH_GEN_TOKENS="${MIT_BENCH_GEN_TOKENS:-17952}"
MIT_BENCH_PG19_STREAMING_JSON="${MIT_BENCH_PG19_STREAMING_JSON:-$MIT_BENCH_DIR/pg19_benchmark_streaming.json}"
MIT_BENCH_PG19_RECOMPUTE_JSON="${MIT_BENCH_PG19_RECOMPUTE_JSON:-$MIT_BENCH_DIR/pg19_benchmark_recompute.json}"
MIT_BENCH_WIKI_STREAMING_JSON="${MIT_BENCH_WIKI_STREAMING_JSON:-$MIT_BENCH_DIR/wikitext_benchmark_streaming.json}"
MIT_BENCH_WIKI_RECOMPUTE_JSON="${MIT_BENCH_WIKI_RECOMPUTE_JSON:-$MIT_BENCH_DIR/wikitext_benchmark_recompute.json}"

SWEEP_R_VALUES="${SWEEP_R_VALUES:-1 4 16 32 64 128}"
SWEEP_SIGMA_VALUES="${SWEEP_SIGMA_VALUES:-0 8 16 32 64}"
SWEEP_DELTA_VALUES="${SWEEP_DELTA_VALUES:-0 8 16 32 64}"

# Baseline generation (run once if missing)
AUTO_BASELINE="${AUTO_BASELINE:-1}"
BASELINE_RUNS="${BASELINE_RUNS:-1}"
# Baseline uses a sliding-window recomputation (no KV cache) and only depends on the total cap.
# Keep the (sink, window) split canonical to avoid confusing baseline JSON consumers.
BASELINE_SINK="${BASELINE_SINK:-4}"
BASELINE_WINDOW="${BASELINE_WINDOW:-2044}"
BASELINE_WIKITEXT="${BASELINE_WIKITEXT:-$BASELINE_DIR/wikitext_baseline_avg.json}"
BASELINE_PG19="${BASELINE_PG19:-$BASELINE_DIR/pg19_baseline_avg.json}"

# Baseline compatibility check:
# - STRICT_BASELINE_CHECK=0 (default): warn on mismatch but keep going (do not block runs).
# - STRICT_BASELINE_CHECK=1: mismatch triggers baseline regeneration to avoid speedup drift.
STRICT_BASELINE_CHECK="${STRICT_BASELINE_CHECK:-0}"

FP32_LOSS="${FP32_LOSS:-1}"
LOSS_FLAG="--fp32-loss"
if [[ "$FP32_LOSS" != "1" ]]; then
  LOSS_FLAG="--no-fp32-loss"
fi

calc_window() {
  local cap="$1"
  local sink="$2"
  local slack="$3"
  local overlap="$4"
  local refresh="$5"
  local w
  w=$((cap - sink - slack - overlap - refresh))
  if (( w < 1 )); then
    echo "ERROR: invalid derived window_size=$w (cap=$cap sink=$sink slack=$slack overlap=$overlap refresh=$refresh)" 1>&2
    exit 1
  fi
  echo "$w"
}

# Derive windows under the auto-cap.
MAIN_WINDOW_MIT="$(calc_window "$CAP_TOTAL" "$MAIN_SINK" "0" "$MAIN_OVERLAP" "$MAIN_REFRESH_BUDGET")"
MAIN_WINDOW_OURS="$(calc_window "$CAP_TOTAL" "$MAIN_SINK" "$OURS_CACHE_SLACK" "$MAIN_OVERLAP" "$MAIN_REFRESH_BUDGET")"

OURS_WINDOW="$(calc_window "$CAP_TOTAL" "$OURS_SINK" "$OURS_CACHE_SLACK" "$MAIN_OVERLAP" "$MAIN_REFRESH_BUDGET")"
MIT_WINDOW="$(calc_window "$CAP_TOTAL" "$MIT_SINK" "0" "$MAIN_OVERLAP" "$MAIN_REFRESH_BUDGET")"
EXTRA_WINDOW="$(calc_window "$CAP_TOTAL" "$EXTRA_SINK" "0" "$MAIN_OVERLAP" "$MAIN_REFRESH_BUDGET")"

echo "----------------------------------------------------------------"
echo "Paper experiment config"
echo "----------------------------------------------------------------"
echo "PYTHON=$PYTHON"
echo "MODEL_NAME=$MODEL_NAME"
echo "CAP_TOTAL=$CAP_TOTAL  OVERLAP=$MAIN_OVERLAP  REFRESH_BUDGET=$MAIN_REFRESH_BUDGET  REFRESH_POLICY=$MAIN_REFRESH_POLICY"
echo "MAIN: sink=$MAIN_SINK  window_mit=$MAIN_WINDOW_MIT  window_ours(slack=$OURS_CACHE_SLACK)=$MAIN_WINDOW_OURS"
echo "MIT : sink=$MIT_SINK  window=$MIT_WINDOW (slack=0)"
echo "OURS: sink=$OURS_SINK  window=$OURS_WINDOW  slack=$OURS_CACHE_SLACK  compress_every=$OURS_COMPRESS_EVERY  max_drop=$OURS_MAX_DROP"
echo "BASELINE(S,W)=($BASELINE_SINK,$BASELINE_WINDOW)  (cap=$((BASELINE_SINK + BASELINE_WINDOW)))"
echo "TOKENS: wikitext=$WIKITEXT_TOKENS  pg19=$PG19_TOKENS"
echo "REPEATS: warmup=$WARMUP_RUNS  runs=$REPEAT_RUNS  skip_existing=$SKIP_EXISTING  force=$FORCE"
echo "POLICY: ALLOW_STALE_RESULTS=$ALLOW_STALE_RESULTS  STRICT_BASELINE_CHECK=$STRICT_BASELINE_CHECK"
echo "LOSS: $LOSS_FLAG"
echo "----------------------------------------------------------------"

run() {
  local name="$1"
  shift
  echo ""
  echo "================================================================"
  echo "$name"
  echo "================================================================"
  printf 'cmd: ' 2>/dev/null || true
  printf '%q ' "$@" 2>/dev/null || true
  echo 2>/dev/null || true
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
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
  local cmd=("$PYTHON" experiments/paper/run_repeated_eval.py \
    --warmup "$WARMUP_RUNS" \
    --runs "$REPEAT_RUNS" \
    $( [[ "$SKIP_EXISTING" == "1" ]] && echo "--skip-existing" ) \
    --out "$out_path" \
    -- "$LOSS_FLAG" "$@")
  printf 'cmd: ' 2>/dev/null || true
  printf '%q ' "${cmd[@]}" 2>/dev/null || true
  echo 2>/dev/null || true
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "${cmd[@]}"
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

START_RECENT_PG19_JSON="${START_RECENT_PG19_JSON:-$RESULTS_DIR/pg19_start_recent.json}"
START_RECENT_WIKITEXT_JSON="${START_RECENT_WIKITEXT_JSON:-$RESULTS_DIR/wikitext_start_recent.json}"

###############################################################################
# 0.5) MIT reference run (use MIT's own example script)
###############################################################################
if [[ "$RUN_MIT_REFERENCE" == "1" ]]; then
  mkdir -p "$MIT_REF_DIR"

  if [[ "$SKIP_EXISTING" == "1" && -f "$MIT_REF_PG19" ]]; then
    echo "Skipping MIT reference (PG19): $MIT_REF_PG19 (already exists)"
  else
    cmd=(
      "$PYTHON" mit-streaming-llm/examples/eval_long_ppl.py
      --model_name_or_path "$MODEL_NAME"
      --dataset_name pg19
      --task pg19
      --split test
      --num_samples 1
      --output_dir "$MIT_REF_DIR/pg19"
      --enable_start_recent_kv_cache
      --start_size "$MAIN_SINK"
      --recent_size "$MAIN_WINDOW_MIT"
      --num_eval_tokens "$PG19_TOKENS"
      --max-eval-tokens "$PG19_TOKENS"
      --data-json "${PG19_SAMPLE_FILE:-data/pg19/long_context_50000.json}"
    )
    if [[ "$MIT_REF_POS_SHIFT" == "1" ]]; then
      cmd+=(--enable_pos_shift)
    fi
    cmd+=(--output-json "$MIT_REF_PG19")
    run "MIT reference (PG19; mit-streaming-llm/examples/eval_long_ppl.py)" "${cmd[@]}"
  fi

  if [[ "$SKIP_EXISTING" == "1" && -f "$MIT_REF_PG19_WINDOWMATCH" ]]; then
    echo "Skipping MIT reference (PG19; window-matched): $MIT_REF_PG19_WINDOWMATCH (already exists)"
  else
    cmd=(
      "$PYTHON" mit-streaming-llm/examples/eval_long_ppl.py
      --model_name_or_path "$MODEL_NAME"
      --dataset_name pg19
      --task pg19
      --split test
      --num_samples 1
      --output_dir "$MIT_REF_DIR/pg19_windowmatch"
      --enable_start_recent_kv_cache
      --start_size "$MAIN_SINK"
      --recent_size "$MAIN_WINDOW_OURS"
      --num_eval_tokens "$PG19_TOKENS"
      --max-eval-tokens "$PG19_TOKENS"
      --data-json "${PG19_SAMPLE_FILE:-data/pg19/long_context_50000.json}"
    )
    if [[ "$MIT_REF_POS_SHIFT" == "1" ]]; then
      cmd+=(--enable_pos_shift)
    fi
    cmd+=(--output-json "$MIT_REF_PG19_WINDOWMATCH")
    run "MIT reference (PG19; window matched to Ours)" "${cmd[@]}"
  fi

  if [[ "$SKIP_EXISTING" == "1" && -f "$MIT_REF_WIKITEXT" ]]; then
    echo "Skipping MIT reference (WikiText): $MIT_REF_WIKITEXT (already exists)"
  else
    cmd=(
      "$PYTHON" mit-streaming-llm/examples/eval_long_ppl.py
      --model_name_or_path "$MODEL_NAME"
      --dataset_name wikitext
      --task wikitext-103-v1
      --split test
      --num_samples 1
      --output_dir "$MIT_REF_DIR/wikitext"
      --enable_start_recent_kv_cache
      --start_size "$MAIN_SINK"
      --recent_size "$MAIN_WINDOW_MIT"
      --num_eval_tokens "$WIKITEXT_TOKENS"
      --max-eval-tokens "$WIKITEXT_TOKENS"
      --data-json "${WIKITEXT_SAMPLE_FILE:-data/wikitext/long_context_4096.json}"
    )
    if [[ "$MIT_REF_POS_SHIFT" == "1" ]]; then
      cmd+=(--enable_pos_shift)
    fi
    cmd+=(--output-json "$MIT_REF_WIKITEXT")
    run "MIT reference (WikiText; mit-streaming-llm/examples/eval_long_ppl.py)" "${cmd[@]}"
  fi

  if [[ "$SKIP_EXISTING" == "1" && -f "$MIT_REF_WIKITEXT_WINDOWMATCH" ]]; then
    echo "Skipping MIT reference (WikiText; window-matched): $MIT_REF_WIKITEXT_WINDOWMATCH (already exists)"
  else
    cmd=(
      "$PYTHON" mit-streaming-llm/examples/eval_long_ppl.py
      --model_name_or_path "$MODEL_NAME"
      --dataset_name wikitext
      --task wikitext-103-v1
      --split test
      --num_samples 1
      --output_dir "$MIT_REF_DIR/wikitext_windowmatch"
      --enable_start_recent_kv_cache
      --start_size "$MAIN_SINK"
      --recent_size "$MAIN_WINDOW_OURS"
      --num_eval_tokens "$WIKITEXT_TOKENS"
      --max-eval-tokens "$WIKITEXT_TOKENS"
      --data-json "${WIKITEXT_SAMPLE_FILE:-data/wikitext/long_context_4096.json}"
    )
    if [[ "$MIT_REF_POS_SHIFT" == "1" ]]; then
      cmd+=(--enable_pos_shift)
    fi
    cmd+=(--output-json "$MIT_REF_WIKITEXT_WINDOWMATCH")
    run "MIT reference (WikiText; window matched to Ours)" "${cmd[@]}"
  fi
fi

###############################################################################
# 0.6) MIT benchmark (speed/VRAM) using MIT's own benchmark script
###############################################################################
if [[ "$RUN_MIT_BENCH" == "1" ]]; then
  mkdir -p "$MIT_BENCH_DIR"

  if [[ "$SKIP_EXISTING" == "1" && -f "$MIT_BENCH_PG19_STREAMING_JSON" && -f "$MIT_BENCH_PG19_RECOMPUTE_JSON" ]]; then
    echo "Skipping MIT benchmark (PG19): outputs already exist"
  else
    run "MIT benchmark (PG19; streaming)" \
      "$PYTHON" mit-streaming-llm/examples/benchmark_streaming.py \
        --model_name_or_path "$MODEL_NAME" \
        --mode streaming \
        --device cuda \
        --start_size "$MAIN_SINK" \
        --recent_size "$MAIN_WINDOW_MIT" \
        --prefix_tokens "$PG19_TOKENS" \
        --prefill_chunk_size 512 \
        --gen_tokens "$MIT_BENCH_GEN_TOKENS" \
        --data_json "${PG19_SAMPLE_FILE:-data/pg19/long_context_50000.json}" \
        --data_text_key text \
        --data_take head \
        --output_json "$MIT_BENCH_PG19_STREAMING_JSON"

    run "MIT benchmark (PG19; recompute baseline)" \
      "$PYTHON" mit-streaming-llm/examples/benchmark_streaming.py \
        --model_name_or_path "$MODEL_NAME" \
        --mode recompute \
        --device cuda \
        --start_size "$MAIN_SINK" \
        --recent_size "$MAIN_WINDOW_MIT" \
        --prefix_tokens "$PG19_TOKENS" \
        --prefill_chunk_size 512 \
        --gen_tokens "$MIT_BENCH_GEN_TOKENS" \
        --data_json "${PG19_SAMPLE_FILE:-data/pg19/long_context_50000.json}" \
        --data_text_key text \
        --data_take head \
        --recompute_window_tokens "$CAP_TOTAL" \
        --recompute_keep_start \
        --output_json "$MIT_BENCH_PG19_RECOMPUTE_JSON"
  fi

  if [[ "$SKIP_EXISTING" == "1" && -f "$MIT_BENCH_WIKI_STREAMING_JSON" && -f "$MIT_BENCH_WIKI_RECOMPUTE_JSON" ]]; then
    echo "Skipping MIT benchmark (WikiText): outputs already exist"
  else
    run "MIT benchmark (WikiText; streaming)" \
      "$PYTHON" mit-streaming-llm/examples/benchmark_streaming.py \
        --model_name_or_path "$MODEL_NAME" \
        --mode streaming \
        --device cuda \
        --start_size "$MAIN_SINK" \
        --recent_size "$MAIN_WINDOW_MIT" \
        --prefix_tokens "$WIKITEXT_TOKENS" \
        --prefill_chunk_size 512 \
        --gen_tokens "$MIT_BENCH_GEN_TOKENS" \
        --data_json "${WIKITEXT_SAMPLE_FILE:-data/wikitext/long_context_4096.json}" \
        --data_text_key text \
        --data_take head \
        --output_json "$MIT_BENCH_WIKI_STREAMING_JSON"

    run "MIT benchmark (WikiText; recompute baseline)" \
      "$PYTHON" mit-streaming-llm/examples/benchmark_streaming.py \
        --model_name_or_path "$MODEL_NAME" \
        --mode recompute \
        --device cuda \
        --start_size "$MAIN_SINK" \
        --recent_size "$MAIN_WINDOW_MIT" \
        --prefix_tokens "$WIKITEXT_TOKENS" \
        --prefill_chunk_size 512 \
        --gen_tokens "$MIT_BENCH_GEN_TOKENS" \
        --data_json "${WIKITEXT_SAMPLE_FILE:-data/wikitext/long_context_4096.json}" \
        --data_text_key text \
        --data_take head \
        --recompute_window_tokens "$CAP_TOTAL" \
        --recompute_keep_start \
        --output_json "$MIT_BENCH_WIKI_RECOMPUTE_JSON"
  fi
fi

if [[ "$RUN_MAIN" == "1" ]]; then
  # StreamingLLM (Start+Recent) baseline (strict prune, i.e. no Lazy Pruning).
  run_repeated "StreamingLLM Start+Recent (PG19; strict prune)" "$START_RECENT_PG19_JSON" \
      --model-name "$MODEL_NAME" \
      --dataset-name pg19 \
      --dataset-config pg19 \
      --split test \
      --max-samples "$MAX_SAMPLES_PG19" \
      --max-eval-tokens "$PG19_TOKENS" \
      --max-length "$CAP_TOTAL" \
      --stride 1022 \
      --n-sink "$OURS_SINK" \
      --window-size "$(calc_window "$CAP_TOTAL" "$OURS_SINK" "0" "$MAIN_OVERLAP" "$MAIN_REFRESH_BUDGET")" \
      --overlap "$MAIN_OVERLAP" \
      --refresh-budget "$MAIN_REFRESH_BUDGET" \
      --refresh-policy "$MAIN_REFRESH_POLICY" \
      --compress-every 1 \
      --cache-slack 0 \
      --max-drop 0 \
      --streaming-mode ours \
      --cache-backend ours \
      --mode streaming \
      --baseline-results "$RESULTS_DIR/pg19_baseline.json"

  run_repeated "StreamingLLM Start+Recent (WikiText; strict prune)" "$START_RECENT_WIKITEXT_JSON" \
      --model-name "$MODEL_NAME" \
      --dataset-name wikitext \
      --dataset-config wikitext-103-v1 \
      --split test \
      --max-samples "$MAX_SAMPLES_WIKI" \
      --max-eval-tokens "$WIKITEXT_TOKENS" \
      --max-length "$CAP_TOTAL" \
      --stride 1022 \
      --n-sink "$OURS_SINK" \
      --window-size "$(calc_window "$CAP_TOTAL" "$OURS_SINK" "0" "$MAIN_OVERLAP" "$MAIN_REFRESH_BUDGET")" \
      --overlap "$MAIN_OVERLAP" \
      --refresh-budget "$MAIN_REFRESH_BUDGET" \
      --refresh-policy "$MAIN_REFRESH_POLICY" \
      --compress-every 1 \
      --cache-slack 0 \
      --max-drop 0 \
      --streaming-mode ours \
      --cache-backend ours \
      --mode streaming \
      --baseline-results "$RESULTS_DIR/wikitext_baseline.json"

  # Ours: Lazy Pruning (amortize cache compaction + RoPE re-alignment).
  run_repeated "Ours (PG19; Lazy Pruning)" "$RESULTS_DIR/pg19_ours.json" \
      --model-name "$MODEL_NAME" \
      --dataset-name pg19 \
      --dataset-config pg19 \
      --split test \
      --max-samples "$MAX_SAMPLES_PG19" \
      --max-eval-tokens "$PG19_TOKENS" \
      --max-length "$CAP_TOTAL" \
      --stride 1022 \
      --n-sink "$OURS_SINK" \
      --window-size "$OURS_WINDOW" \
      --overlap "$MAIN_OVERLAP" \
      --refresh-budget "$MAIN_REFRESH_BUDGET" \
      --refresh-policy "$MAIN_REFRESH_POLICY" \
      --compress-every "$OURS_COMPRESS_EVERY" \
      --cache-slack "$OURS_CACHE_SLACK" \
      --max-drop 0 \
      --streaming-mode ours \
      --cache-backend ours \
      --mode streaming \
      --baseline-results "$RESULTS_DIR/pg19_baseline.json"

  run_repeated "Ours (WikiText; Lazy Pruning)" "$RESULTS_DIR/wikitext_ours.json" \
      --model-name "$MODEL_NAME" \
      --dataset-name wikitext \
      --dataset-config wikitext-103-v1 \
      --split test \
      --max-samples "$MAX_SAMPLES_WIKI" \
      --max-eval-tokens "$WIKITEXT_TOKENS" \
      --max-length "$CAP_TOTAL" \
      --stride 1022 \
      --n-sink "$OURS_SINK" \
      --window-size "$OURS_WINDOW" \
      --overlap "$MAIN_OVERLAP" \
      --refresh-budget "$MAIN_REFRESH_BUDGET" \
      --refresh-policy "$MAIN_REFRESH_POLICY" \
      --compress-every "$OURS_COMPRESS_EVERY" \
      --cache-slack "$OURS_CACHE_SLACK" \
      --max-drop 0 \
      --streaming-mode ours \
      --cache-backend ours \
      --mode streaming \
      --baseline-results "$RESULTS_DIR/wikitext_baseline.json"
fi

###############################################################################
# 2.5) Ablations (paper): Start+Recent vs Lazy Pruning
###############################################################################
if [[ "$RUN_ABLATIONS" == "1" ]]; then
  ABL_DIR="$RESULTS_DIR/ablations"
  mkdir -p "$ABL_DIR"

  # A(-Lazy): strict pruning (ours impl, immediate prune semantics).
  # This is the "w/o Lazy" control: R=1, slack=0, max_drop=0.
  run_repeated "Ablation A(-Lazy) (PG19) w/o Lazy (strict prune)" "$ABL_DIR/pg19_Aneg_lazy_strict.json" \
      --model-name "$MODEL_NAME" \
      --dataset-name pg19 \
      --dataset-config pg19 \
      --split test \
      --max-samples "$MAX_SAMPLES_PG19" \
      --max-eval-tokens "$PG19_TOKENS" \
      --max-length "$CAP_TOTAL" \
      --stride 1022 \
      --n-sink "$OURS_SINK" \
      --window-size "$(calc_window "$CAP_TOTAL" "$OURS_SINK" "0" "$MAIN_OVERLAP" "$MAIN_REFRESH_BUDGET")" \
      --overlap "$MAIN_OVERLAP" \
      --refresh-budget "$MAIN_REFRESH_BUDGET" \
      --refresh-policy "$MAIN_REFRESH_POLICY" \
      --compress-every 1 \
      --cache-slack 0 \
      --max-drop 0 \
      --streaming-mode ours \
      --mode streaming \
      --baseline-results "$RESULTS_DIR/pg19_baseline.json"

  # A1: Lazy only (ours impl, no slack/max_drop).
  run_repeated "Ablation A1 (PG19) +Lazy (ours, no slack/max_drop)" "$ABL_DIR/pg19_A1_lazy.json" \
      --model-name "$MODEL_NAME" \
      --dataset-name pg19 \
      --dataset-config pg19 \
      --split test \
      --max-samples "$MAX_SAMPLES_PG19" \
      --max-eval-tokens "$PG19_TOKENS" \
      --max-length "$CAP_TOTAL" \
      --stride 1022 \
      --n-sink "$OURS_SINK" \
      --window-size "$(calc_window "$CAP_TOTAL" "$OURS_SINK" "0" "$MAIN_OVERLAP" "$MAIN_REFRESH_BUDGET")" \
      --overlap "$MAIN_OVERLAP" \
      --refresh-budget "$MAIN_REFRESH_BUDGET" \
      --refresh-policy "$MAIN_REFRESH_POLICY" \
      --compress-every "$OURS_COMPRESS_EVERY" \
      --cache-slack 0 \
      --max-drop 0 \
      --streaming-mode ours \
      --mode streaming \
      --baseline-results "$RESULTS_DIR/pg19_baseline.json"

  if [[ "$RUN_QUALITY_HEURISTICS" == "1" ]]; then
    # Optional quality heuristics (Appendix): Slack / Max_Drop.
    run_repeated "Ablation A2 (PG19) +Lazy+Slack (exploratory)" "$ABL_DIR/pg19_A2_lazy_slack.json" \
        --model-name "$MODEL_NAME" \
        --dataset-name pg19 \
        --dataset-config pg19 \
        --split test \
        --max-samples "$MAX_SAMPLES_PG19" \
        --max-eval-tokens "$PG19_TOKENS" \
        --max-length "$CAP_TOTAL" \
        --stride 1022 \
        --n-sink "$OURS_SINK" \
        --window-size "$(calc_window "$CAP_TOTAL" "$OURS_SINK" "$OURS_CACHE_SLACK" "$MAIN_OVERLAP" "$MAIN_REFRESH_BUDGET")" \
        --overlap "$MAIN_OVERLAP" \
        --refresh-budget "$MAIN_REFRESH_BUDGET" \
        --refresh-policy "$MAIN_REFRESH_POLICY" \
        --compress-every "$OURS_COMPRESS_EVERY" \
        --cache-slack "$OURS_CACHE_SLACK" \
        --max-drop 0 \
        --streaming-mode ours \
        --mode streaming \
        --baseline-results "$RESULTS_DIR/pg19_baseline.json"

    run_repeated "Ablation A3 (PG19) +Lazy+Slack+Max_Drop (exploratory)" "$ABL_DIR/pg19_A3_full.json" \
        --model-name "$MODEL_NAME" \
        --dataset-name pg19 \
        --dataset-config pg19 \
        --split test \
        --max-samples "$MAX_SAMPLES_PG19" \
        --max-eval-tokens "$PG19_TOKENS" \
        --max-length "$CAP_TOTAL" \
        --stride 1022 \
        --n-sink "$OURS_SINK" \
        --window-size "$(calc_window "$CAP_TOTAL" "$OURS_SINK" "$OURS_CACHE_SLACK" "$MAIN_OVERLAP" "$MAIN_REFRESH_BUDGET")" \
        --overlap "$MAIN_OVERLAP" \
        --refresh-budget "$MAIN_REFRESH_BUDGET" \
        --refresh-policy "$MAIN_REFRESH_POLICY" \
        --compress-every "$OURS_COMPRESS_EVERY" \
        --cache-slack "$OURS_CACHE_SLACK" \
        --max-drop "$OURS_MAX_DROP" \
        --streaming-mode ours \
        --mode streaming \
        --baseline-results "$RESULTS_DIR/pg19_baseline.json"
  fi
fi

###############################################################################
# 2.6) Sweeps (R / sigma / delta) for PG19
###############################################################################
if [[ "$RUN_SWEEPS" == "1" ]]; then
  SWEEP_DIR="$RESULTS_DIR/sweeps"
  mkdir -p "$SWEEP_DIR/R" "$SWEEP_DIR/sigma" "$SWEEP_DIR/delta"

  for R in $SWEEP_R_VALUES; do
    run_repeated "Sweep R=$R PG19" "$SWEEP_DIR/R/pg19_R${R}.json" \
        --model-name "$MODEL_NAME" \
        --dataset-name pg19 \
        --dataset-config pg19 \
        --split test \
        --max-samples "$MAX_SAMPLES_PG19" \
        --max-eval-tokens "$PG19_TOKENS" \
        --max-length "$CAP_TOTAL" \
        --stride 1022 \
        --n-sink "$OURS_SINK" \
        --window-size "$(calc_window "$CAP_TOTAL" "$OURS_SINK" "$OURS_CACHE_SLACK" "$MAIN_OVERLAP" "$MAIN_REFRESH_BUDGET")" \
        --overlap "$MAIN_OVERLAP" \
        --refresh-budget "$MAIN_REFRESH_BUDGET" \
        --refresh-policy "$MAIN_REFRESH_POLICY" \
        --compress-every "$R" \
        --cache-slack "$OURS_CACHE_SLACK" \
        --max-drop "$OURS_MAX_DROP" \
        --streaming-mode ours \
        --mode streaming \
        --baseline-results "$RESULTS_DIR/pg19_baseline.json"
  done

  if [[ "$RUN_QUALITY_HEURISTICS" == "1" ]]; then
    for SIGMA in $SWEEP_SIGMA_VALUES; do
      run_repeated "Sweep sigma=$SIGMA PG19" "$SWEEP_DIR/sigma/pg19_sigma${SIGMA}.json" \
          --model-name "$MODEL_NAME" \
          --dataset-name pg19 \
          --dataset-config pg19 \
          --split test \
          --max-samples "$MAX_SAMPLES_PG19" \
          --max-eval-tokens "$PG19_TOKENS" \
          --max-length "$CAP_TOTAL" \
          --stride 1022 \
          --n-sink "$OURS_SINK" \
          --window-size "$(calc_window "$CAP_TOTAL" "$OURS_SINK" "$SIGMA" "$MAIN_OVERLAP" "$MAIN_REFRESH_BUDGET")" \
          --overlap "$MAIN_OVERLAP" \
          --refresh-budget "$MAIN_REFRESH_BUDGET" \
          --refresh-policy "$MAIN_REFRESH_POLICY" \
          --compress-every "$OURS_COMPRESS_EVERY" \
          --cache-slack "$SIGMA" \
          --max-drop "$OURS_MAX_DROP" \
          --streaming-mode ours \
          --mode streaming \
          --baseline-results "$RESULTS_DIR/pg19_baseline.json"
    done

    for DELTA in $SWEEP_DELTA_VALUES; do
      run_repeated "Sweep delta=$DELTA PG19" "$SWEEP_DIR/delta/pg19_delta${DELTA}.json" \
          --model-name "$MODEL_NAME" \
          --dataset-name pg19 \
          --dataset-config pg19 \
          --split test \
          --max-samples "$MAX_SAMPLES_PG19" \
          --max-eval-tokens "$PG19_TOKENS" \
          --max-length "$CAP_TOTAL" \
          --stride 1022 \
          --n-sink "$OURS_SINK" \
          --window-size "$(calc_window "$CAP_TOTAL" "$OURS_SINK" "$OURS_CACHE_SLACK" "$MAIN_OVERLAP" "$MAIN_REFRESH_BUDGET")" \
          --overlap "$MAIN_OVERLAP" \
          --refresh-budget "$MAIN_REFRESH_BUDGET" \
          --refresh-policy "$MAIN_REFRESH_POLICY" \
          --compress-every "$OURS_COMPRESS_EVERY" \
          --cache-slack "$OURS_CACHE_SLACK" \
          --max-drop "$DELTA" \
          --streaming-mode ours \
          --mode streaming \
          --baseline-results "$RESULTS_DIR/pg19_baseline.json"
    done
  fi
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
