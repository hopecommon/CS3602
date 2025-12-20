#!/bin/bash
set -euo pipefail

# Speculative decoding comparison script (normal vs speculative, baseline vs streaming)

# Load .env if present
ENV_FILE=".env"
if [ -f "$ENV_FILE" ]; then
  set -o allexport
  source "$ENV_FILE"
  set +o allexport
fi

export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

PYTHON="${PYTHON_BIN:-kvpress/.venv/bin/python}"
if command -v realpath >/dev/null 2>&1; then
  PYTHON="$(realpath "$PYTHON" 2>/dev/null || echo "$PYTHON")"
fi

MODEL_NAME="${MODEL_NAME:-EleutherAI/pythia-2.8b}"
DRAFT_MODEL_NAME="${DRAFT_MODEL_NAME:-EleutherAI/pythia-70m}"
RESULT_ROOT="${RESULT_ROOT:-results/spec_decode}"
mkdir -p "$RESULT_ROOT"
ONLY_SPEC="${ONLY_SPEC:-0}"

# Experiment parameters
N_SINK="${N_SINK:-4}"
CACHE_SIZE="${WINDOW_SIZE:-2048}"
PROMPT_LENGTH="${PROMPT_LENGTH:-512}"
NUM_TOKENS="${NUM_TOKENS:-2000}"
WARMUP_TOKENS="${WARMUP_TOKENS:-200}"
NUM_RUNS="${NUM_RUNS:-3}"
DRAFT_K="${DRAFT_K:-8}"
TEMPERATURE="${TEMPERATURE:-0.0}"
SPECULATIVE_MODE="${SPECULATIVE_MODE:-exact}"
PREFILL_FILE="${PREFILL_FILE:-}"
PREFILL_TOKENS="${PREFILL_TOKENS:-0}"
PREFILL_CHUNK="${PREFILL_CHUNK:-256}"
PREFILL_FIELD="${PREFILL_FIELD:-text}"

run_experiment() {
  local name="$1"
  shift
  echo "=== $name ==="
  printf '%q ' "$@"
  echo
  "$@"
  echo
}

run_if_missing() {
  local name="$1"
  local output="$2"
  shift 2
  if [ -f "$output" ]; then
    echo "=== $name ==="
    echo "Skipping $output (already exists)"
    echo
    return 0
  fi
  run_experiment "$name" "$@"
}

BASE_NORMAL="$RESULT_ROOT/base_normal.json"
BASE_SPEC="$RESULT_ROOT/base_speculative.json"
STREAM_NORMAL="$RESULT_ROOT/stream_normal.json"
STREAM_SPEC="$RESULT_ROOT/stream_speculative.json"

if [ "$ONLY_SPEC" != "1" ]; then
  run_if_missing "baseline normal" "$BASE_NORMAL" \
    "$PYTHON" experiments/eval_decoding_latency.py \
    --model-name "$MODEL_NAME" \
    --mode baseline \
    --decoder normal \
    --cache-size "$CACHE_SIZE" \
    --n-sink "$N_SINK" \
    --prompt-length "$PROMPT_LENGTH" \
    --num-tokens "$NUM_TOKENS" \
    --warmup-tokens "$WARMUP_TOKENS" \
    --num-runs "$NUM_RUNS" \
    ${PREFILL_FILE:+--prefill-file "$PREFILL_FILE"} \
    ${PREFILL_FILE:+--prefill-field "$PREFILL_FIELD"} \
    --prefill-tokens "$PREFILL_TOKENS" \
    --prefill-chunk "$PREFILL_CHUNK" \
    --output "$BASE_NORMAL"
fi

run_if_missing "baseline speculative" "$BASE_SPEC" \
  "$PYTHON" experiments/eval_decoding_latency.py \
  --model-name "$MODEL_NAME" \
  --draft-model-name "$DRAFT_MODEL_NAME" \
  --mode baseline \
  --decoder speculative \
  --speculative-mode "$SPECULATIVE_MODE" \
  --draft-k "$DRAFT_K" \
  --temperature "$TEMPERATURE" \
  --cache-size "$CACHE_SIZE" \
  --n-sink "$N_SINK" \
  --prompt-length "$PROMPT_LENGTH" \
  --num-tokens "$NUM_TOKENS" \
  --warmup-tokens "$WARMUP_TOKENS" \
  --num-runs "$NUM_RUNS" \
  ${PREFILL_FILE:+--prefill-file "$PREFILL_FILE"} \
  ${PREFILL_FILE:+--prefill-field "$PREFILL_FIELD"} \
  --prefill-tokens "$PREFILL_TOKENS" \
  --prefill-chunk "$PREFILL_CHUNK" \
  --output "$BASE_SPEC"

if [ "$ONLY_SPEC" != "1" ]; then
  run_if_missing "streaming normal" "$STREAM_NORMAL" \
    "$PYTHON" experiments/eval_decoding_latency.py \
    --model-name "$MODEL_NAME" \
    --mode streaming \
    --decoder normal \
    --cache-size "$CACHE_SIZE" \
    --n-sink "$N_SINK" \
    --prompt-length "$PROMPT_LENGTH" \
    --num-tokens "$NUM_TOKENS" \
    --warmup-tokens "$WARMUP_TOKENS" \
    --num-runs "$NUM_RUNS" \
    ${PREFILL_FILE:+--prefill-file "$PREFILL_FILE"} \
    ${PREFILL_FILE:+--prefill-field "$PREFILL_FIELD"} \
    --prefill-tokens "$PREFILL_TOKENS" \
    --prefill-chunk "$PREFILL_CHUNK" \
    --baseline-results "$BASE_NORMAL" \
    --output "$STREAM_NORMAL"
fi

run_if_missing "streaming speculative" "$STREAM_SPEC" \
  "$PYTHON" experiments/eval_decoding_latency.py \
  --model-name "$MODEL_NAME" \
  --draft-model-name "$DRAFT_MODEL_NAME" \
  --mode streaming \
  --decoder speculative \
  --speculative-mode "$SPECULATIVE_MODE" \
  --draft-k "$DRAFT_K" \
  --temperature "$TEMPERATURE" \
  --cache-size "$CACHE_SIZE" \
  --n-sink "$N_SINK" \
  --prompt-length "$PROMPT_LENGTH" \
  --num-tokens "$NUM_TOKENS" \
  --warmup-tokens "$WARMUP_TOKENS" \
  --num-runs "$NUM_RUNS" \
  ${PREFILL_FILE:+--prefill-file "$PREFILL_FILE"} \
  ${PREFILL_FILE:+--prefill-field "$PREFILL_FIELD"} \
  --prefill-tokens "$PREFILL_TOKENS" \
  --prefill-chunk "$PREFILL_CHUNK" \
  --baseline-results "$BASE_SPEC" \
  --output "$STREAM_SPEC"

echo "=========================================="
echo "All comparisons complete!"
echo "Results saved to: $RESULT_ROOT"
echo "=========================================="
