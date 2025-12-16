#!/bin/bash
set -euo pipefail

# 读取 .env, 提供可共享配置
ENV_FILE=".env"
if [ -f "$ENV_FILE" ]; then
  set -o allexport
  source "$ENV_FILE"
  set +o allexport
fi

# Set offline Hugging Face cache (matches other scripts)
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

PYTHON="${PYTHON_BIN:-kvpress/.venv/bin/python}"
if [ ! -f "$PYTHON" ]; then
  echo "Python interpreter not found: $PYTHON"
  exit 1
fi

DATASET_NAME=${DATASET_NAME:-wikitext}
DATASET_CONFIG=${DATASET_CONFIG:-wikitext-103-v1}
SPLIT=${SPLIT:-test}
MAX_SAMPLES=${MAX_SAMPLES:-64}
MAX_EVAL_TOKENS=${MAX_EVAL_TOKENS:-4096}
N_SINK=${N_SINK:-4}
WINDOW_SIZE=${WINDOW_SIZE:-2048}
OUTPUT=${OUTPUT:-results/fixed_eval/${DATASET_NAME}_kvpress.json}

"$PYTHON" experiments/run_decode_perplexity.py \
  --method kvpress \
  --model-name "${MODEL_NAME:-EleutherAI/pythia-2.8b}" \
  --dataset-name "$DATASET_NAME" \
  --dataset-config "$DATASET_CONFIG" \
  --split "$SPLIT" \
  --max-samples "$MAX_SAMPLES" \
  --max-eval-tokens "$MAX_EVAL_TOKENS" \
  --n-sink "$N_SINK" \
  --window-size "$WINDOW_SIZE" \
  --output "$OUTPUT"
