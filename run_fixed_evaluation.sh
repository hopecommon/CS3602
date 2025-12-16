#!/bin/bash
set -euo pipefail

# 修复后的评测脚本 - 使用 decode-loop 逐 token 解码（可比）
# baseline / ours / mit / kvpress 都走同一条 decode-loop 评测逻辑。

# 读取 .env 自定义配置
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
RESULT_ROOT="${RESULT_ROOT:-results/fixed_eval}"
mkdir -p "$RESULT_ROOT"

# Experiment parameters
N_SINK="${N_SINK:-4}"
WINDOW_SIZE="${WINDOW_SIZE:-2048}"
COMPRESS_EVERY="${COMPRESS_EVERY:-4}"
WIKITEXT_MAX_TOKENS="${WIKITEXT_MAX_TOKENS:-4096}"
PG19_20K_MAX_TOKENS="${PG19_20K_MAX_TOKENS:-20000}"

run_experiment() {
  local name="$1"
  shift
  echo "=== $name ==="
  printf '%q ' "$@"
  echo
  "$@"
  echo
}

# 数据集配置
datasets=("wikitext" "pg19_20k")
declare -A configs
declare -A max_tokens
declare -A tags

configs["wikitext"]="wikitext-103-v1"
max_tokens["wikitext"]=$WIKITEXT_MAX_TOKENS
tags["wikitext"]="wikitext"

configs["pg19_20k"]=""
max_tokens["pg19_20k"]=$PG19_20K_MAX_TOKENS
tags["pg19_20k"]="pg19_20k"

# 固定评测默认跑 decode-loop 可比的 baseline/ours/kvpress。
# 如需额外方法（例如内部 mit-style slicing），可通过环境变量覆盖：
#   FIXED_EVAL_METHODS="baseline ours kvpress mit"
IFS=' ' read -r -a methods <<< "${FIXED_EVAL_METHODS:-baseline ours kvpress}"

for dataset in "${datasets[@]}"; do
  config="${configs[$dataset]}"
  max_tok="${max_tokens[$dataset]}"
  dataset_tag="${tags[$dataset]}"
  
  # 确定实际的数据集名称
  actual_dataset="$dataset"
  if [[ "$dataset" == pg19_* ]]; then
    actual_dataset="pg19"
  fi
  
  echo "=========================================="
  echo "Dataset: $dataset_tag"
  echo "  Actual dataset: $actual_dataset"
  echo "  Config: $config"
  echo "  Max tokens: $max_tok"
  echo "  N_sink: $N_SINK"
  echo "  Window size: $WINDOW_SIZE"
  echo "=========================================="

  for method in "${methods[@]}"; do
    output_file="$RESULT_ROOT/${dataset_tag}_${method}.json"
    if [ -f "$output_file" ]; then
      if [[ "$method" == "kvpress" ]] && ! grep -q '"total_time"' "$output_file" 2>/dev/null; then
        legacy_file="${output_file%.json}_legacy.json"
        echo "Found legacy KVPress result (non-decode-loop). Renaming to $legacy_file and re-running..."
        mv "$output_file" "$legacy_file"
      else
        echo "Skipping $output_file (already exists)"
        continue
      fi
    fi

    cmd=(
      "$PYTHON" experiments/run_decode_perplexity.py
      --model-name "$MODEL_NAME"
      --method "$method"
      --dataset-name "$actual_dataset"
      --dataset-config "$config"
      --max-eval-tokens "$max_tok"
      --n-sink "$N_SINK"
      --window-size "$WINDOW_SIZE"
      --compress-every "$COMPRESS_EVERY"
      --output "$output_file"
    )

    run_experiment "$dataset_tag $method" "${cmd[@]}"
  done
done

echo "=========================================="
echo "Generating summary table..."
"$PYTHON" experiments/summarize_fixed_eval_results.py --results-dir "$RESULT_ROOT" --output "$RESULT_ROOT/summary.md" || true

echo "Generating plots..."
"$PYTHON" experiments/plot_fixed_eval_results.py || true

if [[ "${RUN_MIT_OFFICIAL_BENCHMARK:-0}" == "1" ]]; then
  echo "=========================================="
  echo "Running MIT official throughput/VRAM benchmark - non-PPL..."
  # MIT 官方 repo 对 transformers/huggingface-hub 版本较敏感，建议用独立环境的 python 运行。
  MIT_BENCH_PYTHON="${MIT_BENCH_PYTHON:-$PYTHON}"
  MIT_BENCH_MODEL_PATH="${MIT_BENCH_MODEL_PATH:-$MODEL_NAME}"
  MIT_BENCH_DATA_JSON="${MIT_BENCH_DATA_JSON:-${PG19_SAMPLE_FILE:-data/pg19/long_context_20000.json}}"
  mkdir -p results/mit_official
  "$PYTHON" experiments/run_mit_official_benchmark.py \
    --python "$MIT_BENCH_PYTHON" \
    --model-name-or-path "$MIT_BENCH_MODEL_PATH" \
    --data-json "$MIT_BENCH_DATA_JSON" \
    --prefix-tokens 20000 \
    --gen-tokens 512 \
    --start-size "$N_SINK" \
    --recent-size "$WINDOW_SIZE" \
    --output results/mit_official/pg19_20k_benchmark.json || true
fi

echo "=========================================="
echo "All evaluations complete!"
echo "Results saved to: $RESULT_ROOT"
echo "=========================================="
