#!/bin/bash
set -euo pipefail

# 修复后的评测脚本 - 只使用正确的评测方法
# 1. decode_loop: 逐 token 解码，与 MIT 标准一致
# 2. kvpress_official: KVPress 的正确用法

# 读取 .env 自定义配置
ENV_FILE=".env"
if [ -f "$ENV_FILE" ]; then
  set -o allexport
  source "$ENV_FILE"
  set +o allexport
fi

export HF_HOME="${HF_HOME:-/data2/jflin/CS3602/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

PYTHON="${PYTHON_BIN:-kvpress/.venv/bin/python}"
MODEL_NAME="${MODEL_NAME:-EleutherAI/pythia-2.8b}"
RESULT_ROOT="${RESULT_ROOT:-results/fixed_eval}"
mkdir -p "$RESULT_ROOT"

# Experiment parameters
N_SINK="${N_SINK:-4}"
WINDOW_SIZE="${WINDOW_SIZE:-1024}"
WIKITEXT_MAX_TOKENS="${WIKITEXT_MAX_TOKENS:-4096}"
PG19_20K_MAX_TOKENS="${PG19_20K_MAX_TOKENS:-20000}"

run_experiment() {
  local name="$1"
  local cmd="$2"
  echo "=== $name ==="
  echo "$cmd"
  eval "$cmd"
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

# 只使用正确的方法
methods=("baseline" "ours" "mit" "kvpress")

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
    if [[ "$method" == "kvpress" ]]; then
      # KVPress 使用 official 实现
      output_file="$RESULT_ROOT/${dataset_tag}_kvpress.json"
      if [ -f "$output_file" ]; then
        echo "Skipping $output_file (already exists)"
        continue
      fi
      
      cmd="$PYTHON experiments/eval_kvpress.py \
        --model-name \"$MODEL_NAME\" \
        --dataset-name $actual_dataset \
        --dataset-config \"$config\" \
        --max-eval-tokens $max_tok \
        --n-sink $N_SINK \
        --window-size $WINDOW_SIZE \
        --output $output_file"
      
      run_experiment "$dataset_tag kvpress" "$cmd"
    else
      # 其他方法使用 decode_loop
      output_file="$RESULT_ROOT/${dataset_tag}_${method}.json"
      if [ -f "$output_file" ]; then
        echo "Skipping $output_file (already exists)"
        continue
      fi
      
      cmd="$PYTHON experiments/run_decode_perplexity.py \
        --model-name \"$MODEL_NAME\" \
        --method $method \
        --dataset-name $actual_dataset \
        --dataset-config \"$config\" \
        --max-eval-tokens $max_tok \
        --n-sink $N_SINK \
        --window-size $WINDOW_SIZE \
        --output $output_file"
      
      run_experiment "$dataset_tag $method" "$cmd"
    fi
  done
done

echo "=========================================="
echo "All evaluations complete!"
echo "Results saved to: $RESULT_ROOT"
echo "=========================================="
