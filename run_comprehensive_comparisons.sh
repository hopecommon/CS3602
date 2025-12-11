#!/bin/bash
set -euo pipefail

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
RESULT_ROOT="${RESULT_ROOT:-results/comprehensive}"
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

# 数据集配置: 名称 -> (config, max_tokens, tag)
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

# 解码评测方法：decode_loop 仅支持 baseline/ours/mit；kvpress 走 eval_kvpress 路径
methods=("baseline" "ours" "mit" "kvpress")
eval_types=("chunked" "decode_loop" "kvpress_official")

for dataset in "${datasets[@]}"; do
  config="${configs[$dataset]}"
  max_tok="${max_tokens[$dataset]}"
  dataset_tag="${tags[$dataset]}"
  
  # 确定实际的数据集名称 (pg19_20k/pg19_50k -> pg19)
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

  baseline_file="results/streaming_llm/${dataset_tag}_baseline.json"
  if [ ! -f "$baseline_file" ]; then
    run_experiment \
      "$dataset_tag baseline chunked" \
      "$PYTHON experiments/eval_streaming_llm.py \
        --model-name \"$MODEL_NAME\" \
        --dataset-name $actual_dataset \
        --dataset-config \"$config\" \
        --mode baseline \
        --max-eval-tokens $max_tok \
        --n-sink $N_SINK \
        --window-size $WINDOW_SIZE \
        --output $baseline_file"
  fi

  for method in "${methods[@]}"; do
    for eval in "${eval_types[@]}"; do
      # 仅在支持的组合上运行
      if [[ "$eval" == "decode_loop" && "$method" == "kvpress" ]]; then
        continue
      fi
      if [[ "$eval" == "kvpress_official" && "$method" != "kvpress" ]]; then
        continue
      fi

      case "$eval" in
        "chunked")
          case "$method" in
            "baseline")
              cmd="$PYTHON experiments/eval_streaming_llm.py \
                --model-name \"$MODEL_NAME\" \
                --dataset-name $actual_dataset \
                --dataset-config \"$config\" \
                --mode baseline \
                --max-eval-tokens $max_tok \
                --n-sink $N_SINK \
                --window-size $WINDOW_SIZE \
                --output $RESULT_ROOT/${dataset_tag}_baseline_chunked.json"
              ;;
            "ours")
              cmd="$PYTHON experiments/eval_streaming_llm.py \
                --model-name \"$MODEL_NAME\" \
                --dataset-name $actual_dataset \
                --dataset-config \"$config\" \
                --mode streaming \
                --streaming-mode ours \
                --max-eval-tokens $max_tok \
                --n-sink $N_SINK \
                --window-size $WINDOW_SIZE \
                --baseline-results $baseline_file \
                --output $RESULT_ROOT/${dataset_tag}_ours_chunked.json"
              ;;
            "mit")
              cmd="$PYTHON experiments/eval_streaming_llm.py \
                --model-name \"$MODEL_NAME\" \
                --dataset-name $actual_dataset \
                --dataset-config \"$config\" \
                --mode streaming \
                --streaming-mode mit \
                --max-eval-tokens $max_tok \
                --n-sink $N_SINK \
                --window-size $WINDOW_SIZE \
                --baseline-results $baseline_file \
                --output $RESULT_ROOT/${dataset_tag}_mit_chunked.json"
              ;;
            "kvpress")
              cmd="$PYTHON experiments/eval_kvpress.py \
                --model-name \"$MODEL_NAME\" \
                --dataset-name $actual_dataset \
                --dataset-config \"$config\" \
                --max-eval-tokens $max_tok \
                --n-sink $N_SINK \
                --window-size $WINDOW_SIZE \
                --output $RESULT_ROOT/${dataset_tag}_kvpress_chunked.json"
              ;;
          esac
          ;;
        "decode_loop")
          cmd="$PYTHON experiments/run_decode_perplexity.py \
            --model-name \"$MODEL_NAME\" \
            --method $method \
            --dataset-name $actual_dataset \
            --dataset-config \"$config\" \
            --max-eval-tokens $max_tok \
            --n-sink $N_SINK \
            --window-size $WINDOW_SIZE \
            --output $RESULT_ROOT/${dataset_tag}_${method}_decode_loop.json"
          ;;
        "kvpress_official")
          cmd="$PYTHON experiments/eval_kvpress.py \
            --model-name \"$MODEL_NAME\" \
            --dataset-name $actual_dataset \
            --dataset-config \"$config\" \
            --max-eval-tokens $max_tok \
            --n-sink $N_SINK \
            --window-size $WINDOW_SIZE \
            --output $RESULT_ROOT/${dataset_tag}_${method}_kvpress_official.json"
          ;;
      esac

      run_experiment "$dataset_tag $method $eval" "$cmd"
    done
  done
done
