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
RESULT_ROOT="${RESULT_ROOT:-results/comprehensive}"
mkdir -p "$RESULT_ROOT"

run_experiment() {
  local name="$1"
  local cmd="$2"
  echo "=== $name ==="
  echo "$cmd"
  eval "$cmd"
  echo
}

datasets=("wikitext" "pg19")
declare -A configs
configs["wikitext"]="wikitext-103-v1"
configs["pg19"]=""
methods=("baseline" "ours" "mit" "kvpress")
eval_types=("chunked" "decode_loop" "kvpress_official")

for dataset in "${datasets[@]}"; do
  config="${configs[$dataset]}"
  dataset_tag="${dataset}"
  if [ "$dataset" == "pg19" ]; then
    dataset_tag="pg19"
  fi

  baseline_file="results/streaming_llm/${dataset_tag}_baseline.json"
  if [ ! -f "$baseline_file" ]; then
    run_experiment \
      "$dataset baseline chunked" \
      "$PYTHON experiments/eval_streaming_llm.py \
        --dataset-name $dataset \
        --dataset-config \"$config\" \
        --mode baseline \
        --output $baseline_file"
  fi

  for method in "${methods[@]}"; do
    for eval in "${eval_types[@]}"; do
      case "$eval" in
        "chunked")
          case "$method" in
            "baseline")
              cmd="$PYTHON experiments/eval_streaming_llm.py \
                --dataset-name $dataset \
                --dataset-config \"$config\" \
                --mode baseline \
                --output $RESULT_ROOT/${dataset_tag}_baseline_chunked.json"
              ;;
            "ours")
              cmd="$PYTHON experiments/eval_streaming_llm.py \
                --dataset-name $dataset \
                --dataset-config \"$config\" \
                --mode streaming \
                --streaming-mode ours \
                --baseline-results $baseline_file \
                --output $RESULT_ROOT/${dataset_tag}_ours_chunked.json"
              ;;
            "mit")
              cmd="$PYTHON experiments/eval_streaming_llm.py \
                --dataset-name $dataset \
                --dataset-config \"$config\" \
                --mode streaming \
                --streaming-mode mit \
                --baseline-results $baseline_file \
                --output $RESULT_ROOT/${dataset_tag}_mit_chunked.json"
              ;;
            "kvpress")
              cmd="$PYTHON experiments/eval_kvpress.py \
                --dataset-name $dataset \
                --dataset-config \"$config\" \
                --output $RESULT_ROOT/${dataset_tag}_kvpress_chunked.json"
              ;;
          esac
          ;;
        "decode_loop")
          cmd="$PYTHON experiments/run_decode_perplexity.py \
            --method $method \
            --dataset-name $dataset \
            --dataset-config \"$config\" \
            --output $RESULT_ROOT/${dataset_tag}_${method}_decode_loop.json"
          ;;
        "kvpress_official")
          cmd="$PYTHON experiments/eval_kvpress.py \
            --dataset-name $dataset \
            --dataset-config \"$config\" \
            --output $RESULT_ROOT/${dataset_tag}_${method}_kvpress_official.json"
          ;;
      esac

      run_experiment "$dataset $method $eval" "$cmd"
    done
  done
done
