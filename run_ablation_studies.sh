#!/bin/bash
################################################################################
# StreamingLLM 消融实验脚本
#
# 功能:
#   1. Window Size 消融实验 (测试不同window_size对性能的影响)
#   2. N_sink 消融实验 (测试不同n_sink对性能的影响)
#   3. Cache Size Sweep (测试不同cache配置)
#
# 使用说明:
#   chmod +x run_ablation_studies.sh
#   ./run_ablation_studies.sh
################################################################################

set -euo pipefail

# 读取 .env 配置
ENV_FILE=".env"
if [ -f "$ENV_FILE" ]; then
  set -o allexport
  source "$ENV_FILE"
  set +o allexport
fi

# 环境变量
export HF_HOME="${HF_HOME:-/data2/jflin/CS3602/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

PYTHON="${PYTHON_BIN:-kvpress/.venv/bin/python}"
MODEL_NAME="${MODEL_NAME:-EleutherAI/pythia-2.8b}"
RESULT_DIR="results/ablation"
mkdir -p "$RESULT_DIR"

# 实验参数
WIKITEXT_MAX_TOKENS="${WIKITEXT_MAX_TOKENS:-4096}"
WIKITEXT_MAX_SAMPLES="${WIKITEXT_MAX_SAMPLES:-64}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
STRIDE="${STRIDE:-1024}"

# 颜色
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

run_experiment() {
  local name="$1"
  local cmd="$2"
  echo ""
  echo -e "${BLUE}========================================${NC}"
  echo -e "${BLUE}$name${NC}"
  echo -e "${BLUE}========================================${NC}"
  echo "命令: $cmd"
  echo "开始时间: $(date)"
  
  local start=$(date +%s)
  eval "$cmd"
  local end=$(date +%s)
  local duration=$((end - start))
  
  echo -e "${GREEN}✓ 完成${NC} (耗时: ${duration}s)"
}

################################################################################
# 1. Window Size 消融实验
################################################################################

echo -e "${YELLOW}=== 消融实验 1: Window Size ===${NC}"
echo "测试不同window_size对StreamingLLM性能的影响"
echo "数据集: WikiText-103 (${WIKITEXT_MAX_TOKENS} tokens)"
echo ""

run_experiment \
  "Window Size 消融实验" \
  "$PYTHON experiments/ablation_study.py \
    --model-name \"$MODEL_NAME\" \
    --dataset-name wikitext \
    --dataset-config wikitext-103-v1 \
    --max-samples $WIKITEXT_MAX_SAMPLES \
    --max-eval-tokens $WIKITEXT_MAX_TOKENS \
    --max-length $MAX_LENGTH \
    --stride $STRIDE \
    --ablation-type window_size \
    --output $RESULT_DIR/ablation_window_size.json"

################################################################################
# 2. N_sink 消融实验
################################################################################

echo ""
echo -e "${YELLOW}=== 消融实验 2: N_sink ===${NC}"
echo "测试不同n_sink对StreamingLLM性能的影响"
echo "数据集: WikiText-103 (${WIKITEXT_MAX_TOKENS} tokens)"
echo ""

run_experiment \
  "N_sink 消融实验" \
  "$PYTHON experiments/ablation_study.py \
    --model-name \"$MODEL_NAME\" \
    --dataset-name wikitext \
    --dataset-config wikitext-103-v1 \
    --max-samples $WIKITEXT_MAX_SAMPLES \
    --max-eval-tokens $WIKITEXT_MAX_TOKENS \
    --max-length $MAX_LENGTH \
    --stride $STRIDE \
    --ablation-type n_sink \
    --output $RESULT_DIR/ablation_n_sink.json"

################################################################################
# 3. Cache Size Sweep (Decoding Latency)
################################################################################

echo ""
echo -e "${YELLOW}=== 消融实验 3: Cache Size Sweep (Decoding Latency) ===${NC}"
echo "测试不同cache size对decoding latency的影响"
echo "注意: 这是latency测试，不是PPL评估"
echo ""

run_experiment \
  "Cache Size Sweep" \
  "$PYTHON experiments/sweep_cache_sizes.py \
    --model-name \"$MODEL_NAME\" \
    --cache-sizes 256 512 1024 2048 4096 \
    --n-sink 4 \
    --num-tokens 2000 \
    --num-runs 3 \
    --output-dir $RESULT_DIR/cache_size_sweep"

################################################################################
# 总结
################################################################################

echo ""
echo "========================================"
echo -e "${GREEN}✓ 所有消融实验完成!${NC}"
echo "========================================"
echo ""
echo "结果文件:"
ls -lh "$RESULT_DIR"/*.json
echo ""
echo "可以使用以下命令查看结果:"
echo "  cat $RESULT_DIR/ablation_window_size.json | jq ."
echo "  cat $RESULT_DIR/ablation_n_sink.json | jq ."
echo "  cat $RESULT_DIR/cache_size_sweep.json | jq ."
