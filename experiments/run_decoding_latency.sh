#!/bin/bash
# 运行 Per-Token Decoding Latency 评估

echo "=========================================="
echo "Per-Token Decoding Latency 评估"
echo "=========================================="
echo ""

ENV_FILE=".env"
if [ -f "$ENV_FILE" ]; then
  set -o allexport
  source "$ENV_FILE"
  set +o allexport
fi

# 设置参数
MODEL="${MODEL_NAME:-EleutherAI/pythia-2.8b}"
DTYPE="float16"
CACHE_SIZE=1024
N_SINK=4
PROMPT_LENGTH=512
NUM_TOKENS=2000
WARMUP_TOKENS=200
NUM_RUNS=3

echo "配置:"
echo "  模型: $MODEL"
echo "  数据类型: $DTYPE"
echo "  Cache Size: $CACHE_SIZE"
echo "  N_sink: $N_SINK"
echo "  Prompt Length: $PROMPT_LENGTH"
echo "  Num Tokens: $NUM_TOKENS"
echo "  Warmup Tokens: $WARMUP_TOKENS"
echo "  Num Runs: $NUM_RUNS"
echo ""

# 运行评估
python experiments/eval_decoding_latency.py \
  --model-name "$MODEL" \
  --dtype "$DTYPE" \
  --cache-size $CACHE_SIZE \
  --n-sink $N_SINK \
  --prompt-length $PROMPT_LENGTH \
  --num-tokens $NUM_TOKENS \
  --warmup-tokens $WARMUP_TOKENS \
  --num-runs $NUM_RUNS \
  --output results/decoding_latency_cache${CACHE_SIZE}.json

echo ""
echo "=========================================="
echo "评估完成!"
echo "=========================================="
