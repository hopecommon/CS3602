#!/bin/bash
################################################################################
# StreamingLLM 完整实验一键运行脚本
#
# 使用说明:
#   chmod +x run_everything.sh
#   ./run_everything.sh
#
# 功能:
#   1. Baseline 实验 (WikiText-103 & PG19)
#   2. 我们的 StreamingLLM 实验 (WikiText-103 & PG19)
#   3. kvpress 官方库对比实验 (WikiText-103 & PG19)
#   4. 消融实验 (Window Size & N_sink)
#   5. 生成可视化图表
#
# 注意:
#   - 使用 kvpress/.venv/bin/python 作为 Python 解释器
#   - 每个实验失败不会中断整个流程
#   - 所有结果保存在 results/ 目录
################################################################################

set +e  # 允许单个实验失败而不中断整个流程

# 读取 .env 文件, 方便自定义参数
ENV_FILE=".env"
if [ -f "$ENV_FILE" ]; then
    set -o allexport
    source "$ENV_FILE"
    set +o allexport
fi

# 离线缓存设置 (确保 Hugging Face 在本地工作)
export HF_HOME="${HF_HOME:-/data2/jflin/CS3602/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Python 解释器
PYTHON="${PYTHON_BIN:-kvpress/.venv/bin/python}"

# 检查 Python 解释器
if [ ! -f "$PYTHON" ]; then
    echo -e "${RED}错误: 找不到 Python 解释器 $PYTHON${NC}"
    echo "请先创建虚拟环境: python -m venv kvpress/.venv"
    exit 1
fi

# 创建结果目录
mkdir -p results/streaming_llm
mkdir -p results/kvpress
mkdir -p results/figures

# 日志文件
LOG_FILE="results/experiment_log_$(date +%Y%m%d_%H%M%S).txt"
SUMMARY_FILE="results/experiment_summary_$(date +%Y%m%d_%H%M%S).txt"

# 记录开始时间
TOTAL_START=$(date +%s)

echo "################################################################################" | tee -a "$LOG_FILE"
echo "# StreamingLLM 完整实验" | tee -a "$LOG_FILE"
echo "# 开始时间: $(date)" | tee -a "$LOG_FILE"
echo "################################################################################" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 实验计数器
TOTAL_EXPERIMENTS=0
SUCCESSFUL_EXPERIMENTS=0
FAILED_EXPERIMENTS=0

# 运行实验的辅助函数
run_experiment() {
    local name="$1"
    local cmd="$2"
    
    TOTAL_EXPERIMENTS=$((TOTAL_EXPERIMENTS + 1))
    
    echo "" | tee -a "$LOG_FILE"
    echo -e "${BLUE}========================================${NC}" | tee -a "$LOG_FILE"
    echo -e "${BLUE}实验 $TOTAL_EXPERIMENTS: $name${NC}" | tee -a "$LOG_FILE"
    echo -e "${BLUE}========================================${NC}" | tee -a "$LOG_FILE"
    echo "命令: $cmd" | tee -a "$LOG_FILE"
    echo "开始时间: $(date)" | tee -a "$LOG_FILE"
    
    local start=$(date +%s)
    
    # 运行命令
    eval "$cmd" 2>&1 | tee -a "$LOG_FILE"
    local exit_code=${PIPESTATUS[0]}
    
    local end=$(date +%s)
    local duration=$((end - start))
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ 成功${NC} (耗时: ${duration}s)" | tee -a "$LOG_FILE"
        SUCCESSFUL_EXPERIMENTS=$((SUCCESSFUL_EXPERIMENTS + 1))
        echo "$name|SUCCESS|${duration}s" >> "$SUMMARY_FILE"
    else
        echo -e "${RED}✗ 失败${NC} (退出码: $exit_code, 耗时: ${duration}s)" | tee -a "$LOG_FILE"
        FAILED_EXPERIMENTS=$((FAILED_EXPERIMENTS + 1))
        echo "$name|FAILED|${duration}s|exit_code=$exit_code" >> "$SUMMARY_FILE"
    fi
    
    echo "" | tee -a "$LOG_FILE"
}

################################################################################
# 1. Baseline 实验
################################################################################

echo -e "${YELLOW}=== 阶段 1: Baseline 实验 ===${NC}" | tee -a "$LOG_FILE"

# WikiText-103 Baseline
run_experiment \
    "WikiText-103 Baseline" \
    "$PYTHON experiments/eval_streaming_llm.py \
        --dataset-name wikitext \
        --dataset-config wikitext-103-v1 \
        --max-samples 64 \
        --max-eval-tokens 4096 \
        --n-sink 0 \
        --window-size 1024 \
        --max-length 2048 \
        --stride 1024 \
        --mode baseline \
        --output results/streaming_llm/wikitext_baseline.json"

# PG19 Baseline
run_experiment \
    "PG19 Baseline" \
    "$PYTHON experiments/eval_streaming_llm.py \
        --dataset-name pg19 \
        --dataset-config \"\" \
        --max-samples 1 \
        --max-eval-tokens 4096 \
        --n-sink 0 \
        --window-size 1024 \
        --max-length 2048 \
        --stride 1024 \
        --mode baseline \
        --output results/streaming_llm/pg19_baseline.json"

################################################################################
# 2. 我们的 StreamingLLM 实验
################################################################################

echo -e "${YELLOW}=== 阶段 2: 我们的 StreamingLLM 实验 ===${NC}" | tee -a "$LOG_FILE"

# WikiText-103 StreamingLLM
run_experiment \
    "WikiText-103 StreamingLLM" \
    "$PYTHON experiments/eval_streaming_llm.py \
        --dataset-name wikitext \
        --dataset-config wikitext-103-v1 \
        --max-samples 64 \
        --max-eval-tokens 4096 \
        --n-sink 4 \
        --window-size 1024 \
        --max-length 2048 \
        --stride 1024 \
        --mode streaming \
        --baseline-results results/streaming_llm/wikitext_baseline.json \
        --output results/streaming_llm/wikitext_result.json"

run_experiment \
    "WikiText-103 MIT StreamingLLM" \
    "$PYTHON experiments/eval_streaming_llm.py \
        --dataset-name wikitext \
        --dataset-config wikitext-103-v1 \
        --max-samples 64 \
        --max-eval-tokens 4096 \
        --n-sink 4 \
        --window-size 1024 \
        --max-length 2048 \
        --stride 1024 \
        --streaming-mode mit \
        --mode streaming \
        --baseline-results results/streaming_llm/wikitext_baseline.json \
        --output results/streaming_llm/wikitext_mit_result.json"

# PG19 StreamingLLM
run_experiment \
    "PG19 StreamingLLM" \
    "$PYTHON experiments/eval_streaming_llm.py \
        --dataset-name pg19 \
        --dataset-config \"\" \
        --max-samples 1 \
        --max-eval-tokens 4096 \
        --n-sink 4 \
        --window-size 1024 \
        --max-length 2048 \
        --stride 1024 \
        --mode streaming \
        --baseline-results results/streaming_llm/pg19_baseline.json \
        --output results/streaming_llm/pg19_result.json"

run_experiment \
    "PG19 MIT StreamingLLM" \
    "$PYTHON experiments/eval_streaming_llm.py \
        --dataset-name pg19 \
        --dataset-config \"\" \
        --max-samples 1 \
        --max-eval-tokens 4096 \
        --n-sink 4 \
        --window-size 1024 \
        --max-length 2048 \
        --stride 1024 \
        --streaming-mode mit \
        --mode streaming \
        --baseline-results results/streaming_llm/pg19_baseline.json \
        --output results/streaming_llm/pg19_mit_result.json"

################################################################################
# 3. kvpress 官方库对比实验
################################################################################

echo -e "${YELLOW}=== 阶段 3: kvpress 官方库对比实验 ===${NC}" | tee -a "$LOG_FILE"

# WikiText-103 kvpress StreamingLLM (decode-loop)
run_experiment \
    "WikiText-103 kvpress StreamingLLM (decode-loop)" \
    "bash experiments/run_kvpress_streaming_decode.sh | tee -a \"$LOG_FILE\""

# WikiText-103 kvpress StreamingLLM (official eval)
run_experiment \
    "WikiText-103 kvpress StreamingLLM (official eval)" \
    "$PYTHON experiments/eval_kvpress.py \
        --dataset-name wikitext \
        --dataset-config wikitext-103-v1 \
        --max-samples 64 \
        --max-eval-tokens 4096 \
        --n-sink 4 \
        --window-size 1024 \
        --max-length 2048 \
        --stride 1024 \
        --output results/kvpress/wikitext_result.json"

# PG19 kvpress StreamingLLM (decode-loop)
run_experiment \
    "PG19 kvpress StreamingLLM (decode-loop)" \
    "DATASET_NAME=pg19 DATASET_CONFIG=\"\" bash experiments/run_kvpress_streaming_decode.sh | tee -a \"$LOG_FILE\""

# PG19 kvpress StreamingLLM (official eval)
run_experiment \
    "PG19 kvpress StreamingLLM (official eval)" \
    "$PYTHON experiments/eval_kvpress.py \
        --dataset-name pg19 \
        --dataset-config \"\" \
        --max-samples 1 \
        --max-eval-tokens 4096 \
        --n-sink 4 \
        --window-size 1024 \
        --max-length 2048 \
        --stride 1024 \
        --output results/kvpress/pg19_result.json"

################################################################################
# 4. 消融实验
################################################################################

echo -e "${YELLOW}=== 阶段 4: 消融实验 ===${NC}" | tee -a "$LOG_FILE"

# Window Size 消融
run_experiment \
    "Window Size 消融实验" \
    "$PYTHON experiments/ablation_study.py \
        --dataset-name wikitext \
        --dataset-config wikitext-103-v1 \
        --max-samples 64 \
        --max-eval-tokens 4096 \
        --max-length 2048 \
        --stride 1024 \
        --ablation-type window_size \
        --output results/streaming_llm/ablation_window_size.json"

# N_sink 消融
run_experiment \
    "N_sink 消融实验" \
    "$PYTHON experiments/ablation_study.py \
        --dataset-name wikitext \
        --dataset-config wikitext-103-v1 \
        --max-samples 64 \
        --max-eval-tokens 4096 \
        --max-length 2048 \
        --stride 1024 \
        --ablation-type n_sink \
        --output results/streaming_llm/ablation_n_sink.json"

################################################################################
# 5. 生成可视化图表
################################################################################

echo -e "${YELLOW}=== 阶段 5: 生成可视化图表 ===${NC}" | tee -a "$LOG_FILE"

run_experiment \
    "生成可视化图表" \
    "$PYTHON experiments/generate_final_figures.py"

################################################################################
# 总结报告
################################################################################

TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))

echo "" | tee -a "$LOG_FILE"
echo "################################################################################" | tee -a "$LOG_FILE"
echo "# 实验完成报告" | tee -a "$LOG_FILE"
echo "################################################################################" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "结束时间: $(date)" | tee -a "$LOG_FILE"
echo "总耗时: ${TOTAL_DURATION}s ($(($TOTAL_DURATION / 60))分钟)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "实验统计:" | tee -a "$LOG_FILE"
echo "  总实验数: $TOTAL_EXPERIMENTS" | tee -a "$LOG_FILE"
echo -e "  ${GREEN}成功: $SUCCESSFUL_EXPERIMENTS${NC}" | tee -a "$LOG_FILE"
echo -e "  ${RED}失败: $FAILED_EXPERIMENTS${NC}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

if [ -f "$SUMMARY_FILE" ]; then
    echo "详细结果:" | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"
    while IFS='|' read -r name status duration extra; do
        if [ "$status" = "SUCCESS" ]; then
            echo -e "  ${GREEN}✓${NC} $name ($duration)" | tee -a "$LOG_FILE"
        else
            echo -e "  ${RED}✗${NC} $name ($duration) [$extra]" | tee -a "$LOG_FILE"
        fi
    done < "$SUMMARY_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "日志文件: $LOG_FILE" | tee -a "$LOG_FILE"
echo "总结文件: $SUMMARY_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

if [ $FAILED_EXPERIMENTS -eq 0 ]; then
    echo -e "${GREEN}✓ 所有实验成功完成!${NC}" | tee -a "$LOG_FILE"
    exit 0
else
    echo -e "${YELLOW}⚠ 部分实验失败,请查看日志文件${NC}" | tee -a "$LOG_FILE"
    exit 1
fi
