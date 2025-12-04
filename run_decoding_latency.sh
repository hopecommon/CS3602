#!/bin/bash
################################################################################
# Decoding Latency 实验一键运行脚本
#
# 使用说明:
#   chmod +x run_decoding_latency.sh
#   ./run_decoding_latency.sh
#
# 功能:
#   测量 per-token decoding latency，对比 Baseline 和 StreamingLLM
#   - 多种 cache size 配置
#   - 多次运行取平均
#   - GPU warmup
#   - 精确计时 (torch.cuda.synchronize)
#
# 注意:
#   - 使用 kvpress/.venv/bin/python 作为 Python 解释器
#   - 需要 CUDA GPU
#   - 结果保存在 results/ 目录
################################################################################

set +e  # 允许单个实验失败而不中断整个流程

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Python 解释器
PYTHON="kvpress/.venv/bin/python"

# 检查 Python 解释器
if [ ! -f "$PYTHON" ]; then
    echo -e "${RED}错误: 找不到 Python 解释器 $PYTHON${NC}"
    echo "请先创建虚拟环境: python -m venv kvpress/.venv"
    exit 1
fi

# 检查 CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}警告: 未检测到 CUDA，decoding latency 实验可能无法运行${NC}"
fi

# 创建结果目录
mkdir -p results/decoding

# 日志文件
LOG_FILE="results/decoding/decoding_latency_log_$(date +%Y%m%d_%H%M%S).txt"
SUMMARY_FILE="results/decoding/decoding_latency_summary_$(date +%Y%m%d_%H%M%S).txt"

# 记录开始时间
TOTAL_START=$(date +%s)

echo "################################################################################" | tee -a "$LOG_FILE"
echo "# Decoding Latency 实验" | tee -a "$LOG_FILE"
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
# Decoding Latency 实验
################################################################################

echo -e "${YELLOW}=== Decoding Latency 实验 ===${NC}" | tee -a "$LOG_FILE"
echo "测量配置:" | tee -a "$LOG_FILE"
echo "  - Prompt Length: 512" | tee -a "$LOG_FILE"
echo "  - Num Tokens: 2000" | tee -a "$LOG_FILE"
echo "  - Warmup Tokens: 200" | tee -a "$LOG_FILE"
echo "  - Num Runs: 3" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Cache Size 512
run_experiment \
    "Decoding Latency (Cache=512)" \
    "$PYTHON experiments/eval_decoding_latency.py \
        --cache-size 512 \
        --n-sink 4 \
        --prompt-length 512 \
        --num-tokens 2000 \
        --warmup-tokens 200 \
        --num-runs 3 \
        --output results/decoding/decoding_latency_512.json"

# Cache Size 1024
run_experiment \
    "Decoding Latency (Cache=1024)" \
    "$PYTHON experiments/eval_decoding_latency.py \
        --cache-size 1024 \
        --n-sink 4 \
        --prompt-length 512 \
        --num-tokens 2000 \
        --warmup-tokens 200 \
        --num-runs 3 \
        --output results/decoding/decoding_latency_1024.json"

# Cache Size 2048
run_experiment \
    "Decoding Latency (Cache=2048)" \
    "$PYTHON experiments/eval_decoding_latency.py \
        --cache-size 2048 \
        --n-sink 4 \
        --prompt-length 512 \
        --num-tokens 2000 \
        --warmup-tokens 200 \
        --num-runs 3 \
        --output results/decoding/decoding_latency_2048.json"

# Cache Size 4096
run_experiment \
    "Decoding Latency (Cache=4096)" \
    "$PYTHON experiments/eval_decoding_latency.py \
        --cache-size 4096 \
        --n-sink 4 \
        --prompt-length 512 \
        --num-tokens 2000 \
        --warmup-tokens 200 \
        --num-runs 3 \
        --output results/decoding/decoding_latency_4096.json"

################################################################################
# 长序列测试 (可选)
################################################################################

echo -e "${YELLOW}=== 长序列 Decoding Latency 测试 ===${NC}" | tee -a "$LOG_FILE"

# 长序列: 5000 tokens
run_experiment \
    "Long Sequence Decoding (5000 tokens, Cache=1024)" \
    "$PYTHON experiments/eval_decoding_latency.py \
        --cache-size 1024 \
        --n-sink 4 \
        --prompt-length 512 \
        --num-tokens 5000 \
        --warmup-tokens 200 \
        --num-runs 3 \
        --output results/decoding/decoding_latency_long_5000.json"

################################################################################
# 不同 n_sink 配置测试
################################################################################

echo -e "${YELLOW}=== 不同 n_sink 配置测试 ===${NC}" | tee -a "$LOG_FILE"

# n_sink = 0
run_experiment \
    "Decoding Latency (n_sink=0, Cache=1024)" \
    "$PYTHON experiments/eval_decoding_latency.py \
        --cache-size 1024 \
        --n-sink 0 \
        --prompt-length 512 \
        --num-tokens 2000 \
        --warmup-tokens 200 \
        --num-runs 3 \
        --output results/decoding/decoding_latency_nsink0.json"

# n_sink = 8
run_experiment \
    "Decoding Latency (n_sink=8, Cache=1024)" \
    "$PYTHON experiments/eval_decoding_latency.py \
        --cache-size 1024 \
        --n-sink 8 \
        --prompt-length 512 \
        --num-tokens 2000 \
        --warmup-tokens 200 \
        --num-runs 3 \
        --output results/decoding/decoding_latency_nsink8.json"

################################################################################
# 总结报告
################################################################################

TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))

echo "" | tee -a "$LOG_FILE"
echo "################################################################################" | tee -a "$LOG_FILE"
echo "# Decoding Latency 实验完成报告" | tee -a "$LOG_FILE"
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
echo "结果文件:" | tee -a "$LOG_FILE"
ls -lh results/decoding/decoding_latency*.json 2>/dev/null | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "日志文件: $LOG_FILE" | tee -a "$LOG_FILE"
echo "总结文件: $SUMMARY_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

if [ $FAILED_EXPERIMENTS -eq 0 ]; then
    echo -e "${GREEN}✓ 所有 Decoding Latency 实验成功完成!${NC}" | tee -a "$LOG_FILE"
    exit 0
else
    echo -e "${YELLOW}⚠ 部分实验失败,请查看日志文件${NC}" | tee -a "$LOG_FILE"
    exit 1
fi