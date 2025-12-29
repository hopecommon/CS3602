#!/bin/bash
################################################################################
# 论文核心数字快速验证脚本
#
# 只运行最关键的实验来验证论文中的核心数字:
#   1. MIT StreamingLLM (baseline for comparison)
#   2. Ours (Best, R=64) - 论文主要结果
#   3. Ours (Softlite, R=32) - 最佳质量-速度平衡
#   4. Table 2 协同效应验证
#
# 使用说明:
#   chmod +x run_paper_quick_check.sh
#   ./run_paper_quick_check.sh
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
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

PYTHON="${PYTHON_BIN:-kvpress/.venv/bin/python}"
MODEL_NAME="${MODEL_NAME:-EleutherAI/pythia-2.8b}"
RESULT_DIR="results/paper_quick_check"
mkdir -p "$RESULT_DIR"

# 实验配置 - 使用更小的样本数快速验证
PG19_MAX_TOKENS=20000
MAX_SAMPLES=3  # 快速验证用 3 samples，完整实验用 10
DATASET_NAME="pg19"
DATASET_CONFIG=""

# 颜色
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

run_experiment() {
  local name="$1"
  shift
  echo ""
  echo -e "${BLUE}========================================${NC}"
  echo -e "${BLUE}$name${NC}"
  echo -e "${BLUE}========================================${NC}"
  
  local start=$(date +%s)
  "$@"
  local end=$(date +%s)
  local duration=$((end - start))
  
  echo -e "${GREEN}✓ 完成${NC} (耗时: ${duration}s)"
}

echo ""
echo -e "${YELLOW}================================================================${NC}"
echo -e "${YELLOW}NeurIPS 论文核心数字 - 快速验证${NC}"
echo -e "${YELLOW}================================================================${NC}"
echo "模型: $MODEL_NAME"
echo "数据集: PG19 (20k tokens)"
echo "样本数: $MAX_SAMPLES (快速验证模式)"
echo "结果目录: $RESULT_DIR"
echo ""
echo "注意: 这是快速验证模式，使用 $MAX_SAMPLES 个样本。"
echo "      完整实验请使用 ./run_paper_experiments.sh (10 samples)"
echo -e "${YELLOW}================================================================${NC}"
echo ""

################################################################################
# 核心实验
################################################################################

# 1. MIT StreamingLLM - 对比基准
echo -e "${YELLOW}实验 1/4: MIT StreamingLLM (对比基准)${NC}"
OUT_MIT="$RESULT_DIR/mit_streaming.json"
run_experiment \
  "MIT StreamingLLM" \
  "$PYTHON" experiments/run_decode_perplexity.py \
    --method mit \
    --model-name "$MODEL_NAME" \
    --dataset-name "$DATASET_NAME" \
    --dataset-config "$DATASET_CONFIG" \
    --max-eval-tokens $PG19_MAX_TOKENS \
    --max-samples $MAX_SAMPLES \
    --n-sink 4 \
    --window-size 2044 \
    --compress-every 1 \
    --output "$OUT_MIT"

# 2. Ours (Best, R=64) - 论文主要结果: 7.11× speedup, +2.82% PPL
echo -e "${YELLOW}实验 2/4: Ours (Best, R=64) - 论文主要结果${NC}"
OUT_BEST="$RESULT_DIR/ours_best_r64.json"
run_experiment \
  "Ours (Best, R=64)" \
  "$PYTHON" experiments/eval_streaming_llm.py \
    --model-name "$MODEL_NAME" \
    --dataset-name "$DATASET_NAME" \
    --dataset-config "$DATASET_CONFIG" \
    --max-eval-tokens $PG19_MAX_TOKENS \
    --max-samples $MAX_SAMPLES \
    --n-sink 32 \
    --window-size 2016 \
    --compress-every 64 \
    --cache-slack 16 \
    --max-drop 32 \
    --mode streaming \
    --output "$OUT_BEST"

# 3. Ours (Softlite, R=32) - 最佳质量-速度平衡: 7.07×, +2.19%
echo -e "${YELLOW}实验 3/4: Ours (Softlite, R=32) - 最佳平衡${NC}"
OUT_SOFTLITE="$RESULT_DIR/ours_softlite_r32.json"
run_experiment \
  "Ours (Softlite, R=32)" \
  "$PYTHON" experiments/eval_streaming_llm.py \
    --model-name "$MODEL_NAME" \
    --dataset-name "$DATASET_NAME" \
    --dataset-config "$DATASET_CONFIG" \
    --max-eval-tokens $PG19_MAX_TOKENS \
    --max-samples $MAX_SAMPLES \
    --n-sink 32 \
    --window-size 2016 \
    --compress-every 32 \
    --cache-slack 16 \
    --max-drop 32 \
    --mode streaming \
    --output "$OUT_SOFTLITE"

# 4. Ablation: σ=0, δ=∞ (验证协同效应基线)
echo -e "${YELLOW}实验 4/4: Ablation Baseline (σ=0, δ=∞)${NC}"
OUT_ABL="$RESULT_DIR/ablation_baseline.json"
run_experiment \
  "Ablation Baseline" \
  "$PYTHON" experiments/eval_streaming_llm.py \
    --model-name "$MODEL_NAME" \
    --dataset-name "$DATASET_NAME" \
    --dataset-config "$DATASET_CONFIG" \
    --max-eval-tokens $PG19_MAX_TOKENS \
    --max-samples $MAX_SAMPLES \
    --n-sink 32 \
    --window-size 2016 \
    --compress-every 32 \
    --cache-slack 0 \
    --max-drop 0 \
    --mode streaming \
    --output "$OUT_ABL"

################################################################################
# 生成对比报告
################################################################################

echo ""
echo -e "${YELLOW}================================================================${NC}"
echo -e "${YELLOW}生成对比报告...${NC}"
echo -e "${YELLOW}================================================================${NC}"

cat > "$RESULT_DIR/compare_results.py" << 'EOF'
#!/usr/bin/env python3
"""快速对比实验结果与论文数字"""
import json
from pathlib import Path

def load_result(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

def extract_metrics(data):
    if not data:
        return None
    streaming = data.get("streaming", {})
    metrics = data.get("metrics", {})
    return {
        "speedup": metrics.get("speedup", 0),
        "tpot": streaming.get("tpot_ms", 0),
        "ppl_inc": metrics.get("ppl_increase_percent", 0),
    }

result_dir = Path("results/paper_quick_check")

print("\n" + "="*80)
print("核心数字对比: 实验结果 vs. 论文数字")
print("="*80)
print("注意: 快速验证使用 3 samples，数字会有波动。完整实验需要 10 samples。")
print("="*80)
print()

# MIT StreamingLLM
print("【1】MIT StreamingLLM")
mit = extract_metrics(load_result(result_dir / "mit_streaming.json"))
if mit:
    print(f"  实验: {mit['speedup']:.2f}× speedup, {mit['tpot']:.1f}ms TPOT, +{mit['ppl_inc']:.2f}% PPL")
    print(f"  论文: 6.86× speedup, 14.6ms TPOT, +2.33% PPL")
else:
    print("  实验: N/A")
print()

# Ours (Best, R=64)
print("【2】Ours (Best, R=64) - 论文主要结果")
best = extract_metrics(load_result(result_dir / "ours_best_r64.json"))
if best:
    print(f"  实验: {best['speedup']:.2f}× speedup, {best['tpot']:.1f}ms TPOT, +{best['ppl_inc']:.2f}% PPL")
    print(f"  论文: 7.11× speedup, 14.1ms TPOT, +2.82% PPL")
    if best['speedup'] >= 6.8:
        print("  ✓ Speedup 接近论文数字")
    else:
        print("  ⚠ Speedup 低于预期")
else:
    print("  实验: N/A")
print()

# Ours (Softlite, R=32)
print("【3】Ours (Softlite, R=32) - 最佳平衡")
soft = extract_metrics(load_result(result_dir / "ours_softlite_r32.json"))
if soft:
    print(f"  实验: {soft['speedup']:.2f}× speedup, {soft['tpot']:.1f}ms TPOT, +{soft['ppl_inc']:.2f}% PPL")
    print(f"  论文: 7.07× speedup, 14.2ms TPOT, +2.19% PPL")
    if soft['ppl_inc'] <= 3.0:
        print("  ✓ PPL increase 低于 3%，质量良好")
    else:
        print("  ⚠ PPL increase 较高")
else:
    print("  实验: N/A")
print()

# Ablation Baseline
print("【4】Ablation Baseline (σ=0, δ=∞) - 协同效应对比")
abl = extract_metrics(load_result(result_dir / "ablation_baseline.json"))
if abl and soft:
    print(f"  Baseline (无优化): {abl['speedup']:.2f}×, +{abl['ppl_inc']:.2f}% PPL")
    print(f"  Combined (σ=16,δ=32): {soft['speedup']:.2f}×, +{soft['ppl_inc']:.2f}% PPL")
    improvement = ((abl['ppl_inc'] - soft['ppl_inc']) / abl['ppl_inc']) * 100
    print(f"  PPL 改进: {improvement:.1f}% (论文: 66%)")
    if improvement >= 50:
        print("  ✓ 协同效应显著")
    else:
        print("  ⚠ 协同效应弱于预期 (可能是样本数不足)")
elif abl:
    print(f"  Baseline: {abl['speedup']:.2f}×, +{abl['ppl_inc']:.2f}% PPL")
    print(f"  论文: 6.94×, +6.48% PPL")
else:
    print("  实验: N/A")

print()
print("="*80)
print("验证完成！")
print("="*80)
print()
print("如需更精确的数字，请运行完整实验:")
print("  ./run_paper_experiments.sh")
print()
EOF

chmod +x "$RESULT_DIR/compare_results.py"
"$PYTHON" "$RESULT_DIR/compare_results.py"

echo ""
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}快速验证完成！${NC}"
echo -e "${GREEN}================================================================${NC}"
echo "结果目录: $RESULT_DIR"
echo ""
echo "查看详细结果:"
echo "  ls -lh $RESULT_DIR/"
echo "  cat $RESULT_DIR/ours_best_r64.json | jq '.metrics'"
echo ""
echo "运行完整实验 (10 samples):"
echo "  ./run_paper_experiments.sh"
echo ""
