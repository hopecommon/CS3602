#!/bin/bash
################################################################################
# 论文核心实验一键运行脚本
#
# 复现 NeurIPS 论文中的关键实验结果:
#   - Table 1: 主要结果 (PG19 20k tokens)
#   - Table 2: 消融实验 (Slack + Max_Drop 协同效应)
#   - Appendix Tables: R, σ, δ, S 参数扫描
#
# 使用说明:
#   chmod +x run_paper_experiments.sh
#   ./run_paper_experiments.sh
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
RESULT_DIR="results/paper_experiments"
mkdir -p "$RESULT_DIR"

# 实验配置
PG19_MAX_TOKENS=20000
MAX_SAMPLES=10  # 论文中使用 10 samples for mean±std
DATASET_NAME="pg19"
DATASET_CONFIG=""

# 颜色
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

run_experiment() {
  local name="$1"
  shift
  echo ""
  echo -e "${BLUE}========================================${NC}"
  echo -e "${BLUE}$name${NC}"
  echo -e "${BLUE}========================================${NC}"
  printf '命令: '
  printf '%q ' "$@"
  echo
  echo "开始时间: $(date)"
  
  local start=$(date +%s)
  "$@"
  local end=$(date +%s)
  local duration=$((end - start))
  
  echo -e "${GREEN}✓ 完成${NC} (耗时: ${duration}s)"
}

should_skip() {
  local path="$1"
  if [[ -f "$path" ]] && [[ -s "$path" ]]; then
    echo -e "${GREEN}✓ 跳过${NC} (已存在): $path"
    return 0
  fi
  return 1
}

echo ""
echo -e "${YELLOW}================================================================${NC}"
echo -e "${YELLOW}NeurIPS 论文核心实验 - 一键复现${NC}"
echo -e "${YELLOW}================================================================${NC}"
echo "模型: $MODEL_NAME"
echo "数据集: PG19 (20k tokens)"
echo "样本数: $MAX_SAMPLES"
echo "结果目录: $RESULT_DIR"
echo -e "${YELLOW}================================================================${NC}"
echo ""

################################################################################
# Table 1: 主要结果
################################################################################

echo -e "${YELLOW}=== Table 1: Main Results (PG19 20k) ===${NC}"
echo ""

# 1. Full Recomputation Baseline
OUT_BASELINE="$RESULT_DIR/table1_baseline.json"
if ! should_skip "$OUT_BASELINE"; then
  run_experiment \
    "Full Recomputation (Baseline)" \
    "$PYTHON" experiments/run_decode_perplexity.py \
      --method baseline \
      --model-name "$MODEL_NAME" \
      --dataset-name "$DATASET_NAME" \
      --dataset-config "$DATASET_CONFIG" \
      --max-eval-tokens $PG19_MAX_TOKENS \
      --max-samples $MAX_SAMPLES \
      --n-sink 4 \
      --window-size 2044 \
      --compress-every 1 \
      --output "$OUT_BASELINE"
fi

# 2. MIT StreamingLLM (S=4, W=2044)
OUT_MIT="$RESULT_DIR/table1_mit_streaming.json"
if ! should_skip "$OUT_MIT"; then
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
fi

# 3. Ours (Eager, R=1) - S=32, W=2016, R=1, σ=16, δ=32
OUT_EAGER="$RESULT_DIR/table1_ours_eager_r1.json"
if ! should_skip "$OUT_EAGER"; then
  run_experiment \
    "Ours (Eager, R=1)" \
    "$PYTHON" experiments/eval_streaming_llm.py \
      --model-name "$MODEL_NAME" \
      --dataset-name "$DATASET_NAME" \
      --dataset-config "$DATASET_CONFIG" \
      --max-eval-tokens $PG19_MAX_TOKENS \
      --max-samples $MAX_SAMPLES \
      --n-sink 32 \
      --window-size 2016 \
      --compress-every 1 \
      --cache-slack 16 \
      --max-drop 32 \
      --mode streaming \
      --output "$OUT_EAGER"
fi

# 4. Ours (Best, R=64) - S=32, W=2016, R=64, σ=16, δ=32
OUT_BEST="$RESULT_DIR/table1_ours_best_r64.json"
if ! should_skip "$OUT_BEST"; then
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
fi

# 5. Ours (Softlite, R=32) - S=32, W=2016, R=32, σ=16, δ=32
OUT_SOFTLITE="$RESULT_DIR/table1_ours_softlite_r32.json"
if ! should_skip "$OUT_SOFTLITE"; then
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
fi

################################################################################
# Table 2: 消融实验 - Slack + Max_Drop 协同效应
################################################################################

echo ""
echo -e "${YELLOW}=== Table 2: Ablation - Synergy (S=32, W=2016, R=32) ===${NC}"
echo ""

# 1. Baseline: σ=0, δ=∞ (Hard)
OUT_ABL_BASELINE="$RESULT_DIR/table2_baseline_hard.json"
if ! should_skip "$OUT_ABL_BASELINE"; then
  run_experiment \
    "Ablation Baseline (σ=0, δ=∞)" \
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
      --output "$OUT_ABL_BASELINE"
fi

# 2. Slack only: σ=16, δ=∞ (Hard)
OUT_ABL_SLACK="$RESULT_DIR/table2_slack_only.json"
if ! should_skip "$OUT_ABL_SLACK"; then
  run_experiment \
    "Slack Only (σ=16, δ=∞)" \
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
      --max-drop 0 \
      --mode streaming \
      --output "$OUT_ABL_SLACK"
fi

# 3. Max_Drop only: σ=0, δ=32
OUT_ABL_MAXDROP="$RESULT_DIR/table2_maxdrop_only.json"
if ! should_skip "$OUT_ABL_MAXDROP"; then
  run_experiment \
    "Max_Drop Only (σ=0, δ=32)" \
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
      --max-drop 32 \
      --mode streaming \
      --output "$OUT_ABL_MAXDROP"
fi

# 4. Combined: σ=16, δ=32 (same as Softlite)
# Already computed above as OUT_SOFTLITE, just create a symlink
if [[ ! -f "$RESULT_DIR/table2_combined.json" ]]; then
  ln -sf "$(basename "$OUT_SOFTLITE")" "$RESULT_DIR/table2_combined.json"
  echo -e "${GREEN}✓ 链接${NC}: table2_combined.json -> $(basename "$OUT_SOFTLITE")"
fi

################################################################################
# Appendix: Parameter Sweeps
################################################################################

echo ""
echo -e "${YELLOW}=== Appendix: Parameter Sweeps ===${NC}"
echo ""

# Lazy Pruning (R) Sweep: S=32, W=2016, σ=16, δ=32
echo "Lazy Pruning (R) Sweep..."
for R in 1 16 32 64 128; do
  OUT_R="$RESULT_DIR/appendix_r_sweep_R${R}.json"
  if ! should_skip "$OUT_R"; then
    run_experiment \
      "R Sweep (R=$R)" \
      "$PYTHON" experiments/eval_streaming_llm.py \
        --model-name "$MODEL_NAME" \
        --dataset-name "$DATASET_NAME" \
        --dataset-config "$DATASET_CONFIG" \
        --max-eval-tokens $PG19_MAX_TOKENS \
        --max-samples $MAX_SAMPLES \
        --n-sink 32 \
        --window-size 2016 \
        --compress-every $R \
        --cache-slack 16 \
        --max-drop 32 \
        --mode streaming \
        --output "$OUT_R"
  fi
done

# Slack (σ) Sweep: S=32, W=2016, R=32, δ=32
echo ""
echo "Slack (σ) Sweep..."
for SIGMA in 0 8 16 32 64; do
  OUT_SIGMA="$RESULT_DIR/appendix_sigma_sweep_S${SIGMA}.json"
  if ! should_skip "$OUT_SIGMA"; then
    run_experiment \
      "Sigma Sweep (σ=$SIGMA)" \
      "$PYTHON" experiments/eval_streaming_llm.py \
        --model-name "$MODEL_NAME" \
        --dataset-name "$DATASET_NAME" \
        --dataset-config "$DATASET_CONFIG" \
        --max-eval-tokens $PG19_MAX_TOKENS \
        --max-samples $MAX_SAMPLES \
        --n-sink 32 \
        --window-size 2016 \
        --compress-every 32 \
        --cache-slack $SIGMA \
        --max-drop 32 \
        --mode streaming \
        --output "$OUT_SIGMA"
  fi
done

# Max_Drop (δ) Sweep: S=32, W=2016, R=32, σ=16
echo ""
echo "Max_Drop (δ) Sweep..."
for DELTA in 0 8 16 32 64; do
  OUT_DELTA="$RESULT_DIR/appendix_delta_sweep_D${DELTA}.json"
  if ! should_skip "$OUT_DELTA"; then
    run_experiment \
      "Delta Sweep (δ=$DELTA)" \
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
        --max-drop $DELTA \
        --mode streaming \
        --output "$OUT_DELTA"
  fi
done

# Enhanced Sinks (S) Sweep: W=2048-S, R=64, σ=16, δ=32
echo ""
echo "Enhanced Sinks (S) Sweep..."
for S in 4 8 16 32 64; do
  W=$((2048 - S))
  OUT_S="$RESULT_DIR/appendix_s_sweep_S${S}.json"
  if ! should_skip "$OUT_S"; then
    run_experiment \
      "S Sweep (S=$S, W=$W)" \
      "$PYTHON" experiments/eval_streaming_llm.py \
        --model-name "$MODEL_NAME" \
        --dataset-name "$DATASET_NAME" \
        --dataset-config "$DATASET_CONFIG" \
        --max-eval-tokens $PG19_MAX_TOKENS \
        --max-samples $MAX_SAMPLES \
        --n-sink $S \
        --window-size $W \
        --compress-every 64 \
        --cache-slack 16 \
        --max-drop 32 \
        --mode streaming \
        --output "$OUT_S"
  fi
done

################################################################################
# 生成汇总报告
################################################################################

echo ""
echo -e "${YELLOW}================================================================${NC}"
echo -e "${YELLOW}生成汇总报告...${NC}"
echo -e "${YELLOW}================================================================${NC}"

cat > "$RESULT_DIR/generate_summary.py" << 'EOF'
#!/usr/bin/env python3
"""生成论文实验结果汇总表"""
import json
from pathlib import Path
import sys

def load_result(path):
    """加载实验结果 JSON"""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return data
    except:
        return None

def extract_metrics(data):
    """提取关键指标"""
    if data is None:
        return None
    
    baseline = data.get("baseline", {})
    streaming = data.get("streaming", {})
    metrics = data.get("metrics", {})
    
    if streaming:
        # Streaming mode
        ppl = streaming.get("perplexity", 0)
        tpot = streaming.get("tpot_ms", 0)
        runtime = streaming.get("runtime_sec", 0)
        speedup = metrics.get("speedup", 0)
        ppl_inc = metrics.get("ppl_increase_percent", 0)
        memory = streaming.get("peak_memory_mb", 0)
    elif baseline:
        # Baseline mode
        ppl = baseline.get("perplexity", 0)
        tpot = baseline.get("tpot_ms", 0)
        runtime = baseline.get("runtime_sec", 0)
        speedup = 1.0
        ppl_inc = 0.0
        memory = baseline.get("peak_memory_mb", 0)
    else:
        return None
    
    return {
        "ppl": ppl,
        "tpot": tpot,
        "runtime": runtime,
        "speedup": speedup,
        "ppl_inc": ppl_inc,
        "memory": memory,
    }

def main():
    result_dir = Path("results/paper_experiments")
    
    print("\n" + "="*80)
    print("Table 1: Main Results (PG19 20k tokens)")
    print("="*80)
    print(f"{'Method':<25} {'Speedup':<10} {'TPOT (ms)':<12} {'PPL Inc (%)':<15} {'Memory (MB)':<12}")
    print("-"*80)
    
    table1_files = [
        ("Full Recomputation", "table1_baseline.json"),
        ("MIT StreamingLLM", "table1_mit_streaming.json"),
        ("Ours (Eager, R=1)", "table1_ours_eager_r1.json"),
        ("Ours (Best, R=64)", "table1_ours_best_r64.json"),
        ("Ours (Softlite, R=32)", "table1_ours_softlite_r32.json"),
    ]
    
    for name, filename in table1_files:
        data = load_result(result_dir / filename)
        m = extract_metrics(data)
        if m:
            print(f"{name:<25} {m['speedup']:<10.2f} {m['tpot']:<12.2f} {m['ppl_inc']:<15.2f} {m['memory']:<12.0f}")
        else:
            print(f"{name:<25} {'N/A':<10} {'N/A':<12} {'N/A':<15} {'N/A':<12}")
    
    print("\n" + "="*80)
    print("Table 2: Ablation - Synergistic Interaction (S=32, W=2016, R=32)")
    print("="*80)
    print(f"{'σ':<8} {'δ':<8} {'Speedup':<10} {'TPOT (ms)':<12} {'PPL Inc (%)':<15}")
    print("-"*80)
    
    table2_files = [
        ("0", "∞", "table2_baseline_hard.json"),
        ("16", "∞", "table2_slack_only.json"),
        ("0", "32", "table2_maxdrop_only.json"),
        ("16", "32", "table2_combined.json"),
    ]
    
    for sigma, delta, filename in table2_files:
        data = load_result(result_dir / filename)
        m = extract_metrics(data)
        if m:
            print(f"{sigma:<8} {delta:<8} {m['speedup']:<10.2f} {m['tpot']:<12.2f} {m['ppl_inc']:<15.2f}")
        else:
            print(f"{sigma:<8} {delta:<8} {'N/A':<10} {'N/A':<12} {'N/A':<15}")
    
    print("\n" + "="*80)
    print("Appendix: Lazy Pruning (R) Sweep (S=32, W=2016, σ=16, δ=32)")
    print("="*80)
    print(f"{'R':<8} {'Speedup':<10} {'TPOT (ms)':<12} {'PPL Inc (%)':<15} {'Memory (MB)':<12}")
    print("-"*80)
    
    for R in [1, 16, 32, 64, 128]:
        data = load_result(result_dir / f"appendix_r_sweep_R{R}.json")
        m = extract_metrics(data)
        if m:
            print(f"{R:<8} {m['speedup']:<10.2f} {m['tpot']:<12.2f} {m['ppl_inc']:<15.2f} {m['memory']:<12.0f}")
        else:
            print(f"{R:<8} {'N/A':<10} {'N/A':<12} {'N/A':<15} {'N/A':<12}")
    
    print("\n" + "="*80)
    print("Appendix: Slack (σ) Sweep (S=32, W=2016, R=32, δ=32)")
    print("="*80)
    print(f"{'σ':<8} {'Speedup':<10} {'TPOT (ms)':<12} {'PPL Inc (%)':<15} {'Memory (MB)':<12}")
    print("-"*80)
    
    for SIGMA in [0, 8, 16, 32, 64]:
        data = load_result(result_dir / f"appendix_sigma_sweep_S{SIGMA}.json")
        m = extract_metrics(data)
        if m:
            print(f"{SIGMA:<8} {m['speedup']:<10.2f} {m['tpot']:<12.2f} {m['ppl_inc']:<15.2f} {m['memory']:<12.0f}")
        else:
            print(f"{SIGMA:<8} {'N/A':<10} {'N/A':<12} {'N/A':<15} {'N/A':<12}")
    
    print("\n" + "="*80)
    print("Appendix: Max_Drop (δ) Sweep (S=32, W=2016, R=32, σ=16)")
    print("="*80)
    print(f"{'δ':<8} {'Speedup':<10} {'TPOT (ms)':<12} {'PPL Inc (%)':<15} {'Memory (MB)':<12}")
    print("-"*80)
    
    for DELTA in [0, 8, 16, 32, 64]:
        data = load_result(result_dir / f"appendix_delta_sweep_D{DELTA}.json")
        m = extract_metrics(data)
        if m:
            print(f"{DELTA:<8} {m['speedup']:<10.2f} {m['tpot']:<12.2f} {m['ppl_inc']:<15.2f} {m['memory']:<12.0f}")
        else:
            print(f"{DELTA:<8} {'N/A':<10} {'N/A':<12} {'N/A':<15} {'N/A':<12}")
    
    print("\n" + "="*80)
    print("Appendix: Enhanced Sinks (S) Sweep (W=2048-S, R=64, σ=16, δ=32)")
    print("="*80)
    print(f"{'S':<8} {'W':<8} {'Speedup':<10} {'TPOT (ms)':<12} {'PPL Inc (%)':<15}")
    print("-"*80)
    
    for S in [4, 8, 16, 32, 64]:
        W = 2048 - S
        data = load_result(result_dir / f"appendix_s_sweep_S{S}.json")
        m = extract_metrics(data)
        if m:
            print(f"{S:<8} {W:<8} {m['speedup']:<10.2f} {m['tpot']:<12.2f} {m['ppl_inc']:<15.2f}")
        else:
            print(f"{S:<8} {W:<8} {'N/A':<10} {'N/A':<12} {'N/A':<15}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
EOF

chmod +x "$RESULT_DIR/generate_summary.py"
"$PYTHON" "$RESULT_DIR/generate_summary.py"

echo ""
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}所有论文实验完成！${NC}"
echo -e "${GREEN}================================================================${NC}"
echo "结果目录: $RESULT_DIR"
echo ""
echo "可以使用以下命令查看详细结果:"
echo "  ls -lh $RESULT_DIR/"
echo "  cat $RESULT_DIR/table1_ours_best_r64.json | jq ."
echo "  $PYTHON $RESULT_DIR/generate_summary.py"
echo ""
