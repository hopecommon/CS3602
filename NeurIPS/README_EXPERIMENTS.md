# NeurIPS 论文实验复现指南

本文档说明如何复现论文 "Efficient KV Cache Management for Long-Context LLM Inference: Lazy Pruning and Soft Eviction Strategies" 中的实验结果。

## 文件结构

```
CS3602/
├── NeurIPS/
│   ├── neurips_2025.tex              # 原版论文 (16页)
│   ├── neurips_2025_compressed.tex   # 压缩版论文 (11页，3-4页正文)
│   ├── references.bib                # 参考文献
│   └── neurips_2025.sty              # NeurIPS 样式文件
├── experiments/
│   ├── eval_streaming_llm.py         # 主评估脚本
│   ├── run_decode_perplexity.py      # Decode-loop PPL 评估
│   └── ...
├── run_paper_experiments.sh          # 完整实验脚本 (所有表格)
└── run_paper_quick_check.sh          # 快速验证脚本 (核心数字)
```

## 快速开始

### 1. 快速验证核心数字 (推荐首次运行)

```bash
# 运行 3 个样本，验证论文中的核心数字
./run_paper_quick_check.sh
```

**验证内容**：
- MIT StreamingLLM (对比基准): 6.86× speedup, +2.33% PPL
- Ours (Best, R=64): 7.11× speedup, +2.82% PPL, 14.1ms TPOT
- Ours (Softlite, R=32): 7.07× speedup, +2.19% PPL
- Ablation 协同效应: 66% PPL improvement

**预计时间**: 约 30-60 分钟 (取决于 GPU)

### 2. 完整实验 (复现所有表格)

```bash
# 运行 10 个样本，复现论文中的所有表格
./run_paper_experiments.sh
```

**复现内容**：
- **Table 1**: Main Results (PG19 20k tokens)
  - Full Recomputation, MIT StreamingLLM, Ours (Eager/Best/Softlite)
  
- **Table 2**: Ablation - Synergistic Interaction
  - σ=0,δ=∞ | σ=16,δ=∞ | σ=0,δ=32 | σ=16,δ=32

- **Appendix Tables**: Parameter Sweeps
  - Lazy Pruning (R): 1, 16, 32, 64, 128
  - Slack (σ): 0, 8, 16, 32, 64
  - Max_Drop (δ): 0, 8, 16, 32, 64
  - Enhanced Sinks (S): 4, 8, 16, 32, 64

**预计时间**: 约 3-6 小时

## 实验参数配置

### 核心配置映射

| 论文配置 | 参数 |
|---------|------|
| **Baseline (MIT StreamingLLM)** | `S=4, W=2044, R=1` |
| **Ours (Eager, R=1)** | `S=32, W=2016, R=1, σ=16, δ=32` |
| **Ours (Best, R=64)** | `S=32, W=2016, R=64, σ=16, δ=32` |
| **Ours (Softlite, R=32)** | `S=32, W=2016, R=32, σ=16, δ=32` |

### 参数说明

- `S` (`--n-sink`): Attention sink token 数量
- `W` (`--window-size`): 滑动窗口大小
- `R` (`--compress-every`): Lazy Pruning 压缩间隔
- `σ` (`--cache-slack`): Slack 缓冲区大小
- `δ` (`--max-drop`): Max_Drop 单次最大驱逐 token 数

## 查看结果

### 快速验证结果

```bash
# 查看对比报告
python results/paper_quick_check/compare_results.py

# 查看详细 JSON
cat results/paper_quick_check/ours_best_r64.json | jq '.metrics'
```

### 完整实验结果

```bash
# 查看汇总表格
python results/paper_experiments/generate_summary.py

# 查看所有结果文件
ls -lh results/paper_experiments/

# 查看特定实验
cat results/paper_experiments/table1_ours_best_r64.json | jq .
```

### 关键指标

实验输出的 JSON 文件包含以下关键指标：

```json
{
  "metrics": {
    "speedup": 7.11,           // 相对 baseline 的加速比
    "tpot_ms": 14.1,           // Time Per Output Token (ms)
    "ppl_increase_percent": 2.82,  // 相对 baseline 的 PPL 增加 (%)
    "peak_memory_mb": 6616     // 峰值显存 (MB)
  },
  "streaming": {
    "perplexity": 42.51,       // 绝对 perplexity
    "runtime_sec": 14.15,      // 总运行时间 (秒)
    "decode_time_sec": 282.1,  // 解码时间 (秒)
    "decode_tokens": 20000     // 解码 token 数
  }
}
```

## 预期结果对比

### Table 1: Main Results (PG19 20k)

| Method | Speedup | TPOT (ms) | PPL Inc (%) | Memory (MB) |
|--------|---------|-----------|-------------|-------------|
| Full Recomputation | 1.00× | 100.5 | 0.00 | 6621 |
| MIT StreamingLLM | 6.86× | 14.6 | +2.33 | 6617 |
| **Ours (Best, R=64)** | **7.11×** | **14.1** | **+2.82** | **6616** |
| Ours (Softlite, R=32) | 7.07× | 14.2 | +2.19 | 6616 |

### Table 2: Ablation Study

| σ | δ | Speedup | TPOT (ms) | PPL Inc (%) |
|---|---|---------|-----------|-------------|
| 0 | ∞ | 6.94× | 14.5 | +6.48 |
| 16 | ∞ | 7.03× | 14.3 | +6.01 |
| 0 | 32 | 6.88× | 14.6 | +5.94 |
| **16** | **32** | **7.07×** | **14.2** | **+2.19** |

**关键发现**: 
- Slack alone 或 Max_Drop alone: 仅 +7% 和 +8% 改进
- Combined (σ=16, δ=32): **66% improvement**, 3.8× better than additive

## 环境配置

### 必需的环境变量

创建 `.env` 文件 (可选):

```bash
# Python 路径
PYTHON_BIN=kvpress/.venv/bin/python

# 模型
MODEL_NAME=EleutherAI/pythia-2.8b

# HuggingFace 缓存
HF_HOME=$PWD/.cache/huggingface
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1

# 实验参数
PG19_MAX_TOKENS=20000
WIKITEXT_MAX_TOKENS=4096
```

### 硬件要求

- **GPU**: NVIDIA A800 (80GB) 或类似 (最低 24GB VRAM)
- **Model**: Pythia-2.8B (FP16)
- **Dataset**: PG19 (20k tokens per sample)

## 故障排除

### 1. 跳过已存在的实验

脚本默认跳过已存在的结果文件。如需重新运行：

```bash
# 删除特定结果
rm results/paper_experiments/table1_ours_best_r64.json

# 删除所有结果并重新运行
rm -rf results/paper_experiments/
./run_paper_experiments.sh
```

### 2. 数据集下载

首次运行会自动下载 PG19 数据集：

```bash
# 手动预下载 (可选)
python -c "
from datasets import load_dataset
dataset = load_dataset('pg19', split='test')
print(f'Downloaded {len(dataset)} samples')
"
```

### 3. 内存不足

如果遇到 OOM 错误：

```bash
# 减少样本数 (快速验证脚本已默认使用 3 samples)
export MAX_SAMPLES=1

# 或使用更小的模型
export MODEL_NAME=EleutherAI/pythia-70m
```

## 论文编译

### 编译压缩版 (推荐提交)

```bash
cd NeurIPS
pdflatex neurips_2025_compressed.tex
bibtex neurips_2025_compressed
pdflatex neurips_2025_compressed.tex
pdflatex neurips_2025_compressed.tex
```

**输出**: 11 页 (3-4 页正文 + 7 页附录)

### 修改作者信息

编辑 `neurips_2025_compressed.tex` 第 23-35 行：

```latex
\author{%
  学生姓名1 \\
  学号: 学号1 \\
  上海交通大学 计算机科学与工程系 \\
  \texttt{email1@sjtu.edu.cn} \\
  \And
  学生姓名2 \\
  学号: 学号2 \\
  上海交通大学 计算机科学与工程系 \\
  \texttt{email2@sjtu.edu.cn} \\
  \AND
  指导教师: 教师姓名 \\
  上海交通大学 计算机科学与工程系 \\
}
```

### 使用 preprint 模式显示作者

确保第 4 行使用 `[preprint]` 选项：

```latex
\usepackage[preprint]{neurips_2025}
```

## 引用

如果使用本代码，请引用：

```bibtex
@inproceedings{streamingllm-efficient,
  title={Efficient KV Cache Management for Long-Context LLM Inference: Lazy Pruning and Soft Eviction Strategies},
  author={学生姓名1 and 学生姓名2 and 指导教师},
  booktitle={NeurIPS 2025 Course Project},
  year={2025}
}
```

## 联系方式

如有问题，请联系：
- 学生姓名1: email1@sjtu.edu.cn
- 学生姓名2: email2@sjtu.edu.cn
- 指导教师: 教师姓名

---

**最后更新**: 2025-12-29
