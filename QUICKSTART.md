# 快速开始指南

本文档提供快速运行项目的完整步骤说明,包括环境配置、快速测试、运行实验和故障排除。

## 📋 目录

- [环境配置](#环境配置)
- [快速测试](#快速测试)
- [运行实验](#运行实验)
- [使用一键脚本](#使用一键脚本)
- [故障排除](#故障排除)

---

## 🔧 环境配置

### 推荐方案：复用 kvpress 环境

**优势**:
- ✅ 无需额外安装 - kvpress 已包含所有必需依赖
- ✅ 版本兼容 - 避免依赖冲突
- ✅ 节省时间 - 立即开始实验
- ✅ 环境一致 - 与 kvpress 基线使用相同环境

### 快速配置步骤

```bash
# 1. 进入项目目录
cd /data2/jflin/CS3602

# 2. 激活 kvpress 的虚拟环境
cd kvpress
source .venv/bin/activate
cd ..

# 3. 配置 Hugging Face 缓存
mkdir -p .cache/huggingface
export HF_HOME=$PWD/.cache/huggingface

# 4. (可选) 使用镜像加速下载 (国内用户)
export HF_ENDPOINT=https://hf-mirror.com
```

### 如果 .venv 不存在

首次使用需要创建 kvpress 环境:

```bash
cd kvpress
UV_CACHE_DIR=$PWD/.cache/uv uv sync --all-groups
UV_CACHE_DIR=$PWD/.cache/uv uv sync --extra eval
source .venv/bin/activate
cd ..
```

**说明**: 
- `uv sync --all-groups` 安装所有依赖组
- `uv sync --extra eval` 安装评估相关依赖

### 验证环境

```bash
# 测试 Python 模块
python -c "
import torch
import transformers
from datasets import load_dataset
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ Transformers: {transformers.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
"

# 测试项目模块
python -c "
from streaming_llm import StreamingLLMWrapper
print('✓ StreamingLLM 模块加载成功')
"
```

### kvpress 环境包含的依赖

核心依赖:
- `torch>=2.3.1` - PyTorch 深度学习框架
- `transformers>=4.56` - Hugging Face Transformers
- `datasets>=2.21.0` - 数据集加载
- `accelerate>=1.0.0` - 模型加速
- `numpy>=2.0.0` - 数值计算

评估依赖:
- `pandas>=2.2.2` - 数据处理
- `tqdm>=4.66.4` - 进度条
- `scipy>=1.13.1` - 科学计算
- `matplotlib` - 可视化

---

## ⚡ 快速测试

### 测试核心功能

```bash
# 测试 StreamingLLM 基本功能 (约 1 分钟)
python experiments/test_streaming_llm.py
```

如果测试通过,你会看到:
```
✓ 所有测试通过!
```

### 运行单个实验

```bash
# WikiText-103 评估 (约 2-3 分钟)
python experiments/eval_streaming_llm.py \
  --dataset-name wikitext \
  --dataset-config wikitext-103-v1 \
  --max-samples 64 \
  --max-eval-tokens 4096 \
  --n-sink 4 \
  --window-size 1024 \
  --output results/streaming_llm/wikitext_result.json
```

### 查看结果

```bash
# 查看 JSON 结果
cat results/streaming_llm/wikitext_result.json

# 或使用 jq 格式化
cat results/streaming_llm/wikitext_result.json | jq .
```

---

## 📊 运行实验

### 方式 1: 使用一键脚本 (推荐)

```bash
# 给脚本添加执行权限 (首次运行)
chmod +x run_everything.sh run_decoding_latency.sh

# 运行所有主实验 (约 25 分钟)
./run_everything.sh

# 运行 decoding latency 实验 (约 20 分钟)
./run_decoding_latency.sh
```

### 方式 2: 使用 Python 脚本

```bash
# 运行所有实验并自动生成图表
python experiments/run_all_experiments.py
```

### 方式 3: 单独运行实验

#### Baseline 实验

```bash
# WikiText-103 Baseline
python experiments/eval_streaming_llm.py \
  --dataset-name wikitext \
  --dataset-config wikitext-103-v1 \
  --max-samples 64 \
  --max-eval-tokens 4096 \
  --n-sink 0 \
  --window-size 999999 \
  --output results/streaming_llm/wikitext_baseline.json

# PG19 Baseline
python experiments/eval_streaming_llm.py \
  --dataset-name pg19 \
  --max-samples 1 \
  --max-eval-tokens 4096 \
  --n-sink 0 \
  --window-size 999999 \
  --trust-remote-code \
  --output results/streaming_llm/pg19_baseline.json
```

#### StreamingLLM 实验

```bash
# WikiText-103 StreamingLLM
python experiments/eval_streaming_llm.py \
  --dataset-name wikitext \
  --dataset-config wikitext-103-v1 \
  --max-samples 64 \
  --max-eval-tokens 4096 \
  --n-sink 4 \
  --window-size 1024 \
  --output results/streaming_llm/wikitext_result.json

# PG19 StreamingLLM
python experiments/eval_streaming_llm.py \
  --dataset-name pg19 \
  --max-samples 1 \
  --max-eval-tokens 4096 \
  --n-sink 4 \
  --window-size 1024 \
  --trust-remote-code \
  --output results/streaming_llm/pg19_result.json
```

#### 消融实验

```bash
# Window Size 消融
python experiments/ablation_study.py \
  --ablation-type window_size \
  --output results/streaming_llm/ablation_window_size.json

# N_sink 消融
python experiments/ablation_study.py \
  --ablation-type n_sink \
  --output results/streaming_llm/ablation_n_sink.json
```

#### 对比实验 (与 kvpress 对比)

```bash
# WikiText 对比
python experiments/run_comparison.py --dataset wikitext

# PG19 对比
python experiments/run_comparison.py --dataset pg19

# 生成对比图表
python experiments/plot_comparison.py
```

### 生成可视化图表

```bash
# 生成所有图表
python experiments/generate_final_figures.py

# 或单独生成
python experiments/plot_results.py
python experiments/plot_comparison.py
```

---

## 🚀 使用一键脚本

### run_everything.sh - 主实验脚本

**包含内容**:
- ✅ Baseline 实验 (WikiText-103 & PG19)
- ✅ 我们的 StreamingLLM 实验
- ✅ kvpress 官方库对比实验
- ✅ 消融实验 (Window Size & N_sink)
- ✅ 自动生成可视化图表
- ✅ 详细的日志和总结报告

**输出文件**:
- 实验结果: `results/streaming_llm/`, `results/kvpress/`
- 可视化图表: `results/figures/`
- 日志文件: `results/experiment_log_*.txt`
- 总结报告: `results/experiment_summary_*.txt`

### run_decoding_latency.sh - Decoding Latency 实验

**功能**:
- 测量 per-token decoding latency
- 对比 Baseline 和 StreamingLLM
- 多种 cache size 配置 (512, 1024, 2048, 4096)
- 长序列测试 (5000 tokens)
- 不同 n_sink 配置测试

**特点**:
- GPU warmup (200 tokens)
- 多次运行取平均 (3 runs)
- 精确计时 (torch.cuda.synchronize)
- 只统计 cache 填满后的 tokens

**输出文件**:
- 结果文件: `results/decoding_latency_*.json`
- 日志文件: `results/decoding_latency_log_*.txt`
- 总结报告: `results/decoding_latency_summary_*.txt`

### 查看生成的图表

```bash
# 图表保存在 results/figures/ 目录
ls results/figures/

# 输出:
# - main_comparison.png          # 主实验对比 (PPL, 加速比, 压缩比, Runtime)
# - ablation_window_size.png     # Window Size 消融
# - ablation_n_sink.png          # N_sink 消融
# - results_summary.png          # 结果总结表格
# - implementation_comparison.png # 实现对比
# - comparison_metrics_table.png # 对比指标表格
```

---

## 🔍 实验脚本参数说明

### eval_streaming_llm.py

主评估脚本,支持多种配置:

**参数说明**:
- `--model-name`: 模型名称 (默认: EleutherAI/pythia-70m)
- `--dataset-name`: 数据集名称 (wikitext, pg19)
- `--dataset-config`: 数据集配置 (如 wikitext-103-v1)
- `--max-samples`: 最大样本数
- `--max-eval-tokens`: 最大评估 token 数
- `--n-sink`: Sink token 数量 (默认: 4)
- `--window-size`: 滑动窗口大小 (默认: 1024)
- `--output`: 输出文件路径
- `--trust-remote-code`: 信任远程代码 (PG19 需要)

### ablation_study.py

消融实验脚本:

**参数说明**:
- `--ablation-type`: 消融类型 (window_size, n_sink)
- `--output`: 输出文件路径

### run_comparison.py

对比实验脚本 (我们的实现 vs kvpress):

**参数说明**:
- `--dataset`: 数据集选择 (wikitext, pg19)
- `--n-sink`: Sink token 数量
- `--window-size`: 滑动窗口大小
- `--max-samples`: 最大样本数
- `--max-eval-tokens`: 最大评估 token 数

---

## 🐛 故障排除

### 问题 1: 环境未激活

**错误信息**:
```
ModuleNotFoundError: No module named 'transformers'
```

**解决方案**:
```bash
# 确保激活了 kvpress 的虚拟环境
cd kvpress
source .venv/bin/activate
cd ..
```

### 问题 2: CUDA 内存不足

**错误信息**:
```
RuntimeError: CUDA out of memory
```

**解决方案**:
- 减少 `--max-eval-tokens` (如 4096 → 2048)
- 减少 `--max-samples` (如 64 → 32)
- 使用 CPU: 脚本会自动检测并使用 CPU

### 问题 3: 数据集下载慢

**解决方案**:
```bash
# 使用镜像 (如果在国内)
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载数据集到缓存目录
```

### 问题 4: Python 解释器找不到

**错误信息**:
```
错误: 找不到 Python 解释器 kvpress/.venv/bin/python
```

**解决方案**:
```bash
# 创建虚拟环境
cd kvpress
UV_CACHE_DIR=$PWD/.cache/uv uv sync --all-groups
UV_CACHE_DIR=$PWD/.cache/uv uv sync --extra eval
cd ..
```

### 问题 5: PG19 下载失败

**错误信息**:
```
Failed to download PG19 dataset
```

**解决方案**:
```bash
# 清理缓存重试
rm -rf data/pg19/
./run_everything.sh
```

---

## 📁 结果文件说明

### 目录结构

```
results/
├── streaming_llm/              # 我们的实现结果
│   ├── wikitext_baseline.json
│   ├── wikitext_result.json
│   ├── pg19_baseline.json
│   ├── pg19_result.json
│   ├── ablation_window_size.json
│   └── ablation_n_sink.json
├── kvpress/                    # kvpress 官方库结果
│   ├── wikitext_result.json
│   └── pg19_result.json
├── figures/                    # 可视化图表
│   ├── main_comparison.png
│   ├── ablation_window_size.png
│   ├── ablation_n_sink.png
│   ├── results_summary.png
│   ├── implementation_comparison.png
│   └── comparison_metrics_table.png
├── decoding_latency_*.json     # Decoding latency 结果
├── experiment_log_*.txt        # 实验日志
└── experiment_summary_*.txt    # 实验总结
```

### JSON 文件格式

每个 JSON 结果文件包含:

```json
{
  "model": "EleutherAI/pythia-70m",
  "dataset": "wikitext:wikitext-103-v1",
  "baseline": {
    "perplexity": 40.31,
    "runtime_sec": 0.401
  },
  "streaming": {
    "perplexity": 40.31,
    "runtime_sec": 0.032
  },
  "metrics": {
    "speedup": 12.4,
    "compression_ratio": 0.70,
    "ppl_increase_percent": 0.0
  }
}
```

---

## 📈 预期运行时间

基于 NVIDIA GPU (如 RTX 3090):

| 实验 | 预计时间 |
|------|---------|
| WikiText-103 Baseline | ~2 分钟 |
| WikiText-103 StreamingLLM | ~1 分钟 |
| PG19 Baseline | ~1 分钟 |
| PG19 StreamingLLM | ~30 秒 |
| kvpress 对比实验 | ~2 分钟 |
| Window Size 消融 | ~10 分钟 |
| N_sink 消融 | ~8 分钟 |
| 可视化生成 | ~5 秒 |
| **总计** | **~25 分钟** |

Decoding Latency 实验:

| 实验 | 预计时间 |
|------|---------|
| 单个配置 (3 runs) | ~3 分钟 |
| 所有配置 | ~20 分钟 |

---

## 🎓 使用示例

### 示例 1: 测试不同的 window_size

```bash
# window_size = 512
python experiments/eval_streaming_llm.py \
  --window-size 512 \
  --output results/streaming_llm/window_512.json

# window_size = 2048
python experiments/eval_streaming_llm.py \
  --window-size 2048 \
  --output results/streaming_llm/window_2048.json
```

### 示例 2: 测试不同的 n_sink

```bash
# n_sink = 0 (无 sink)
python experiments/eval_streaming_llm.py \
  --n-sink 0 \
  --output results/streaming_llm/n_sink_0.json

# n_sink = 8
python experiments/eval_streaming_llm.py \
  --n-sink 8 \
  --output results/streaming_llm/n_sink_8.json
```

### 示例 3: 运行完整对比实验

```bash
# 运行 WikiText 对比
python experiments/run_comparison.py --dataset wikitext

# 运行 PG19 对比
python experiments/run_comparison.py --dataset pg19

# 生成对比图表
python experiments/plot_comparison.py

# 查看结果
ls results/streaming_llm/*_comparison.json
ls results/kvpress/*_comparison.json
ls results/figures/implementation_comparison.png
```

---

## 💡 提示

- 首次运行会下载模型和数据集,需要一些时间
- 建议先运行 `test_streaming_llm.py` 确保环境正确
- 使用 `--max-eval-tokens` 控制实验时间
- 所有脚本都支持 `--help` 查看完整参数
- PG19 数据集会流式下载一条样本并保存到本地 (`data/pg19/sample.json`)
- 后续运行直接使用本地缓存,无需重新下载

---

## 📝 下一步

1. ✅ 运行实验并收集结果
2. 📊 分析结果数据
3. 📈 查看可视化图表
4. 📄 阅读 README.md 了解详细结果
5. 🔍 查看 DESIGN.md 了解技术细节

---

## 📚 相关文档

- [README.md](README.md) - 项目总览和实验结果
- [DESIGN.md](DESIGN.md) - 技术设计文档
- [EXPERIMENT_SCRIPTS_GUIDE.md](EXPERIMENT_SCRIPTS_GUIDE.md) - 实验脚本详细说明
- [EXPERIMENT_VALIDATION_REPORT.md](EXPERIMENT_VALIDATION_REPORT.md) - 实验验证报告

---

**CS3602 NLP 大作业**