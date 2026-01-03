# 快速启动与运行指南

这个文档记录当前版本实验的标准运行流程、依赖说明和 legacy 脚本说明。请优先阅读并执行“推荐流程”;老脚本仅当需要和历史结果对比时才使用。

## 1. 环境准备（推荐复用 kvpress 虚拟环境）

```bash
cd ./CS3602
source kvpress/.venv/bin/activate

# 离线缓存（推荐）
export HF_HOME=$PWD/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"
```

> kvpress/.venv 已包含 torch/transformers/datasets 等依赖，无需另行 pip 安装。确保在每次运行前设置 HF cache 以避免网络请求。

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

## 1.1 可选 `.env`（共享配置）

在根目录复制 `.env.example` 为 `.env`，修改其中的 `HF_HOME`、`DATASET_NAME`、`PYTHON_BIN`、`N_SINK` 等字段即可，脚本会自动加载：

```
cp .env.example .env
# 编辑 .env 以适应当前环境
```

> `.env` 里新增的 `MODEL_NAME` 变量默认指向 `EleutherAI/pythia-2.8b`（该模型是本阶段的官方要求，运行后会在 `results/comprehensive/` 生成 2.8B 的 JSON）；若想保留旧的 70M 结果，可将 `MODEL_NAME` 改回 `EleutherAI/pythia-70m` 并将旧输出拷贝到 `results/legacy/`。

## 2. 推荐流程（当前主入口）

**最快复现**：`run_fixed_evaluation.sh`（decode-loop评估，对应 README 主表格）
```bash
chmod +x run_fixed_evaluation.sh
./run_fixed_evaluation.sh   # 结果输出到 results/fixed_eval/
```

如需额外方法（不默认运行）：
```bash
# 加上内部 mit-style slicing（仅用于调试对齐，非 MIT 官方 benchmark）
FIXED_EVAL_METHODS="baseline ours kvpress mit" ./run_fixed_evaluation.sh
```

**全量矩阵（可选）**：`run_comprehensive_comparisons.sh` 会在 WikiText-103 (4k) 和 PG19 (20k) 上运行完整对比（chunked + decode-loop），输出到 `results/comprehensive/`。
```bash
chmod +x run_comprehensive_comparisons.sh
./run_comprehensive_comparisons.sh
```
完成后可用 `experiments/plot_fixed_eval_results.py` 或 `experiments/plot_comprehensive_results.py` 生成图表。

### Windows 运行指南

- 使用 `PowerShell` 运行（推荐）：
  ```powershell
  # 运行主评估流程
  .\run_final_experiments.ps1
  ```
- 或直接用 Python（无需 Shell 脚本）：
  ```powershell
  python experiments/run_final_experiments.py --model-name EleutherAI/pythia-2.8b
  ```
- **运行完整论文实验 (Run Paper Experiments)**:
  ```powershell
  # 运行完整的论文实验流程（基线、对比、消融、参数扫描）并生成 LaTeX 表格
  .\run_paper_experiments.ps1
  ```
- 说明：
  - Windows 环境会自动检测 GPU 并回退到合适的注意力后端；若未安装 FlashAttention，代码会使用 PyTorch SDPA 的 math/flash 后端自适配。
  - 如需设置缓存路径，PowerShell 中可执行：`$env:HF_HOME = "$PWD\.cache\huggingface"`。
  - **环境激活**: 脚本会自动尝试检测 Conda 环境 `lora_env` 或 `kvpress/.venv`。如果您的环境不同，请先手动激活或设置 `$env:PYTHON_BIN`。

### 数据样本准备

**WikiText-103**：使用 `scripts/prepare_wikitext_samples.py` 生成拼接样本
```bash
python scripts/prepare_wikitext_samples.py \
  --lengths 4096 8192 \
  --output-dir data/wikitext
```

**PG19**：使用 `scripts/prepare_pg19_samples.py` 从本地parquet或HuggingFace生成
```bash
python scripts/prepare_pg19_samples.py \
  --parquet-dir /path/to/pg19/data \
  --split test \
  --lengths 20000 \
  --output-dir data/pg19
```

> 注意：仓库已包含默认样本文件（`data/wikitext/long_context_4096.json` 和 `data/pg19/long_context_20000.json`），可直接使用。如需重新生成或使用不同长度，运行上述脚本即可。

### 数据集角色说明
* **WikiText-103 (4k tokens)**：短到中等上下文评估，验证 StreamingLLM 在常规长度下的性能表现
* **PG19 (20k tokens)**：长上下文评估，测试超长序列下的加速效果和PPL质量

## 3. 辅助工具（按需调用）

**单独运行评估**：
```bash
# decode-loop评估
python experiments/run_decode_perplexity.py \
  --method ours \
  --dataset-name wikitext \
  --dataset-config wikitext-103-v1 \
  --max-eval-tokens 4096 \
  --output results/test_ours.json
```

**生成图表**：
```bash
# 从 fixed_eval 结果生成完整图表集
python experiments/plot_fixed_eval_results.py

# 生成 fixed_eval 汇总表（Markdown）
python experiments/summarize_fixed_eval_results.py

# 从 comprehensive 结果生成decode-loop对比图
python experiments/plot_comprehensive_results.py

# 运行消融实验（会自动生成消融图表到 results/figures/）
chmod +x run_ablation_studies.sh
./run_ablation_studies.sh
```

**MIT 官方吞吐/显存 benchmark（非 PPL）**：
```bash
#
# 注意：MIT 官方 repo 依赖较老版本的 transformers/huggingface-hub，建议用独立环境运行。
# 例如先按 mit-streaming-llm/README.md 建一个 conda env，然后把 python 路径传给 MIT_BENCH_PYTHON。
#
python experiments/run_mit_official_benchmark.py \
  --model-name-or-path /path/to/local/hf/snapshot_or_model_dir \
  --data-json data/pg19/long_context_20000.json \
  --output results/mit_official/pg19_20k_benchmark.json

# 或在主脚本里启用（推荐）
RUN_MIT_OFFICIAL_BENCHMARK=1 MIT_BENCH_PYTHON=/path/to/mit_env/bin/python ./run_fixed_evaluation.sh
```

**快速测试**：
```bash
python experiments/test_streaming_llm.py  # smoke test
```

## 4. Legacy 脚本（仅限对比 / 参考）

1. `run_everything.sh`：旧版一键实验（baseline + streaming + kvpress + ablations），依然可用但结果生成方式与当前推荐流程不同，现标注为 **Legacy**。
2. `run_decoding_latency.sh`：旧版 decoding latency 评估。（仍可运行）
3. `experiments/run_all_experiments.py`：已停止维护，建议改用 shell 脚本。

## 5. 结果复现要求

- 主评估结果保存在 `results/fixed_eval/`（8个JSON文件：2数据集 × 4方法）
- 全量对比结果保存在 `results/comprehensive/`（16个JSON文件）
- 图表在 `results/figures/`，可用 `plot_fixed_eval_results.py` 重新生成
- 实验报告在 README.md 中，包含完整的数据表格和分析
