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

**全量矩阵（可选）**：`run_comprehensive_comparisons.sh` 会在 WikiText-103 (4k) 和 PG19 (20k) 上运行完整对比（chunked + decode-loop），输出到 `results/comprehensive/`。
```bash
chmod +x run_comprehensive_comparisons.sh
./run_comprehensive_comparisons.sh
```
完成后可用 `experiments/plot_fixed_eval_results.py` 或 `experiments/plot_comprehensive_results.py` 生成图表。

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

# 从 comprehensive 结果生成decode-loop对比图
python experiments/plot_comprehensive_results.py
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
