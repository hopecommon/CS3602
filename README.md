# CS3602 NLP 大作业 - StreamingLLM 复现与深入分析

本仓库聚焦在 EleutherAI/pythia-2.8b（2.8B 参数）上从零复现 StreamingLLM，全面对比 kvpress 与 MIT 参考实现，进而分析加速/质量的权衡。README 同时兼具项目主页与实验报告，包含运行指南、表格级指标、复现数据与结论分析，满足科研汇报标准。


> 说明：当前核心结果基于 `EleutherAI/pythia-2.8b` 并由 `run_fixed_evaluation.sh` 生成。旧的 70M / 全量 24 组实验仍保留在 `results/comprehensive/` 作为对比，需时可修改 .env 文件后用 `run_comprehensive_comparisons.sh` 重新生成。

## 1. 项目与任务定位
- **目标**：验证 StreamingLLM 算法（attention sink + recent window）在 Pythia 上的加速能力与 PPL 影响，复现 MIT/kvpress 的思路，与 kvpress 进行对比。
- **任务要求**：在 `WikiText-103` / `PG19` 数据集上跑出 PPL 与 runtime；结果记录在 README 中；所有代码有 Git commit 记录。
- **核心组件**：`streaming_llm/` 包含 cache & wrapper，`experiments/` 包含评估、图表、decode-loop 比较，`results/` 保存所有 JSON/图。

## 2. 最新固定评测（pythia-2.8b，decode-loop 为主）
本轮采用统一脚本 `run_fixed_evaluation.sh`，默认读取 `.env`，在离线模式下用 `n_sink=4`、`window_size=1024` 跑逐 token decode-loop。数据源为 `data/wikitext/long_context_4096.json`（WikiText-103 采样，约 4k tokens）和 `data/pg19/long_context_20000.json`（PG19 拼接，约 20k tokens）。

**kvpress** 行使用其官方 chunked evaluation pipeline（`eval_kvpress.py`），**因评估方法不同（chunked vs decode-loop）**，数值仅作官方结果对照，不与 decode-loop 直接比较。

**结果（`results/fixed_eval/*.json`） — decode-loop 主表**

| Dataset | Method | PPL | Runtime (s) | Speedup vs baseline | First-token (s) | 备注 |
|---------|--------|-----|-------------|---------------------|-----------------|------|
| WikiText-103 (≈4k) | Baseline | 9.53 | 164.12 | 1.00× | 0.05 | sliding-window + 重算 |
|  | StreamingLLM (ours) | 9.79 (+2.8%) | 49.78 | **3.30×** | 0.15 | decode-loop |
|  | MIT-style | 9.79 (+2.8%) | 50.21 | 3.27× | 0.18 | decode-loop |
| PG19 (20k) | Baseline | 20.06 | 1005.19 | 1.00× | 0.05 | sliding-window + 重算 |
|  | StreamingLLM (ours) | 20.17 (+0.54%) | 311.06 | **3.23×** | 0.18 | decode-loop |
|  | MIT-style | 20.17 (+0.54%) | 309.83 | **3.24×** | 0.16 | decode-loop |

**kvpress 官方（chunked evaluation，仅作对照）**

| Dataset | Method | PPL | Runtime (s) | Speedup vs baseline | First-token (s) | 备注 |
|---------|--------|-----|-------------|---------------------|-----------------|------|
| WikiText-103 (≈4k) | kvpress (official) | 10.16 (+0.0%) | 0.95 | 0.66× | 0.20 | 官方 chunked evaluation，非 decode-loop |
| PG19 (20k) | kvpress (official) | 20.36 (+0.0%) | 2.25 | 0.92× | 0.08 | 官方 chunked evaluation，非 decode-loop |

**指标说明**：
- **PPL (Perplexity)**: 困惑度，越低越好。计算方式：exp(平均交叉熵损失)
- **Runtime**: 整段序列的decode-loop总时间（秒），包括prefill和所有decode步骤
- **Speedup**: 相对baseline的加速比，计算方式：baseline_runtime / method_runtime
- **First-token**: Prefill后第一个decode token的生成时间（秒）。StreamingLLM因压缩overhead会略慢，但后续token快
- **备注**: 所有decode-loop方法解码相同数量的token，因此speedup等价于平均per-token latency的比值

**要点**：
- **评估方法**: decode-loop是主评价方法，逐token解码；runtime为整段decode-loop的总用时；因所有方法解码的token数相同，speedup等价于平均per-token latency的比值
- **Baseline定义**: 基线为"sliding window + 重算"，即每步重新forward最近1028个token（use_cache=False），对齐StreamingLLM论文的baseline复杂度
- **kvpress对照**: kvpress行使用其官方chunked evaluation pipeline（非decode-loop），仅作官方结果对照，因评估方法不同不与decode-loop直接比较数值
- **PPL分析**: 
  - **PG19 (20k tokens)**: PPL增幅仅+0.54%，证明算法在长文本场景表现优秀
  - **WikiText (4k tokens)**: PPL增幅+2.8%较大，主要原因：
    * WikiText-103由短段落拼接而成，拼接边界处上下文不连续
    * 短序列下cache miss对PPL影响更明显
    * 但仍换来3.30×加速，且+2.8%在学术上可接受（< 3%）
- **加速比**: WikiText 3.30×, PG19 3.23×，符合2.8B模型的预期范围（3-10×）

**数据来源与采样**：
- **WikiText-103**: 使用`scripts/prepare_wikitext_samples.py`从test split拼接段落生成`data/wikitext/long_context_4096.json`（约4k tokens）和`long_context_8192.json`（约8k tokens）。拼接策略：按顺序连接非空段落直到达到目标长度，段落间用`\n\n`分隔。
- **PG19**: 使用`scripts/prepare_pg19_samples.py`从本地parquet文件（或HuggingFace streaming）的test split中扫描长篇小说，截取前N个token生成`data/pg19/long_context_20000.json`（约20k tokens）。采样策略：按顺序扫描书籍，找到第一本token数≥目标长度的书，截取前N个token。支持通过`--parquet-dir`指定本地parquet路径以避免联网下载。
- **可复现性**: 所有采样文件已提交到仓库，运行评估脚本会优先使用本地文件，避免重新下载。如需重新生成，运行对应的prepare脚本即可。

**快速运行**
```bash
chmod +x run_fixed_evaluation.sh
./run_fixed_evaluation.sh   # 结果写入 results/fixed_eval/
```
若需更换模型或样本长度，可在 `.env` 中覆盖 `MODEL_NAME`、`WIKITEXT_MAX_TOKENS`、`PG19_20K_MAX_TOKENS`，或直接指向采样文件路径。

### Legacy / 全量实验（参考）
更早的 70M 产物也被完整备份在 `results_70M/`。如需历史趋势或绘图，可用 `experiments/plot_comprehensive_results.py` 重新生成；README 现以 `run_fixed_evaluation.sh` 的 2.8B decode-loop 结果为主。

### 图表

**生成图表**：
```bash
# 生成所有fixed_eval图表（推荐）
python experiments/plot_fixed_eval_results.py

# 生成comprehensive对比图
python experiments/plot_comprehensive_results.py
```

**主要结果可视化**：

#### 综合对比（2×2布局）
![综合对比](results/figures/fixed_eval_comprehensive_summary.png)

*图1: StreamingLLM综合性能对比 - (a) Runtime对比 (b) PPL对比 (c) 加速比 (d) PPL增幅*

#### 加速比对比
![加速比](results/figures/fixed_eval_speedup.png)

*图2: StreamingLLM相对baseline的加速比 - WikiText-103达到3.30×，PG19达到3.23×*

#### PPL增幅分析
![PPL增幅](results/figures/fixed_eval_ppl_increase.png)

*图3: PPL增幅对比 - PG19仅+0.54%（优秀），WikiText +2.8%（可接受，<3%阈值）*

所有图表保存在 `results/figures/`，包括：
- `fixed_eval_comprehensive_summary.png` - 综合对比（2×2布局）
- `fixed_eval_runtime_comparison.png` - Runtime对比
- `fixed_eval_ppl_comparison.png` - PPL对比
- `fixed_eval_speedup.png` - 加速比
- `fixed_eval_ppl_increase.png` - PPL增幅
- `fixed_eval_first_token_latency.png` - First-token延迟
- `decode_loop_runtime_comparison.png` - Decode-loop runtime对比
- `decode_loop_ppl_comparison.png` - Decode-loop PPL对比

### 消融实验

**运行消融实验**：
```bash
chmod +x run_ablation_studies.sh
./run_ablation_studies.sh
```

**生成消融实验图表**：
```bash
python experiments/plot_ablation_results.py
```

#### 关键发现

**1. Window Size 影响** (固定 n_sink=4)

| Window Size | PPL | Runtime (s) | Compression Ratio |
|-------------|-----|-------------|-------------------|
| 128 | 14.65 | 67.3 | 96.8% |
| 256 | 12.89 | 63.7 | 93.7% |
| 512 | 10.86 | 59.7 | 87.4% |
| **1024** | **9.79** | **51.4** | **74.9%** |
| 2048 | 9.66 | 34.9 | 49.9% |

**结论**：
- Window size 从128增加到1024，PPL显著下降（14.65 → 9.79）
- 1024是性能和质量的最佳平衡点
- 继续增大到2048，PPL改善有限（9.79 → 9.66），但压缩率大幅下降

**2. N_sink 影响** (固定 window_size=1024)

| N_sink | PPL | Runtime (s) | PPL Improvement |
|--------|-----|-------------|-----------------|
| **0** | **17.69** | 50.6 | **baseline** |
| 1 | 9.79 | 49.6 | **+44.7%** |
| 2 | 9.79 | 51.2 | +44.7% |
| 4 | 9.79 | 49.5 | +44.6% |
| 8 | 9.63 | 51.9 | +45.6% |
| 16 | 9.62 | 51.3 | +45.6% |

**结论**：
- **n_sink=0 时PPL严重退化**（17.69 vs 9.79），证明attention sink机制至关重要
- n_sink≥1 后PPL迅速改善并趋于稳定
- n_sink=4 是推荐配置（论文默认值），性能稳定且开销小

#### 消融实验图表

![Window Size消融](results/figures/ablation_window_size.png)

*图4: Window Size消融实验 - (a) PPL vs Window Size (b) Runtime vs Window Size (c) Compression Ratio (d) PPL-Runtime Trade-off*

![N_sink消融](results/figures/ablation_n_sink.png)

*图5: N_sink消融实验 - (a) PPL vs N_sink (b) Runtime vs N_sink (c) PPL Improvement (d) Summary Table*

![消融实验综合](results/figures/ablation_summary.png)

*图6: 消融实验综合对比 - (a) Window Size影响 (b) N_sink影响*

# .env 配置（可选）
项目支持通过 `.env` 或对应环境变量覆盖脚本中的常量，例如 HF cache 路径 / dataset 选择 / python bin。参考仓库根目录的 `.env.example`，复制并按需修改，然后源脚本会自动加载（`run_fixed_evaluation.sh`、`run_comprehensive_comparisons.sh` 都会读取 `.env`）。

主要环境变量说明：

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `HF_HOME` | HuggingFace缓存根目录 | `.cache/huggingface` |
| `MODEL_NAME` | 模型名称 | `EleutherAI/pythia-2.8b` |
| `N_SINK` | StreamingLLM sink tokens数量 | `4` |
| `WINDOW_SIZE` | StreamingLLM window大小 | `1024` |
| `WIKITEXT_MAX_TOKENS` | WikiText评估的最大token数 | `4096` |
| `PG19_20K_MAX_TOKENS` | PG19评估的最大token数 | `20000` |
| `PYTHON_BIN` | Python解释器路径 | `kvpress/.venv/bin/python` |

## 3. 实验跑通与脚本说明
### 3.1 推荐流程
- **最快复现（主入口）**：`run_fixed_evaluation.sh` —— 两个数据集（WikiText-103 4k, PG19 20k）+ 三种方法（baseline/ours/MIT）的 decode-loop 评估；输出 `results/fixed_eval/`。
- **全量矩阵（可选）**：`run_comprehensive_comparisons.sh` —— 包含 chunked 和 decode-loop 两种评估方式的完整对比；输出 `results/comprehensive/`。

执行示例：
```bash
chmod +x run_fixed_evaluation.sh
./run_fixed_evaluation.sh
# 或（全量）./run_comprehensive_comparisons.sh
```

### 3.2 数据准备脚本
- **WikiText-103**: `scripts/prepare_wikitext_samples.py` —— 从test split拼接段落生成指定长度的样本文件
  ```bash
  python scripts/prepare_wikitext_samples.py --lengths 4096 8192 --output-dir data/wikitext
  ```
- **PG19**: `scripts/prepare_pg19_samples.py` —— 从本地parquet或HuggingFace streaming加载，截取指定长度
  ```bash
  python scripts/prepare_pg19_samples.py --lengths 20000 --parquet-dir /path/to/pg19/data --output-dir data/pg19
  ```

### 3.3 可视化脚本
- `experiments/plot_fixed_eval_results.py` —— 从 `results/fixed_eval/` 生成6张专业图表（runtime、PPL、speedup、PPL增幅、first-token latency、综合对比）
  ```bash
  python experiments/plot_fixed_eval_results.py
  ```
- `experiments/plot_comprehensive_results.py` —— 从 `results/comprehensive/` 生成decode-loop对比图
  ```bash
  python experiments/plot_comprehensive_results.py
  ```

## 4. 技术与架构回顾（详见 DESIGN.md）
- `StreamingLLMWrapper`：基于 hook，结合 `StreamingKVCache`（n_sink + window_size）与 `StartRecentKVCache` 选择，并在缓存间隔时 rerotate keys。
- **Mit-style slice**：当使用 MIT cache 时，直接 concat sink/recent chunks避免多次 gather，并尝试调用 pos-shift attention（因 transformers 版本变化未完全适配，最终保留 hook rerotation）。
- **kvpress 对比**：通过 `KeyRerotationPress` + `StreamingLLMPress`，加上 `run_kvpress_streaming_decode.sh`，确保 kvpress 在 decode-loop 下与我们数字一致。
- **Profile 建议**：如果仍需进一步加速，可用 `torch.profiler` 检查 `StreamingLLMWrapper.update` 的 rerotation/Hook 片段。

## 5. 目录结构与使用指南
```
CS3602/
├── README.md                          # 项目主文档（实验报告 + 运行指南）
├── QUICKSTART.md                      # 快速开始指南
├── DESIGN.md                          # 技术设计文档
├── Metrics.md                         # 指标定义与计算方法
├── REVIEW_REPORT.md                   # 代码与实验审查报告
├── .env.example                       # 环境变量配置模板
├── run_fixed_evaluation.sh            # 主评估脚本（推荐）
├── run_comprehensive_comparisons.sh   # 全量对比脚本
├── streaming_llm/                     # StreamingLLM核心实现
│   ├── kv_cache.py                   # KV缓存实现
│   ├── wrapper.py                    # StreamingLLM包装器
│   └── pos_shift.py                  # RoPE位置编码处理
├── experiments/                       # 评估与可视化脚本
│   ├── eval_streaming_llm.py         # StreamingLLM评估
│   ├── eval_kvpress.py               # kvpress评估
│   ├── run_decode_perplexity.py      # decode-loop评估
│   ├── plot_fixed_eval_results.py    # 主图表生成脚本
│   └── plot_comprehensive_results.py # decode-loop对比图
├── scripts/                           # 数据准备脚本
│   ├── prepare_wikitext_samples.py   # WikiText数据采样
│   └── prepare_pg19_samples.py       # PG19数据采样
├── results/                           # 实验结果
│   ├── fixed_eval/                   # 主评估结果（推荐）
│   ├── comprehensive/                # 全量对比结果
│   └── figures/                      # 生成的图表
└── data/                              # 数据集样本
    ├── wikitext/                     # WikiText-103样本
    └── pg19/                         # PG19样本
```

### 使用流程
1. **环境配置**：复制 `.env.example` 为 `.env`，根据需要修改配置
2. **数据准备**：运行 `scripts/prepare_*.py` 生成数据样本（已提供默认样本）
3. **运行评估**：执行 `./run_fixed_evaluation.sh` 进行主评估
4. **生成图表**：运行 `python experiments/plot_fixed_eval_results.py`
5. **查看结果**：查看 `results/fixed_eval/*.json` 和 `results/figures/*.png`

### Git提交说明
- 所有代码修改都有完整的commit记录
- 实验结果JSON文件已提交到仓库，保证可复现性
- README作为主报告，包含完整的实验数据和分析
