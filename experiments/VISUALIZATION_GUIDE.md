# StreamingLLM 可视化指南

本指南介绍如何使用 [`generate_final_figures.py`](generate_final_figures.py:1) 生成专业的实验结果图表。

## 快速开始

### 一键生成所有图表

```bash
python experiments/generate_final_figures.py
```

**输出**:
- ✅ `results/figures/main_comparison.png` - 主实验对比 (2×2)
- ✅ `results/figures/ablation_window_size.png` - Window Size 消融 (2×1)
- ✅ `results/figures/ablation_n_sink.png` - Sink Tokens 消融 (2×1)
- ✅ `results/figures/results_summary.png` - 综合结果表格

## 生成的图表详解

### 1. 主实验对比图 (main_comparison.png)

**2×2 布局,包含 4 个子图:**

#### (a) Perplexity 对比
- **类型**: 分组柱状图
- **数据集**: WikiText-103, PG19
- **对比**: Baseline vs StreamingLLM
- **关键发现**: PPL 完全相同,无质量损失

#### (b) Runtime 对比
- **类型**: 分组柱状图
- **单位**: 毫秒 (ms)
- **关键发现**: StreamingLLM 显著更快

#### (c) 加速比
- **类型**: 柱状图
- **显示**: 相对于 Baseline 的倍数
- **结果**: 
  - WikiText-103: **10.02×**
  - PG19: **5.07×**

#### (d) 压缩比
- **类型**: 柱状图
- **单位**: 百分比 (%)
- **结果**:
  - WikiText-103: **69.83%**
  - PG19: **87.45%**

---

### 2. Window Size 消融图 (ablation_window_size.png)

**2×1 布局,测试不同窗口大小的影响:**

#### (a) PPL vs Window Size
- **主轴**: Perplexity (蓝色)
- **次轴**: Compression Ratio (橙色)
- **测试值**: 128, 256, 512, 1024, 2048, 4096
- **最佳点**: 用红色星号标注
- **发现**: 所有窗口大小保持相同 PPL

#### (b) Runtime vs Window Size
- **显示**: 运行时间随窗口大小的变化
- **发现**: 运行时间相对稳定
- **推荐**: Window Size = 1024 (平衡性能)

---

### 3. Sink Tokens 消融图 (ablation_n_sink.png)

**2×1 布局,测试 sink tokens 数量的影响:**

#### (a) PPL vs Sink Tokens
- **主轴**: Perplexity (蓝色)
- **次轴**: Compression Ratio (橙色)
- **测试值**: 0, 1, 2, 4, 8, 16
- **特别标注**: n_sink=0 的情况 (红色 X)
- **发现**: Sink tokens 不影响 PPL

#### (b) Runtime vs Sink Tokens
- **显示**: 运行时间随 sink tokens 的变化
- **关键发现**: n_sink=0 时延迟显著增加
  - n_sink=0: ~333ms
  - n_sink≥1: ~32ms
- **推荐**: n_sink = 4

---

### 4. 综合结果表格 (results_summary.png)

**专业表格,包含所有关键指标:**

| 指标 | WikiText Baseline | WikiText Streaming | PG19 Baseline | PG19 Streaming |
|------|-------------------|-------------------|---------------|----------------|
| Perplexity | 40.31 | 40.31 | 59.49 | 59.49 |
| Runtime (ms) | 326.74 | 32.61 | 376.85 | 74.28 |
| Speedup | 1.00× | 10.02× | 1.00× | 5.07× |
| Compression | 0.00% | 69.83% | 0.00% | 87.45% |
| PPL Increase | - | 0.00% | - | 0.00% |

**配置信息**:
- Model: EleutherAI/pythia-70m
- Window Size: 1024
- Sink Tokens: 4
- Device: CUDA
- Dtype: torch.float16

## 图表特性

### 视觉设计
- ✅ **统一配色**: 专业学术风格
- ✅ **高分辨率**: 300 DPI (适合打印)
- ✅ **清晰标注**: 所有关键数值都有标签
- ✅ **网格线**: 提高可读性
- ✅ **图例**: 清晰的说明
- ✅ **阴影效果**: 专业外观

### 配色方案
```python
Baseline:    #2E86AB (深蓝色)
StreamingLLM: #A23B72 (紫红色)
Accent:      #F18F01 (橙色)
Success:     #06A77D (绿色)
```

### 技术规格
- **格式**: PNG
- **分辨率**: 300 DPI
- **文件大小**: ~0.3 MB/图
- **总大小**: ~1.2 MB

## 使用场景

### 📄 学术论文
```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=\textwidth]{results/figures/main_comparison.png}
  \caption{StreamingLLM performance comparison on WikiText-103 and PG19 datasets.}
  \label{fig:main_comparison}
\end{figure}
```

### 📊 演示文稿
- 直接插入 PowerPoint/Keynote
- 高分辨率确保投影清晰
- 配色适合深色/浅色背景

### 📝 技术报告
- 插入 Word/Markdown 文档
- 图表自带标题和说明
- 专业外观

## 自定义选项

### 修改分辨率

编辑 [`generate_final_figures.py`](generate_final_figures.py:35):
```python
DPI = 300  # 改为 150 (预览) 或 600 (超高质量)
```

### 修改配色

编辑 [`generate_final_figures.py`](generate_final_figures.py:27):
```python
COLORS = {
    'baseline': '#YOUR_COLOR',
    'streaming': '#YOUR_COLOR',
    'accent': '#YOUR_COLOR',
    'success': '#YOUR_COLOR',
}
```

### 修改图表大小

在各个绘图函数中:
```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # 宽×高 (英寸)
```

## 数据要求

脚本自动从以下文件读取数据:

```
results/streaming_llm/
├── wikitext_result.json      # WikiText-103 主实验
├── pg19_result.json           # PG19 主实验
├── ablation_window_size.json  # Window Size 消融
└── ablation_n_sink.json       # Sink Tokens 消融
```

**数据格式示例**:
```json
{
  "model": "EleutherAI/pythia-70m",
  "dataset": "wikitext:wikitext-103-v1",
  "baseline": {
    "perplexity": 40.31,
    "runtime_sec": 0.327
  },
  "streaming": {
    "perplexity": 40.31,
    "runtime_sec": 0.033
  },
  "metrics": {
    "speedup": 10.02,
    "compression_ratio": 0.698
  }
}
```

## 依赖项

```bash
pip install matplotlib seaborn numpy
```

**版本要求**:
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- numpy >= 1.21.0

## 故障排除

### 问题 1: 找不到数据文件
**错误**: `FileNotFoundError: results/streaming_llm/xxx.json`

**解决方案**: 先运行实验生成数据
```bash
python experiments/run_final_experiments.py
```

### 问题 2: 中文显示异常
**症状**: 中文显示为方框

**解决方案**: 安装中文字体
```bash
# Ubuntu/Debian
sudo apt-get install fonts-wqy-zenhei

# 或修改脚本使用英文
```

### 问题 3: 内存不足
**症状**: `MemoryError`

**解决方案**: 降低 DPI
```python
DPI = 150  # 从 300 降低到 150
```

### 问题 4: 图表重叠
**症状**: 标签或图例重叠

**解决方案**: 已使用 `tight_layout()` 自动处理,如仍有问题可调整 `figsize`

## 输出示例

运行脚本后的输出:
```
============================================================
StreamingLLM 实验结果可视化
============================================================
✓ 图表保存目录: results/figures

生成主实验对比图...
✓ 已保存: results/figures/main_comparison.png

生成Window Size消融图...
✓ 已保存: results/figures/ablation_window_size.png

生成N_sink消融图...
✓ 已保存: results/figures/ablation_n_sink.png

生成综合结果表格...
✓ 已保存: results/figures/results_summary.png

============================================================
✓ 所有图表生成完成!
✓ 保存位置: results/figures
============================================================

生成的图表文件:
  - ablation_n_sink.png (0.35 MB)
  - ablation_window_size.png (0.32 MB)
  - main_comparison.png (0.30 MB)
  - results_summary.png (0.24 MB)
```

## 最佳实践

### ✅ 推荐做法
1. 先运行实验生成数据
2. 使用默认 300 DPI 生成图表
3. 检查图表质量
4. 根据需要调整配色/大小
5. 重新生成

### ❌ 避免做法
1. 不要手动编辑 PNG 文件
2. 不要使用过低的 DPI (<150)
3. 不要修改数据文件格式
4. 不要在没有数据时运行脚本

## 进阶使用

### 批量生成不同配置

```python
# 修改 DPI 列表
for dpi in [150, 300, 600]:
    DPI = dpi
    plot_main_comparison()
    # 重命名输出文件
```

### 生成单个图表

```python
from experiments.generate_final_figures import *

setup_figure_dir()
plot_main_comparison()  # 只生成主对比图
```

### 自定义数据源

```python
# 修改 RESULTS_DIR
RESULTS_DIR = Path("custom/path/to/results")
```

## 相关文档

- 📖 [图表详细说明](../results/figures/README.md)
- 📊 [实验运行指南](run_final_experiments.py)
- 📈 [结果分析](../results/streaming_llm/)

## 更新日志

- **v1.0** (2024-12-04)
  - 初始版本
  - 4 个主要图表
  - 专业学术风格
  - 300 DPI 高分辨率

## 贡献

欢迎提交改进建议!

## 许可证

与主项目相同。