# Advanced Improvement Plan (Post-Streaming)

> 目标：在 **StreamingLLM 已经把 attention/KV 压到很低** 的前提下，进一步把 **TPOT/总时间** 再往下压，同时把 **PPL 增幅** 稳住（最好优于 SnapKV/SnapKV-like 的长期质量退化）。
>
> 本计划基于现有结论与动机，详见：
> - `docs/Optimize_R.md`
> - `docs/Optimize_R_2.md`
> - `docs/IMPROVEMENT.md`
> - `docs/LAZY_PRUNE_REPORT.md`（我们已验证 lazy/periodic prune 确实能加速）
>
> 如果你需要“可直接执行的工程作战书版本”，见 `docs/PLAN.md`。

---

## 0. 当前状态（作为出发点）

### 0.1 现象归因（已由实验支撑）
- Streaming 之后 decode 进入 **MLP + LN/残差 + kernel launch/框架** 主导。
- `compress_every=R`（lazy/periodic prune）能显著降低 TPOT，但存在：
  - **边际收益递减**（R 继续增大，速度几乎不涨）
  - **PPL 逐步变差**（R 过大可能出现“断崖式”质量崩溃）

### 0.2 我们已有的“可用 best points”
以 `window=2048, n_sink=4` 为例（A800）：
- Wikitext：`compress_every=32` TPOT 降到 ~14.39ms，PPL 增幅仍稳定在 ~3% 左右
- PG19：`compress_every=16/32` TPOT ~14.72/14.42ms，但 `compress_every=32` 的 PPL 增幅更大

**结论**：要“再快”不能只靠继续调大 R（收益小且 PPL 风险高），必须：
1) 把“驱逐/裁剪”从一次性硬砍改成 **受控、平滑、可解释** 的策略（稳 PPL）
2) 把每步 forward 的 **MLP/LN/残差/launch** 再压下去（抬上限）
3) 从中长期看，把 cache 管理做到更接近 **O(1)**（ring buffer / block-table / static cache）以消除 copy 成本

---

## 1. 总目标与成功判据（Go/No-go）

### 1.1 目标（按优先级）
1) **PG19 20k**：在 PPL 增幅 **≤ 5%**（或与 SnapKV 持平/更好）的前提下，TPOT **低于当前 best streaming（lazy prune）**。
2) **Wikitext 4k**：TPOT 继续降低，同时保持 PPL 增幅 ≤ 3%（主要用于 sanity 与回归）。
3) 模块化：所有改动必须 **可开关**（CLI flag / config），便于 ablation。

### 1.2 实验协议固定（避免“比较不公平”）
统一输出：TPOT、TTFT、total time、prefill time、PPL、peak mem。
- 数据集：wikitext=4096 tokens；pg19=20000 tokens
- 统计：TPOT = decode_time / decode_tokens（现已在 `experiments/eval_streaming_llm.py` 输出 `tpot_ms`）
- 每组实验至少跑 2 次，取中位数（避免抖动）

---

## 2. Phase A（1–2 天）：把“PPL 断崖”证据链做完整

> 目的：在动手加新策略前，先把“为什么 R 大 PPL 会炸”的机制定位清楚，并让后续优化有可度量的目标（压尖峰）。

### A1) 记录 token-level NLL 并标出 prune 边界
交付物：
- 新脚本：`experiments/diagnose_prune_nll.py`（建议）
  - 输出：每 token NLL 序列、prune step 列表、NLL 尖峰统计（peak/width/area）
- 图：NLL vs token index（叠加 prune 边界竖线）

成功判据：
- 能看到 **prune 边界后 NLL 尖峰**（或明确证明没有尖峰但 PPL 仍变差 → 进入 Phase A2 排查“有效窗口漂移”）

### A2) 区分两种质量劣化来源
1) **断崖尖峰**：每次 prune 丢太多，引发分布突变
2) **有效窗口漂移**：R 周期内 KV 会增长，模型“有时看到 W，有时看到 W+R”，导致规则不一致

交付物：
- 在日志中输出每步实际可见 KV 长度（min/mean/max）与 prune 后 drop 数量分布

---

## 3. Phase B（2–4 天）：受控驱逐（保持加速，显著稳住 PPL）

> 核心目标：不牺牲太多 TPOT 的前提下，把 PPL 从 “R=32 的 +6%/R=128 的 +34%” 这类风险点拉回到可接受区间。
> 这些策略优先级高、工程 ROI 高，也最适合写进论文 “Method”。

### B1) max_drop：限制单次最大丢弃量（最推荐优先做）
动机：阻止 “一次性丢太多 token” 的灾难事件。

实现要点（建议设计成 pluggable policy）：
- 新参数：`max_drop`（例如 256/512）
- 当触发 prune 时：
  - 如果 overflow 很大，不一次 prune 到位，而是多次小 prune（或进入“pruning mode”持续 K 步）

实验矩阵（最小但信息量大）：
- 固定 `window=2048, sink=4`
- `compress_every ∈ {32, 64, 128}`
- `max_drop ∈ {None, 256, 512}`

成功判据：
- 与同 R 的 plain lazy prune 相比：
  - PPL 明显下降（优先看 PG19）
  - TPOT 下降不超过 ~1–2%（或甚至更快）

### B2) overlap：每次驱逐保留额外重叠（平滑上下文变化）
动机：让模型在 prune 后仍能看到一段“过渡上下文”，减少断崖。

实现建议：
- 新参数：`overlap`（128/256）
- 逻辑保留长度变为：`sink + window + overlap`
- 注意：overlap 作为“常数额外 attention 长度”，TPOT 可能略有回升，但在 MLP-dominated 下通常可控。

实验矩阵：
- 固定 `compress_every=32`
- `overlap ∈ {0, 128, 256}`
- 对比 PPL 改善 vs TPOT 损失

成功判据：
- PPL 改善显著（尤其 PG19），TPOT 损失 ≤ 3%

### B3) tiny refresh：在驱逐前把“摘要 token”搬入动态 sink
动机：在不训练的前提下，把要丢弃段落的关键信息“留个摘要”，对长文本质量特别重要（目标：优于 SnapKV 的长期退化）。

实现建议（由易到难）：
1) 均匀采样 refresh（M=8/16）
2) 结构 token refresh（换行/句号/引号等）
3) 基于不确定性 proxy（entropy/top1 prob）挑选 token

工程要点：
- 设计一个 `RefreshPolicy`，不改 attention kernel，只改 cache 的“保留 token 集合”
- 维持总保留预算：`sink + refresh + window`（可与 overlap 组合）

实验矩阵：
- 固定 `compress_every=64` 或 `128`
- `refresh_budget ∈ {0, 8, 16}`
- `sink ∈ {4, 16, 32}`（refresh 存放区也可算入“动态 sink”）

成功判据：
- 在较大 R 下 PPL 显著回落（目标：PG19 ≤ 5%）
- TPOT 基本不变（refresh 只涉及少量 KV copy）

### B4) 自适应 R（slack-based / entropy-based）
动机：避免固定大 R 在“危险段落”踩雷。

实现建议：
- slack-based：用 `extra_slack=Δ` 代替固定 R
  - past 长度超过 `sink+window+Δ` 就 prune（Δ 建议 48/64 起步）
- entropy-based：风险高就提前 prune；风险低就放宽

成功判据：
- 平均 TPOT 接近大 R
- PPL 接近小 R（或优于）

---

## 4. Phase C（3–6 天）：把 cache 管理从 O(W) copy 推向 O(1)（冲击速度上限）

> Phase B 的目标是“保住 7.1–7.3× 同时把 PPL 拉稳”。Phase C 的目标是“把上限再抬高”，不再依赖大 R。

### C1) Ring buffer（最现实的 O(1) 方向）
目标：把“裁剪=拷贝”变成“裁剪=移动指针”。

可落地的分层实现（从易到难）：
1) **软 ring（wrap 时才 copy）**：只有发生 wrap-around 时 `torch.cat` 一次；平时仅 slice/view
2) 真 ring（完全无 cat）：需要 attention 支持分段读取（风险更高，DDL 前谨慎）

成功判据：
- 在不增大 PPL 的前提下，TPOT 进一步下降（尤其在大 window/长文本时）
- 让“R 的作用”变弱（因为几乎不需要 prune copy）

### C2) Static cache / 预分配（避免每步 append 时的隐式拷贝）
若 transformers 的 Cache append 仍存在动态扩容拷贝，可尝试：
- 使用 static/paged 风格的 cache 实现（或自行实现 Cache subclass）
- 目标：避免 “每步增长 past -> 分配/拷贝” 的隐藏成本

成功判据：
- profiling 中 cache 相关 op 的时间占比明显下降

---

## 5. Phase D（并行战线，3–8 天）：压 MLP/LN/残差/launch（真正超越最快 streaming）

> 当 cache 管理成本被压下去后，TPOT 下限主要由 MLP/LN/残差决定。要超越“最快 streaming”，必须动这里。

### D1) fused LayerNorm / residual-add（优先级最高、风险最低）
目标：减少 elementwise kernels 和 launch。

实现路径（按侵入性）：
1) 先用 `torch.compile` + 固定 shape 尝试让 inductor 自动融合（可配合 CUDA graphs）
2) 若环境允许，引入 xFormers 的 fused layernorm/linear 组件进行替换

成功判据：
- TPOT 再下降 5–15%（以 long-text decode 为主）
- PPL 基本不变（仅有极小数值差异）

### D2) CUDA Graphs（对 batch=1/seq=1 很对症）
目标：把 Python/launch 开销进一步压下去。

关键前置条件：
- 形状稳定（Streaming + 固定 window 有利）
- 正确处理 step boundary（避免 cudagraph overwrite 类错误）

成功判据：
- TPOT 稳定下降（长文本更明显）

### D3) fused MLP（高潜力但风险更高）
目标：减少 MLP 中间张量读写和 kernel 数。

实现路径：
1) 先尝试现成 fused 实现（若环境允许）
2) 再考虑更重的引擎路线（TensorRT-LLM 等）作为对照或终极方案

成功判据：
- 在 PPL 稳定的前提下 TPOT 显著下降（>10%）

---

## 6. Phase E（收口）：对齐 SnapKV 等方法并形成“论文级结论”

### E1) 统一对比对象与图表
输出至少 2 张 Pareto：
- x=TPOT（或总时间），y=PPL 增幅（wikitext + pg19）
- 点包括：
  - base streaming（严格 prune）
  - lazy prune（R=16/32/64）
  - + max_drop / overlap / refresh / adaptive
  -（若能跑）SnapKV / H2O / Scissorhands 等

### E2) 给出明确“最佳推荐配置”
按应用场景给 2 套默认：
- 短文本优先（wikitext）：追 TPOT 极限
- 长文本优先（pg19）：追 TPOT 与 PPL 的最优折中（优先 PPL）

---

## 7. 建议的最小实验矩阵（避免爆炸式扫参）

> 每个阶段只做“信息量最大”的 8–12 个点，过线就进入下一阶段。

### (1) 断崖修复（Phase B 的核心 8 点）
- 固定：`window=2048, sink=4`
- `compress_every ∈ {32, 64}`
- `max_drop ∈ {None, 256}`
- `refresh_budget ∈ {0, 16}`

### (2) 抬上限（Phase D 的核心 4 点）
- 固定最优配置（来自 Phase B）
- `fused_ln ∈ {off, on}`
- `cudagraph ∈ {off, on}`

成功标准：
- PG19：PPL ≤ 5% 且 TPOT 低于当前 best（lazy prune）
- 通过则再考虑 Phase C 的 ring/static cache 作为“终极冲上限”方案

---

## 8. 风险与止损点（必须写清楚）
1) **Ring/paged**：可能需要自定义 kernel，DDL 风险高 → 作为 Phase C 可选冲刺，不作为主线交付依赖。
2) **fused MLP**：依赖外部实现/环境，可能集成困难 → 先做 fused LN + CUDA graphs（更稳）。
3) **refresh 策略**：过度复杂会拖慢速度 → 先用均匀采样（M=8/16），确认有效再升级策略。

---

## 9. 最终愿景（论文叙事）
我们要讲清楚一个“系统性发现 → 方法设计 → 实验验证”的闭环：
1) Streaming 后 attention 不再主导，copy/prune/launch 成为瓶颈（证据：window sweep + lazy prune）
2) 纯粹增大 R 虽能加速，但会引入周期性上下文断崖/窗口漂移导致 PPL 崩（证据：NLL 尖峰）
3) 我们提出受控驱逐（max_drop/overlap/refresh/adaptive）恢复质量，同时保留加速
4) 在此基础上，通过 fused LN/graphs（以及可选 ring/static cache）进一步抬高速度上限
