# Execution Plan: Beyond StreamingLLM (Speed ↑, PPL Stable)

目标：在 **StreamingLLM 已经把 attention/KV 压到很低** 的前提下，进一步提升 **TPOT/总时间**，并把 **PPL 增幅** 稳住（目标：PG19 优于或不逊于 SnapKV 类方法）。

参考与已知证据：
- `docs/LAZY_PRUNE_REPORT.md`：`compress_every` 增大可显著提升 TPOT，但存在收益递减与 PPL 风险。
- `docs/Optimize_R.md` / `docs/Optimize_R_2.md`：R 大导致“上下文断崖/窗口漂移”的机制与候选修复策略。
- `docs/IMPROVEMENT.md`：ROI 排序（先稳 PPL，再抬上限；soft ring 优于硬啃 paged kernel）。

---

## 0) 基线锁死（公平性与可复现）

### 0.1 固定模型/数据/协议
- 模型：`EleutherAI/pythia-2.8b`
- Dtype：默认 `float16`（必要时记录 TF32 开关）
- 数据：
  - Wikitext：4096 tokens（用于 sanity + 回归）
  - PG19：20000 tokens（主结论）
- 统计口径（必须一致）：
  - `TPOT(ms)` = `decode_time_sec / decode_tokens * 1000`
  - **steady-state TPOT**：丢掉前 `warmup_tokens=64` 的 per-token time 后再统计均值
  - **P50/P90 TPOT**：记录 per-token time 分布（lazy prune/受控驱逐会有周期性尖峰）
  - `TTFT`、`prefill_sec`、`total_sec`、`peak_memory_mb`、`PPL`

### 0.2 环境/版本锁死（每次实验都记录）
每次实验输出都必须包含元信息（用于防止“环境差异污染结论”）：
- git commit hash
- `pip freeze`（或关键包版本：torch/transformers/cuda/xformers/flash-attn 等）
- GPU 型号/driver/cuda runtime
- dtype、TF32 开关、seed
- 运行参数（window/sink/R/max_drop/overlap/refresh 等）

建议落地方式：
- 每次写结果 JSON 时同时写一个 `metadata` 字段（或 `results/**/metadata.json`）。

---

## 1) 关键判断：为何 R 越大更快，但 PPL 可能崩？

把“质量劣化”拆成两个可度量目标（Phase A 的产物必须做出来）：

1) **断崖尖峰（hard eviction）**
- 每次 prune “一次丢太多 token” → prune 边界后 token-level NLL 出现尖峰
- 量化指标：`peak/width/area`（尖峰高度/宽度/面积）

2) **窗口漂移（effective window drift）**
- R 周期内 past 会增长，模型“有时看到 W，有时看到 W+R”
- 量化指标：
  - 可见 KV 长度 `min/mean/max`
  - 每次 prune 的 `drop_size` 分布（均值/最大值/分位数）

后续每个策略必须对应其主要改善的 KPI：
- `max_drop` → 压 drop_size 尾部（防止一次性大 drop）
- `overlap` → 降尖峰 peak/area
- `refresh` → 降尖峰 area + 缓解长期累积误差
- `adaptive/slack` → 降极端事件频率 + 收敛 drop_size 分布

---

## 2) Phase A（诊断，1–2 天）：把证据链补齐

### A1) NLL 尖峰诊断（必须）
交付物：
- 新诊断脚本：`experiments/diagnose_prune_nll.py`
  - 输出：token-level NLL、per-token time、prune step 列表
  - 自动计算：尖峰 `peak/width/area` + `drop_size` 分布
- 产物：两张图（pg19 为主）
  1) NLL vs token index（标注 prune 边界）
  2) per-token latency vs token index（标注 prune 边界）

Go/No-go：
- 若无尖峰但 PPL 仍显著变差 → 优先排查“窗口漂移”与实现逻辑（R 周期内模型到底看到了什么长度）。

### A2) 固定 TPOT 指标（必须）
实现/记录：
- steady-state TPOT（warmup=64）
- P50/P90 TPOT

原因：受控驱逐会把“平均 TPOT”变好，但也可能引入周期性尖峰；必须用 P90 约束“体验/稳定性”。

---

## 3) Phase B（受控驱逐，2–4 天）：稳住 PPL，同时保持加速

目标（写死成功判据，避免跑偏）：
- PG19：`ΔPPL ≤ 5%` 且 `TPOT ≤ (best_lazy_prune_TPOT * 1.02)`
- Wikitext：`ΔPPL ≤ 3%`，用于回归验证

### B1) max_drop（优先级最高）
实现：
- 新参数：`max_drop`（例如 256/512）
- 触发 prune 时，若 overflow 太大，不一次性 prune 到位：
  - 分多次小 prune（或进入 pruning mode 持续 K 步）

预期：
- 尖峰 peak/area 明显下降
- TPOT 接近大 R（剪裁频率仍低）

### B2) overlap（第二优先）
实现：
- 新参数：`overlap`（0/128/256）
- 逻辑保留：`sink + window + overlap`

预期：
- PPL 改善明显（尤其长文本段落过渡）
- TPOT 可能略回升，但在 MLP-dominated 下通常小于 3%

### B3) tiny refresh（第三优先，但论文价值高）
实现：
- 新参数：`refresh_budget`（0/8/16）
- refresh 选择策略（由易到难，必须模块化可插拔）：
  1) 均匀采样
  2) 结构 token（换行/句号/引号等）
  3) 不确定性 proxy（entropy/top1 prob）

预期：
- 在更大 R 下显著回拉 PPL（目标：把 “R 大导致的断崖”压平）
- refresh 只复制少量 KV，TPOT 影响应很小

### B4) slack/adaptive（仅在 B1–B3 仍不稳时启用）
实现：
- slack-based：用 `extra_slack=Δ` 控制最大超窗长度（Δ=48/64 起步）
- entropy-based：风险高就提前 prune，风险低就放宽

---

## 4) Phase B 的“最小实验矩阵”（先跑这 8 点）

固定：`window=2048, sink=4`（先复用你们已验证的 best 区间）
- `compress_every ∈ {32, 64}`
- `max_drop ∈ {None, 256}`
- `refresh_budget ∈ {0, 16}`

总计 8 点（wikitext + pg19），每点输出：
- `ΔPPL`
- steady-state `TPOT(mean/P50/P90)`
- NLL 尖峰指标（peak/width/area）
- prune drop_size 分布

优先跑序：
1) 先跑 PG19（主结论）
2) 再跑 wikitext 做回归/ sanity

---

## 5) Phase D（抬上限，3–8 天）：压 MLP/LN/残差/launch（目标：超越当前最快 streaming）

### D1) fused LayerNorm / residual-add（先做，风险最低）
目标：减少 elementwise kernels 与 launch。
止损条件：
- 2 天内无法稳定接入且无 >1% TPOT 改善 → 暂停并记录为 negative result。

### D2) CUDA Graphs（次优先）
目标：降低 launch/框架开销（batch=1/seq=1 更敏感）。
止损条件：
- 若出现不可控 cudagraph overwrite/shape 不稳定问题且 1 天内无法解决 → 降级，不阻塞主线结论。

### D3) fused MLP（最后冲刺，风险最高）
只在 D1/D2 已稳定、并且 profile 明确 MLP 仍占主导时尝试。
止损条件：
- 需要“换引擎/大范围重构”且 2 天内无稳定 demo → 立即停止，作为对照/讨论写入论文。

---

## 6) Phase C（缓存管理 O(1)，可选冲刺）：soft ring（wrap-only copy）

目标：减少“每次 prune 的拷贝/拼接成本”，但不引入高风险 kernel 改造。
原则：
- **只做 soft ring**（wrap 时 `cat`，平时仅指针/视图）
- 真正的 paged/block-table 或改 attention kernel 视为高风险，不作为主线依赖

止损条件：
- 若实现需要改 attention kernel / RoPE / FlashAttention 逻辑 → 立即降级或停止

---

## 7) Stop-loss（强制止损条款）

任何方向（B/C/D）若出现以下任一情况，必须止损并写入 negative result：
- 连续两轮改动无法同时改善（TPOT, ΔPPL）或改善 < 测量噪声（<1%）
- 引入不稳定（NaN、显著抖动、不可复现）
- 需要改变 attention 算法或大幅改模型结构（与当前“安全可插拔”目标冲突）

---

## 8) 论文收口（E 阶段）
必须产出两类 Pareto：
- x=steady-state TPOT（或 total time），y=ΔPPL（wikitext + pg19）
- 同时报告 P50/P90 TPOT（证明“尖峰被压平”）

推荐叙事闭环：
1) window sweep + lazy prune → 证明 cache 管理/裁剪频率是瓶颈
2) NLL 尖峰诊断 → 解释 PPL 断崖机制
3) 受控驱逐（max_drop/overlap/refresh/adaptive）→ 稳住 PPL 并保留速度
4) fused LN/graphs（可选）→ 抬高速度上限，尝试超越最快 streaming

