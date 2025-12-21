收到：A800（Ampere）+ Streaming（`N_SINK=4, WINDOW_SIZE=2048`）+ decode 已经 **MLP dominated**，那你们把主线放在**整模型量化（优先打 MLP 的 Linear）**是对的。

下面给你一套“能跑通、可复现、PPL 可控、并且能写进小论文”的完整方案（先主线整模型量化；最后再讲 KV 量化与 RefreshLite 怎么作为补充）。

---

## 0) 先定一个现实前提：A800 不要把 FP8 当主力

A800 是 **Ampere** 架构，FP8 Tensor Core 是 Hopper(H100) 的特性。业界框架也通常把 **W8A8（权重+激活 FP8）**作为 Hopper/Ada 的“官方硬件加速路径”，Ampere更多是用替代 kernel 或只做 weight-only 的兼容路径。
**结论**：你们在 A800 上要优先做 **INT8 / INT4（Tensor Core 友好）**，别把赌注押在 FP8。

---

## 1) 主线目标与评价方式（你们关注 TPOT）

你们要“平均 decode 每 token 越快越好、总处理时间越短”，那 benchmark 要刻意让 decode 占主导：

**推荐 benchmark（固定下来，所有 ablation 共用）**

* Prompt 长度：≥ window（比如 8k/16k）
* 生成：`max_new_tokens >= 1024`（越长越能稳定到 steady-state）
* 统计 TPOT：丢掉前 32~64 token（warmup），对后续 token 取均值
* 同时记录：peak GPU mem、吞吐（tok/s）

> 你们当前 Streaming 让 KV 长度基本固定在 window+sink，这对后面 `torch.compile(reduce-overhead)` 很友好（shape 更稳定）。

---

## 2) 整模型量化：推荐 TorchAO（主打“真加速”而不只是省显存）

TorchAO 是 PyTorch 原生优化库，专门做量化/稀疏等，并且强调和 `torch.compile` 可组合。它的 Int4/Int8 配置是为推理 kernel/pack layout 设计的，例如 Int4 weight-only 默认就是 **TensorCoreTiledLayout**，配 tinygemm/GemLite 这类实现来提速。

### Phase A（先稳住 PPL）：INT8 weight-only（全模型 Linear）

**目标**：PPL 基本不变，先验证 TPOT 是否下降。
用 TorchAO 的 `Int8WeightOnlyConfig` 或 `Int8DynamicActivationInt8WeightConfig`（后者更激进，可能更快但更需要盯 PPL）

**建议先跑：**

1. `int8_weight_only`（更稳）
2. 如果 TPOT 几乎没动，再试 `int8_dynamic_act + int8_weight`（看是否真能打到 MLP）

### Phase B（冲 TPOT）：INT4 weight-only（优先只量化 MLP，再逐步扩大）

你们瓶颈在 MLP，所以最有效/最可控的做法是做一个“论文级”混合量化策略：

* **MLP 两个 Linear：INT4 weight-only**（group_size 从 128 开始）
* **Attention 相关 Linear：先 INT8 或保持 BF16**
* **Embedding / LM head：保持 BF16**（最稳 PPL）

Int4 的 group_size 是可调粒度（256/128/64/32，越小越接近原模型、但可能慢一点）。我建议从 **128** 开始：通常是速度/精度的甜点位（也好写进论文）。

---

## 3) 关键加成：一定要配 `torch.compile(mode="reduce-overhead")`

你们 profiling 已经看到“残差/框架”开销很大，所以量化后依然可能被 kernel launch / Python 循环吃掉收益。

`reduce-overhead` 会用 CUDA graphs 降低 Python 开销，适合小 batch。这跟你们“单流 decode”非常匹配。

**推荐组合：**

* baseline（BF16）+ compile
* quant（INT8/INT4）+ compile

这样你们能回答论文里最容易被问的问题：

> “你们的加速来自量化算子，还是来自框架开销下降？”

---

## 4) 具体落地做法（尽量不和 Streaming 冲突）

你们现在有 StreamingLLM wrapper。整模型量化建议做到“**不动 Streaming 逻辑**、只在模型加载后做量化 + compile”。

### 4.1 最稳的工程路径

1. load BF16/BF16-ish 模型（A800 上 BF16 OK）
2. apply TorchAO quant（先 INT8WO，再 MLP-only INT4WO）
3. `torch.compile`（只编译 forward，decode loop 不用大改）
4. 用你们现有 Streaming decode 跑 benchmark + PPL

### 4.2 MLP-only INT4 的过滤（GPT-NeoX / Pythia 常见模块名）

Pythia-2.8B 是 GPT-NeoX 系列，MLP 典型层名通常类似：

* `...mlp.dense_h_to_4h`
* `...mlp.dense_4h_to_h`

你们可以用 TorchAO 的 `quantize_` + filter_fn 只量化这些层（这就是你们的“创新/综合优化”：**只量化瓶颈层**）。

---

## 5) 你们应该跑的 ablation（最少但论文够用）

以 Streaming（`W=2048, sink=4`）为固定前提，做：

1. **BF16 + Streaming**（你们当前最强 baseline）
2. **BF16 + Streaming + torch.compile(reduce-overhead)**
3. **INT8WO(all Linear) + Streaming + compile**
4. **INT4WO(MLP only, group=128) + Streaming + compile**
5. （可选）**INT4WO(MLP only, group=64)**（如果 PPL 需要更稳）

输出表格：TPOT、总时间、peak_mem、PPL（pg-19 / wikitext）。

---

## 6) 其次：KV cache 量化怎么放（建议定位：显存/带宽，不承诺 TPOT）

既然你们已经 MLP dominated，KV 量化对 TPOT 的贡献可能有限；但它依然有价值：

* peak_mem 更低（论文很好看）
* 如果你们想把 window 从 2048 提到 4096 还稳住显存，它是最直接手段

实现上尽量做“**存 int8，算 bf16**”的 KV：

* 写入 cache 时 per-head/per-channel scale + int8 pack
* 读出时 dequant 回 bf16 给 attention 用

注意：如果 dequant 开销太大，可能 **不降反升**，所以把它放在“可选增强”，不要作为主线结论。

---

## 7) 最后：RefreshLite 何时值得做（用来砍 window）

你们现在 `W=2048`，如果 PPL 只变 3%，说明你们窗口已经很健康。RefreshLite 的价值主要在：

* 想把 window 砍到 1024/512 进一步提速/省显存
* 但 PPL 开始明显变差

这时 RefreshLite 用很轻的“周期性补充 global tokens/sink”把 PPL 拉回来，帮助你们画出 **TPOT–PPL Pareto 曲线**（非常加分）。

---

## 默认行动清单（按 A800 + W=2048）

**本周优先做：**

1. BF16 + Streaming 的 decode benchmark 固定下来（TPOT / 总时间 / PPL）
2. 加 `torch.compile(mode="reduce-overhead")`（看看框架开销能不能直接被吃掉）
3. 上 TorchAO **INT8 weight-only（all Linear）**（看 TPOT 与 PPL）
4. 上 TorchAO **INT4 weight-only（MLP only, group=128）**（主冲 TPOT）

**如果 Phase B 的 PPL 不理想：**

* 把 group_size 改 64/32（更细粒度更稳）
* 或只量化 `dense_h_to_4h`，把 `dense_4h_to_h` 留 BF16（再稳一档）

---

## 实验脚本说明（重要）

当前 `torch.compile` 默认 **禁用 cudagraphs**，避免在循环调用中出现输出覆盖错误。
需要强行启用时再加 `--compile-cudagraphs`（可能不稳定）。
