结论：

* **论文里 StreamingLLM 的最高加速比是：≈ 22.2×**
* 这个 22.2× 是**相对“sliding window + 重新计算 KV（re-computation）”的基线**，在 **Llama-2-7B/13B** 上测的 per-token decoding latency（每个 token 的解码时间），与 PG-19 或 Pythia-70M 这个具体组合没直接给数值。([ar5iv][1])
* 对 **PG-19 + Pythia 系列**，论文主要给的是 **perplexity（困惑度）**，说明 StreamingLLM 在保持窗口大小不变的情况下可以做到**几乎不掉点**，但没专门报 Pythia-70M 的加速倍数。([ar5iv][1])

下面分三块说：

1. 论文里到底是怎么测到 22.2× 的
2. 在 PG-19 + Pythia 上，他们实际做了什么
3. 你如果想在 EleutherAI/pythia-70m 上复现，大概要改什么代码 & 参考实现

---

## 1. 论文里的加速比是怎么来的？

### 基线：Sliding Window with Re-computation

他们认为对长序列、想保持性能的情况下，**唯一“靠谱”的基线**是：

> Sliding Window with Re-computation
> 每生成 1 个新 token，就把最近 W 个 token 整个再过一遍模型，重新建立这段窗口的 KV 缓存。

* 每步复杂度 ~ **O(W²)**（在长 text 的 steady-state 下）
* 解码延迟随着窗口 W **二次方**增长（cache 越大越慢）。([ar5iv][1])

### StreamingLLM 做了什么不同？

StreamingLLM 的核心是：
KV 缓存被拆成两部分：([ar5iv][1])

1. **Attention sinks：前面若干个“坑位 token”**（默认 4 个）

   * 永远保留在 KV cache 里，不被丢弃；
   * 它们并不一定语义重要，但模型在训练时学会了总是喜欢把多余的 attention dump 在这些位置上。
2. **Rolling window：最近 W 个 token**

   * 和普通 sliding window 一样，只保留最近的一段历史。

每一步生成新 token 时：

* 只计算 **当前这个 token** 的 K/Q/V（以及可能需要重算的 RoPE 旋转）
* **不会重跑**整段最近窗口；
* Attention 只在 **[4 个 sink] + [W 个最近 token]** 上做一次点积，**复杂度 ~ O(W)**。

于是：

* 基线：每步复杂度 O(W²)
* StreamingLLM：每步复杂度 O(W)
* 所以理论加速比 ~ **O(W)**，W 越大，收益越高。

### 实际的 22.2× 是什么配置？

论文第 4.5 节的效率实验里，他们：([ar5iv][1])

* 模型：**Llama-2-7B / 13B**
* 硬件：**NVIDIA A6000**
* 比较对象：

  * Baseline：sliding window + re-computation
  * Ours：StreamingLLM（4 个 sink token + rolling window）
* 横轴：cache size（attention window size）
* 纵轴：per-token decoding latency

结果图（Fig. 10）显示：

> 随着 cache size 增大，基线的延迟呈明显 **二次方上升**，StreamingLLM 近似 **线性上升**，在某些较大的窗口配置下，StreamingLLM 相比基线 **达到最高 22.2× 的加速**。([ar5iv][1])

这 22.2× **不是某个特定数据集/模型（比如 Pythia-70M）专属**，而是“给定窗口大小，针对 Llama-2 的长序列解码”的总体 speedup。

---

## 2. 在 PG-19 + Pythia 上，他们具体做了什么？

### PG-19 的用法

PG-19 是一堆长小说，特别适合做长文本语言建模。论文里是这么用的：([ar5iv][1])

* 把 PG-19 test set 100 本书 **拼接成一个长序列**（4M tokens 级别），做语言建模 perplexity；
* 对 Llama-2，cache size 设为 2048；
* 对 Falcon、Pythia、MPT，cache size 设为 1024（它们预训练窗口是 2048）。

### 在 PG-19 上的主要结论

1. **Window attention 会崩溃**：只保留最近 W 个 token（不保留开头的 sink）的话，一旦序列长度超过 cache size，perplexity 会炸到几千。([ar5iv][1])

2. **只要把前 4 个 token 当 sink 加回来，就能恢复性能**：

   * Table 1 / 2 中，Llama-2-13B 在 PG-19 第一书上：

     * 0+1024（纯 window）PPL ≈ 5158
     * 4+1020（加 4 个初始 token）PPL ≈ 5.4
     * 把这 4 个 token换成 4 个“换行符”也行（说明位置比语义更重要）。([ar5iv][1])

3. **跨模型族（包括 Pythia-12B）都类似**：

   * Table 2 里，Falcon-7B / MPT-7B / Pythia-12B / Llama-2-7B 都在 PG-19 上测了不同缓存配置（0+W,1+W-1,2+W-2,4+W-4,8+W-8）。
   * 结论：

     * 0+W（没有 sink）PPL 非常差；
     * 加到 4 个 sink token 后，PPL 基本恢复；继续加到 8 个收益很小。([ar5iv][1])

> 注意：这里的 Pythia 型号是 **2.8B / 6.9B / 12B**，不是 70M；他们还用 **Pythia-160M 代码库**从头训了一些模型来研究“learnable sink token”的效果。([ar5iv][1])

### 那 Pythia-70M 呢？

* 论文本身 **没有**专门跑 EleutherAI/pythia-70m + PG-19 的实验，也没有给这个组合的加速倍数或 PPL 表。
* 但从原理上讲：Pythia-70M 也是标准的 GPT-NeoX + RoPE 架构，只要满足“自回归 + RoPE/ALiBi”这些条件，就**可以直接套 StreamingLLM 的方法**。([ar5iv][1])

所以如果你想要“PG-19 + Pythia-70M 的 speedup 数字”，需要：

* 用他们的实现（或你自己的）在 PG-19 上重新测：

  * Baseline：sliding window + recomputation
  * Ours：StreamingLLM（4 个 sink + W 窗口）
* 然后自己算每 token latency 的比值。

---

## 3. 怎么在 EleutherAI/pythia-70m 上用 StreamingLLM？（代码思路）

### 官方代码和第三方实现

可以直接参考的实现：

1. **官方仓库：mit-han-lab/streaming-llm**

   * 论文作者维护，MIT 许可。([GitHub][2])
   * README 提到：支持 Llama-2, MPT, Falcon, Pythia 等模型；
   * 有示例脚本 `examples/run_streaming_llama.py` 展示如何在 HuggingFace Transformers 上启用 `--enable_streaming`。([GitHub][2])

2. **第三方 Attention Sinks 实现**

   * README 里提到一个叫 **Attention Sinks** 的第三方 repo，可以在更多 HF 模型上启用 StreamingLLM 思路。([GitHub][2])
   * 对于 Pythia-70M 这种小模型，很适合作为你调试 StreamingLLM 的 playground。

> 由于 GitHub 页面是 Web 动态渲染，我在这里看不到仓库里每个文件的源码内容，但从 README 可以确认：StreamingLLM 的核心逻辑和 demo 都已经开源。

### 核心实现思路（RoPE + Pythia）

以 **EleutherAI/pythia-70m** 为例（HuggingFace Transformers）：

1. **加载模型和 tokenizer**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "EleutherAI/pythia-70m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cuda",  # 或 "auto"
)
model.eval()
```

2. **维护一个“带 sink 的 KV cache”**

逻辑上，对每一层我们维护：

```text
KV_cache[layer] = {
    "sink_k": [K_0, K_1, K_2, K_3],        # 前 4 个 sink tokens
    "sink_v": [V_0, V_1, V_2, V_3],
    "roll_k": [K_{t-W+1}, ..., K_{t}],     # 滑动窗口中的 K
    "roll_v": [V_{t-W+1}, ..., V_{t}],
}
```

解码每一步：

* 把新的 token index 喂进去，只算当前 token 的 K/Q/V；
* 把它 append 到 `roll_k` / `roll_v`；
* 如果 `roll_k` 长度 > W，把最旧的那个弹掉；
* Attention 的 key/value 就是 `concat(sink_k, roll_k)` / `concat(sink_v, roll_v)`。

3. **RoPE 的特殊处理**

论文强调了一点：**位置编码要在“cache 内的位置”上重算，而不是原始文本的绝对位置**。([ar5iv][1])

* 假设缓存里现在有 `[sink0, sink1, sink2, sink3, token_x, token_{x+1}, ..., token_t]`，
* 那么在这一时刻，它们的**位置**应该被视为 `[0, 1, 2, 3, 4, 5, ..., cache_size-1]`，
* 而不是原始文本的 `[0, 1, 2, 3, x, x+1, ..., t]`。

对 RoPE 的实现含义是：

* **缓存的是“RoPE 之前”的 K**（未旋转）；
* 每次解码时，根据当前 cache 内的位置，对所有缓存 K 做一次 RoPE 旋转，然后和当前 Q 做 attention。([ar5iv][1])

伪代码类似（注意：只是示意，和实际 HF 代码结构会有差别）：

```python
def apply_rope(k_unrotated, positions, rope_cache):
    # 根据 positions 对每个 head、每个 token 旋转
    # positions 从 0 到 len(cache)-1
    ...
    return k_rotated

def streaming_attention_step(q, k_unrotated_cache, v_cache, num_sink, rope_cache):
    # k_unrotated_cache = concat(sink_k_unrot, roll_k_unrot)
    positions = torch.arange(k_unrotated_cache.size(1), device=q.device)
    k_rotated = apply_rope(k_unrotated_cache, positions, rope_cache)

    # 正常做 attention: softmax(q @ k_rotated^T / sqrt(d)) @ v_cache
    ...
```

4. **把这套 KV cache 嵌到 HF 的 `generate` 流程里**

你可以：

* 继承 Pythia 模型的 `forward`，增加 `use_streaming_kv` / `num_sink_tokens` 等参数；
* 或者写一个 wrapper，在每次调用 `model(**inputs, past_key_values=...)` 时，
  把返回的 `past_key_values` 手动截断成 `[sink + window]` 的形式，并重新拼装 RoPE 位置。

伪代码示意（极度简化）：

```python
class StreamingWrapper:
    def __init__(self, base_model, num_sink=4, window_size=1024):
        self.model = base_model
        self.num_sink = num_sink
        self.window_size = window_size
        self.past_key_values = None  # 自己管理

    def step(self, input_ids):
        # 第一次调用：正常 forward，填满 sink + 第一段窗口
        # 之后：用自家的 past_key_values 替换 HF 内部的缓存
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=self.past_key_values,
            use_cache=True,
        )

        logits = outputs.logits[:, -1, :]
        new_past = outputs.past_key_values

        # new_past 结构：(num_layers, 2, batch, num_heads, seq_len, head_dim)
        # 这里对每一层截成 [sink + window]
        self.past_key_values = truncate_to_sink_and_window(
            new_past, num_sink=self.num_sink, window_size=self.window_size
        )
        return logits
```

真正的实现细节推荐直接参考官方仓库和 Attention Sinks 的代码，这里只是给出你要改的“几刀”大致位置。

---

## 4. 小结（结合你问的点）

1. **加速比是多少？**

   * 论文里明确给的最高 speedup 是 **22.2×**，
   * 相对基线：**sliding window + recomputation**；
   * 在 Llama-2-7B / 13B 上测出的 per-token decoding latency。([ar5iv][1])

2. **PG-19 + Pythia / Llama-2 具体怎么做？**

   * 把 PG-19 的 test set 长书拼成大长序列；
   * 缓存配置统一为“**4 个 attention sink + 最近 W 个 token**”；
   * 在这个配置下，StreamingLLM 在 PG-19 上的 perplexity **基本等同于 sliding window + 重算基线**，而纯 window attention 则完全崩溃。([ar5iv][1])

3. **论文有没有专门给 “PG-19 + EleutherAI/pythia-70m 的加速比”？**

   * **没有**，他们在 Pythia 上主要是 2.8B / 6.9B / 12B 和 160M codebase，没提 70M。([ar5iv][1])
   * 但从架构上讲，Pythia-70M 完全能用同一套 StreamingLLM 技巧，你可以用开源实现自己测。

4. **有没有可参考的实现？**

   * 官方：**mit-han-lab/streaming-llm**（强烈推荐直接读这个）([GitHub][2])
   * 第三方：Attention Sinks（在更多 HF 模型上集成 StreamingLLM 思路）([GitHub][2])
   * 实现要点就是：

     * KV cache = `sinks + rolling window`；
     * 每步只算新 token 的 K/Q/V，复杂度 O(W)；
     * RoPE/ALiBi 的位置编码基于“cache 内位置”而不是原始绝对位置。([ar5iv][1])

如果你愿意，我也可以帮你**针对 EleutherAI/pythia-70m 写一个更接近 HuggingFace 实际接口的最小可运行 demo（带 generate loop）**，你可以直接改成跑 PG-19。

[1]: https://ar5iv.org/pdf/2309.17453 "[2309.17453] Efficient Streaming Language Models with Attention Sinks"
[2]: https://github.com/mit-han-lab/streaming-llm "GitHub - mit-han-lab/streaming-llm: [ICLR 2024] Efficient Streaming Language Models with Attention Sinks"
