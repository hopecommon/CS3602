# Fused Kernel Implementation Progress

## 项目目标
手写CUDA算子实现 Fused LayerNorm + Residual Add，以降低kernel launch开销和内存带宽消耗。

## 当前进展 (2024-12-23)

### ✓ 已完成

#### 1. **基础CUDA Kernel实现**
- 文件：`fused_kernels/fused_ln_residual.cu`
- 实现了完整的fused LayerNorm + Residual Add kernel
- 特性：
  - Warp-level parallel reduction (mean/variance计算)
  - 支持FP32和FP16数据类型
  - 使用共享内存优化规约操作
  - 每个thread block处理一个token

#### 2. **Python绑定**
- 文件：`fused_kernels/fused_ln_residual.py`, `fused_ln_residual_cuda.cpp`
- JIT编译支持（使用torch.utils.cpp_extension）
- 类型检查和错误处理
- 便捷的Python接口

#### 3. **单元测试**
- 文件：`fused_kernels/test_fused_ln_residual.py`
- **正确性测试**：✓ 全部通过
  - FP32最大误差：9.5e-07
  - FP16最大误差：3.9e-03
  - 多种输入shape验证
- **性能测试**：
  - Decode (1,1,2560): 1.27x加速
  - Small batch: 1.25x加速
  - Large batch: 1.24x加速

### 进行中

#### 4. **模型集成**
- 文件：`fused_kernels/gptneox_integration.py`
- **状态**: 框架已搭建，但尚未真正融合到forward pass
- **当前实现**: 
  - Monkey-patching基础设施已完成
  - 可以enable/disable fused kernels
  - 测试函数可验证数值正确性
- **局限**: 
  - 当前版本只是替换了forward函数框架，实际计算仍使用原生PyTorch
  - 需要更深入地修改layer的计算顺序才能真正融合

### 待完成

#### 5. **优化CUDA Kernel**
目标：将1.25x加速提升到2-3x
- [ ] 向量化内存访问 (float4)
- [ ] 更好的线程块配置
- [ ] Welford算法提升数值稳定性
- [ ] 针对特定hidden_size优化 (如2560)

#### 6. **真正的模型集成**
挑战：GPTNeoX的结构需要深入修改
- [ ] 修改attention输出处理，融合第一个LN+residual
- [ ] 修改MLP输出处理，融合第二个LN+residual
- [ ] 或者：提供自定义GPTNeoXLayer实现

#### 7. **端到端评估**
- [ ] 在decode-loop中测试实际加速比
- [ ] 验证PPL不变性
- [ ] 与原始StreamingLLM对比

## 技术细节

### Kernel设计

**输入**:
```
x:        [batch, seq_len, hidden_size]  # 需要归一化的tensor
residual: [batch, seq_len, hidden_size]  # 残差
weight:   [hidden_size]                  # LayerNorm weight (gamma)
bias:     [hidden_size]                  # LayerNorm bias (beta)
eps:      float                          # 数值稳定常数
```

**输出**:
```
output: [batch, seq_len, hidden_size]  # LayerNorm(x) + residual
```

**算法流程**:
1. 每个thread block处理一个token (一行hidden_size个元素)
2. 并行规约计算mean
3. 并行规约计算variance
4. 逐元素计算: (x - mean) / sqrt(var + eps) * gamma + beta + residual

**优化点**:
- Warp shuffle实现高效规约
- 共享内存存储中间结果
- 避免中间tensor的global memory写回

### 性能分析

**为何当前加速比不高 (1.25x)?**

1. **Kernel本身不是瓶颈**: Profiling显示LayerNorm只占总时间的~15%
2. **MLP占主导**: addmm (矩阵乘法) 占用大部分时间
3. **集成未完成**: 当前monkey-patch没有真正使用fused kernel
4. **测试规模小**: 单个kernel的开销摊销不明显

**预期加速来源**:
- 减少kernel launch次数 (2个→1个)
- 减少中间tensor内存访问 (避免LayerNorm输出写回global memory)
- 在整个模型中累积效果 (32层 × 2次/层 = 64次融合机会)

### GPTNeoX架构

每个GPTNeoXLayer包含:
```python
# 第一部分: Attention
hidden = input_layernorm(hidden_states)
attn_output = attention(hidden, ...)
attn_output = dropout(attn_output)
attn_output = attn_output + hidden_states  # <-- 融合点1

# 第二部分: MLP
hidden = post_attention_layernorm(attn_output)
mlp_output = mlp(hidden)
mlp_output = dropout(mlp_output)
output = mlp_output + attn_output  # <-- 融合点2
```

**挑战**: 无法直接用`fused_layernorm_residual`替换，因为：
- LayerNorm在residual add**之前**
- 我们的kernel计算`LN(x) + residual`，而实际流程是`operation(LN(x)) + residual`

**可能的解决方案**:
1. 修改kernel计算`LN(x)`但不立即加residual，只是避免写回中间结果
2. 调整计算顺序（需要验证数值等价性）
3. 实现`LN + Linear`融合（更复杂）

## 下一步行动

### 短期 (优先)
1. **优化现有kernel**: 实现向量化访存，争取2x加速
2. **真正集成到模型**: 解决计算顺序问题，让fused kernel真正被调用
3. **端到端测试**: 在decode-loop中测量实际TPOT改善

### 中期
4. **更激进的融合**: LN + Linear, MLP fusion
5. **生产级优化**: 调优线程块配置，多GPU支持
6. **文档和消融**: 写清楚每个优化的贡献

### 长期
7. **Triton实现**: 对比手写CUDA vs Triton的性能/可维护性
8. **通用化**: 支持其他模型架构 (LLaMA, GPT-2等)

## 关键代码文件

```
fused_kernels/
├── __init__.py                 # 模块入口
├── fused_ln_residual.cu        # CUDA kernel实现
├── fused_ln_residual_cuda.cpp  # C++绑定
├── fused_ln_residual.py        # Python接口
├── gptneox_integration.py      # 模型集成（未完成）
├── test_fused_ln_residual.py   # 单元测试 ✓
└── test_integration.py         # 集成测试
```

## 测试命令

```bash
# 单元测试（验证kernel正确性和性能）
python fused_kernels/test_fused_ln_residual.py

# 集成测试（验证模型集成）
python fused_kernels/test_integration.py

# 清除编译缓存（如果修改了.cu或.cpp文件）
rm -rf ~/.cache/torch_extensions/py312_cu121/fused_ln_residual
```

## 参考资料

- NVIDIA CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- PyTorch Custom C++ and CUDA Extensions: https://pytorch.org/tutorials/advanced/cpp_extension.html
- FlashAttention paper (for fusion inspiration): https://arxiv.org/abs/2205.14135
- Transformer Engine (NVIDIA): https://github.com/NVIDIA/TransformerEngine
