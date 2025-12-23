# 手写CUDA算子实现日志 (2024-12-23)

## 背景

根据profiling结果，在StreamingLLM优化attention/KV cache之后，剩余的性能瓶颈主要在：
1. MLP (addmm操作，占用最多时间)
2. LayerNorm (高频调用，33345次)
3. Residual add (高频调用，65664次)
4. Kernel launch overhead

为了进一步提升性能，我们尝试手写CUDA算子来融合LayerNorm和Residual Add操作。

## 实施步骤

### Phase 1: Kernel实现与测试 ✓ (完成)

#### 1.1 设计与实现
- **目标算子**: Fused LayerNorm + Residual Add
  ```python
  # 原始实现 (两个kernel)
  normalized = F.layer_norm(x, normalized_shape, weight, bias, eps)
  output = normalized + residual
  
  # 融合实现 (一个kernel)
  output = fused_layernorm_residual(x, residual, weight, bias, eps)
  ```

- **CUDA Kernel特性**:
  - 每个thread block处理一个token (hidden_size维度)
  - Warp-level parallel reduction计算mean/variance
  - 共享内存优化规约操作
  - 支持FP32/FP16

#### 1.2 编译和测试结果

**环境**:
- PyTorch 2.3.0 + CUDA 12.1
- NVIDIA A800 (Ampere架构)
- JIT编译 (torch.utils.cpp_extension)

**正确性测试** ✓:
```
FP32 - small batch [2, 128, 2560]:
  Max absolute error: 9.536743e-07
  Max relative error:  6.128661e-03
  Status: ✓ PASS

FP16 - small batch [2, 128, 2560]:
  Max absolute error: 3.906250e-03
  Max relative error:  inf (部分zero导致，实际可接受)
  Status: ✓ PASS

FP16 - decode [1, 1, 2560]:
  Max absolute error: 3.906250e-03
  Status: ✓ PASS

FP16 - large batch [4, 256, 2560]:
  Max absolute error: 3.906250e-03
  Status: ✓ PASS

FP16 - long sequence [1, 2048, 2560]:
  Max absolute error: 3.906250e-03
  Status: ✓ PASS
```

**性能测试** (初步):
```
FP16 - decode (1, 1, 2560):
  PyTorch time: 0.0305 ms
  Fused time:   0.0240 ms
  Speedup:      1.27x

FP16 - small batch (2, 128, 2560):
  PyTorch time: 0.0302 ms
  Fused time:   0.0242 ms
  Speedup:      1.25x

FP16 - large batch (4, 256, 2560):
  PyTorch time: 0.0304 ms
  Fused time:   0.0244 ms
  Speedup:      1.24x
```

### Phase 2: 模型集成 (进行中)

#### 2.1 GPTNeoX架构分析

每个GPTNeoXLayer的结构:
```python
# Part 1: Attention path
hidden = input_layernorm(hidden_states)              # LN1
attn_output = attention(hidden, ...)                  # Attention
attn_output = post_attention_dropout(attn_output)     # Dropout
attn_output = attn_output + hidden_states             # Residual Add 1

# Part 2: MLP path
hidden = post_attention_layernorm(attn_output)        # LN2
mlp_output = mlp(hidden)                              # MLP
mlp_output = post_mlp_dropout(mlp_output)             # Dropout
output = mlp_output + attn_output                     # Residual Add 2
```

**融合点分析**:
- 目标：融合LN1+Residual1, LN2+Residual2
- 挑战：LayerNorm在residual add **之前**，中间有其他操作
- 我们的kernel: `output = LN(x) + residual`
- 实际流程: `output = operation(LN(x)) + residual`

#### 2.2 集成策略

**方案A**: Monkey-patching (已实现框架)
- 文件：`fused_kernels/gptneox_integration.py`
- 实现：`apply_fused_kernels(model, enabled=True/False)`
- 状态：框架完成，但未真正调用fused kernel
- 问题：需要修改计算顺序或kernel接口

**方案B**: 自定义Layer (待实现)
- 创建`FusedGPTNeoXLayer`继承原始Layer
- 重写forward函数，真正使用fused kernel
- 更灵活但需要更多代码

**方案C**: 修改Kernel接口 (可能最优)
- 实现两个独立kernel:
  1. `fused_ln_forward`: 计算LN但不加residual，避免写回
  2. `fused_residual_add`: 在最后加residual
- 匹配实际计算流程

## 当前状态总结

### ✓ 已完成
1. **CUDA Kernel实现**: 完整的Fused LN+Residual kernel
2. **Python绑定**: JIT编译，易用接口
3. **正确性验证**: 所有测试通过，数值精度符合预期
4. **初步性能测试**: 单kernel有1.25x加速

### 🔄 进行中
5. **模型集成**: 框架搭建完成，需要解决计算顺序问题

### 📋 待完成
6. **Kernel优化**:
   - 向量化内存访问 (float4)
   - 更好的线程块配置
   - Welford算法提升数值稳定性
   
7. **真正集成到forward pass**:
   - 修改GPTNeoXLayer的forward
   - 确保fused kernel被实际调用
   
8. **端到端评估**:
   - 在decode-loop中测试
   - 测量实际TPOT改善
   - 验证PPL不变性

## 性能预期

**理论分析**:
- 每层2个融合点 × 32层 = 64个融合机会
- 每次融合节省：
  - 1个kernel launch
  - 1次中间tensor写回global memory
  - 1次读取global memory

**实际挑战**:
- LayerNorm只占总时间~15% (profiling数据)
- MLP (addmm) 占主导地位 (~45%)
- 预期整体加速: 5-10% (保守估计)

**进一步优化方向**:
- Fused MLP (更高价值，但更复杂)
- Fused LN + Linear
- CUDA Graphs (减少launch overhead)

## 技术难点

### 1. FP16类型转换
**问题**: CUDA中`__half`类型不能直接cast
**解决**: 实现helper函数
```cuda
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
__device__ __forceinline__ __half from_float(float x) { return __float2half(x); }
```

### 2. C++中的Kernel调用语法
**问题**: CUDA kernel调用语法`<<<>>>`在.cpp文件中无效
**解决**: 创建launcher函数在.cu中
```cpp
// .cu file
extern "C" void fused_layernorm_residual_cuda_forward_float(...) {
    fused_layernorm_residual_kernel<float><<<blocks, threads, shared_mem_size>>>(...);
}

// .cpp file
extern "C" void fused_layernorm_residual_cuda_forward_float(...);
```

### 3. 计算顺序不匹配
**问题**: `LN(x) + residual` vs `op(LN(x)) + residual`
**待解决**: 需要重新设计kernel接口或修改模型forward

## 代码结构

```
fused_kernels/
├── __init__.py                 # 模块入口
├── fused_ln_residual.cu        # CUDA kernel实现 ✓
├── fused_ln_residual_cuda.cpp  # C++绑定 ✓
├── fused_ln_residual.py        # Python接口 ✓
├── gptneox_integration.py      # 模型集成 (框架完成)
├── test_fused_ln_residual.py   # 单元测试 ✓
├── test_integration.py         # 集成测试 (待运行)
└── README.md                   # 详细文档 ✓
```

## 测试命令

```bash
# 单元测试 (kernel正确性和性能)
python fused_kernels/test_fused_ln_residual.py

# 集成测试 (模型级别)
python fused_kernels/test_integration.py

# 清除编译缓存
rm -rf ~/.cache/torch_extensions/py312_cu121/fused_ln_residual
```

## 下一步计划

### 立即行动 (1-2天)
1. 解决计算顺序问题，实现真正的模型集成
2. 运行端到端测试，测量实际加速比
3. 如果加速不明显，考虑：
   - 优化现有kernel (向量化等)
   - 或转向更高价值目标 (MLP fusion)

### 中期目标 (3-5天)
4. Kernel优化，争取2-3x单kernel加速
5. 实现多个融合算子 (如果证明有价值)
6. 完整的消融实验

### 评估标准
- **成功**: TPOT降低≥5%, PPL不变
- **可接受**: TPOT降低2-5%, PPL不变
- **失败**: TPOT无改善或PPL下降

## 参考资料

- [PyTorch C++ Extensions Tutorial](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)

## 结论

✓ **成功实现了第一个手写CUDA算子**
- Kernel本身正确且有1.25x加速
- 编译和测试基础设施完备
- 为后续优化打下基础

⚠ **模型集成仍需解决**
- 计算顺序不匹配
- 需要修改forward pass或重新设计kernel

📊 **性能改善预期**
- 保守估计: 5-10%整体TPOT改善
- 如果配合其他优化: 可能更高

---
**记录时间**: 2024-12-23  
**状态**: 进行中
**下次更新**: 完成模型集成后
