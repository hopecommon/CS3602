/**
 * Fused LayerNorm + Residual Add CUDA Kernel
 * 
 * This kernel fuses two operations:
 * 1. LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta
 * 2. Residual Add: output = y + residual
 * 
 * Input shapes:
 * - x: [batch, seq_len, hidden_size] - input tensor
 * - residual: [batch, seq_len, hidden_size] - residual tensor
 * - gamma: [hidden_size] - LayerNorm weight
 * - beta: [hidden_size] - LayerNorm bias
 * 
 * Output:
 * - output: [batch, seq_len, hidden_size] - fused result
 * 
 * Algorithm:
 * Each thread block processes one token (one row of hidden_size elements)
 * 1. Compute mean and variance using parallel reduction
 * 2. Normalize and apply affine transformation
 * 3. Add residual
 * 
 * Optimizations:
 * - Use shared memory for parallel reduction (mean/variance)
 * - Vectorized memory access (float4) when hidden_size % 4 == 0
 * - Minimal intermediate storage
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

// Helper functions for type conversion
__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }

template<typename T>
__device__ __forceinline__ T from_float(float x);

template<>
__device__ __forceinline__ float from_float<float>(float x) { return x; }

template<>
__device__ __forceinline__ __half from_float<__half>(float x) { return __float2half(x); }

// Warp-level reduction primitives
template<typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * Fused LayerNorm + Residual kernel - naive version (correctness first)
 * 
 * Each thread block handles one token (one row).
 * Grid dimension: (batch * seq_len)
 * Block dimension: min(hidden_size, 1024)
 */
template<typename T>
__global__ void fused_layernorm_residual_kernel(
    const T* __restrict__ x,
    const T* __restrict__ residual,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    T* __restrict__ output,
    int hidden_size,
    float eps
) {
    // Each block processes one token
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    // Shared memory for reduction
    extern __shared__ float shared_mem[];
    
    // Pointer to input for this token
    const T* x_row = x + token_idx * hidden_size;
    const T* residual_row = residual + token_idx * hidden_size;
    T* output_row = output + token_idx * hidden_size;
    
    // Step 1: Compute mean using parallel reduction
    float sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        sum += to_float(x_row[i]);
    }
    
    // Warp-level reduction
    sum = warpReduceSum(sum);
    
    // Write to shared memory (one value per warp)
    int lane = tid % 32;
    int wid = tid / 32;
    if (lane == 0) {
        shared_mem[wid] = sum;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (wid == 0) {
        sum = (tid < (blockDim.x + 31) / 32) ? shared_mem[tid] : 0.0f;
        sum = warpReduceSum(sum);
        if (tid == 0) {
            shared_mem[0] = sum / hidden_size;  // mean
        }
    }
    __syncthreads();
    
    float mean = shared_mem[0];
    
    // Step 2: Compute variance
    float var_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float centered = to_float(x_row[i]) - mean;
        var_sum += centered * centered;
    }
    
    // Warp-level reduction
    var_sum = warpReduceSum(var_sum);
    
    // Write to shared memory
    if (lane == 0) {
        shared_mem[wid] = var_sum;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (wid == 0) {
        var_sum = (tid < (blockDim.x + 31) / 32) ? shared_mem[tid] : 0.0f;
        var_sum = warpReduceSum(var_sum);
        if (tid == 0) {
            shared_mem[1] = var_sum / hidden_size;  // variance
        }
    }
    __syncthreads();
    
    float variance = shared_mem[1];
    float inv_std = rsqrtf(variance + eps);
    
    // Step 3: Normalize, apply affine, and add residual
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float normalized = (to_float(x_row[i]) - mean) * inv_std;
        float affine = normalized * to_float(gamma[i]) + to_float(beta[i]);
        float result = affine + to_float(residual_row[i]);
        output_row[i] = from_float<T>(result);
    }
}

// Explicit instantiation for float and half
template __global__ void fused_layernorm_residual_kernel<float>(
    const float*, const float*, const float*, const float*, float*, int, float);

template __global__ void fused_layernorm_residual_kernel<__half>(
    const __half*, const __half*, const __half*, const __half*, __half*, int, float);

// Launcher functions callable from C++
extern "C" {

void fused_layernorm_residual_cuda_forward_float(
    const float* x,
    const float* residual,
    const float* gamma,
    const float* beta,
    float* output,
    int blocks,
    int threads,
    int shared_mem_size,
    int hidden_size,
    float eps
) {
    fused_layernorm_residual_kernel<float><<<blocks, threads, shared_mem_size>>>(
        x, residual, gamma, beta, output, hidden_size, eps
    );
}

void fused_layernorm_residual_cuda_forward_half(
    const __half* x,
    const __half* residual,
    const __half* gamma,
    const __half* beta,
    __half* output,
    int blocks,
    int threads,
    int shared_mem_size,
    int hidden_size,
    float eps
) {
    fused_layernorm_residual_kernel<__half><<<blocks, threads, shared_mem_size>>>(
        x, residual, gamma, beta, output, hidden_size, eps
    );
}

}  // extern "C"
