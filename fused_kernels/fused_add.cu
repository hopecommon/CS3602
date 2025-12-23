/**
 * Simple Fused Add kernel for residual connections.
 * 
 * This is a minimal kernel that fuses element-wise addition,
 * which is simpler and more realistic than fused LN+residual for Pre-LN models.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Helper for type conversions
template<typename T>
__device__ __forceinline__ T add_values(T a, T b) {
    return a + b;
}

template<>
__device__ __forceinline__ __half add_values<__half>(__half a, __half b) {
    return __hadd(a, b);
}

// Simple element-wise addition kernel
template<typename T>
__global__ void fused_add_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = add_values(a[idx], b[idx]);
    }
}

// Vectorized version using float4 for better memory bandwidth
template<typename T>
__global__ void fused_add_kernel_vectorized(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ output,
    int n
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < n) {
        // Load as float4 for vectorized access
        float4 a_vec = reinterpret_cast<const float4*>(a)[idx / 4];
        float4 b_vec = reinterpret_cast<const float4*>(b)[idx / 4];
        
        float4 result;
        result.x = a_vec.x + b_vec.x;
        result.y = a_vec.y + b_vec.y;
        result.z = a_vec.z + b_vec.z;
        result.w = a_vec.w + b_vec.w;
        
        reinterpret_cast<float4*>(output)[idx / 4] = result;
    } else if (idx < n) {
        // Handle remainder
        for (int i = idx; i < n && i < idx + 4; i++) {
            output[i] = a[i] + b[i];
        }
    }
}

// Launcher functions
extern "C" {

void fused_add_cuda_forward(
    const void* a,
    const void* b,
    void* output,
    int n,
    int dtype_size,
    bool use_vectorized
) {
    if (use_vectorized && n % 4 == 0 && dtype_size == 4) {
        // Use vectorized kernel for float (4 bytes)
        int threads = 256;
        int blocks = (n / 4 + threads - 1) / threads;
        fused_add_kernel_vectorized<float><<<blocks, threads>>>(
            static_cast<const float*>(a),
            static_cast<const float*>(b),
            static_cast<float*>(output),
            n
        );
    } else if (dtype_size == 4) {
        // Standard kernel for float
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        fused_add_kernel<float><<<blocks, threads>>>(
            static_cast<const float*>(a),
            static_cast<const float*>(b),
            static_cast<float*>(output),
            n
        );
    } else if (dtype_size == 2) {
        // Standard kernel for half
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        fused_add_kernel<__half><<<blocks, threads>>>(
            static_cast<const __half*>(a),
            static_cast<const __half*>(b),
            static_cast<__half*>(output),
            n
        );
    }
}

}  // extern "C"
