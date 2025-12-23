/**
 * C++ binding for fused_add kernel
 */

#include <torch/extension.h>
#include <cuda_runtime.h>

extern "C" {
void fused_add_cuda_forward(
    const void* a,
    const void* b,
    void* output,
    int n,
    int dtype_size,
    bool use_vectorized
);
}

torch::Tensor fused_add_cuda(
    torch::Tensor a,
    torch::Tensor b,
    bool use_vectorized = true
) {
    // Input validation
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have same shape");
    TORCH_CHECK(a.dtype() == b.dtype(), "a and b must have same dtype");
    
    // Explicitly check and reject BF16
    TORCH_CHECK(a.dtype() != torch::kBFloat16, 
                "BF16 is not supported. Please use FP16 or FP32.");
    TORCH_CHECK(a.dtype() == torch::kFloat16 || a.dtype() == torch::kFloat32,
                "Only FP16 and FP32 are supported, got ", a.dtype());
    
    // Force contiguous to handle non-contiguous tensors correctly
    // This is CRITICAL - without it, raw pointer access gives wrong results
    a = a.contiguous();
    b = b.contiguous();
    
    auto output = torch::empty_like(a);
    int n = a.numel();
    int dtype_size = a.element_size();
    
    // Check alignment for vectorized path
    bool is_aligned = (reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0) &&
                      (reinterpret_cast<uintptr_t>(b.data_ptr()) % 16 == 0) &&
                      (reinterpret_cast<uintptr_t>(output.data_ptr()) % 16 == 0);
    
    // Only use vectorized if: enabled, aligned, divisible by 4, and FP32
    bool can_vectorize = use_vectorized && is_aligned && (n % 4 == 0) && (dtype_size == 4);
    
    fused_add_cuda_forward(
        a.data_ptr(),
        b.data_ptr(),
        output.data_ptr(),
        n,
        dtype_size,
        can_vectorize
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel error: ", cudaGetErrorString(err));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_add", &fused_add_cuda, 
          "Fused element-wise add (CUDA)",
          py::arg("a"),
          py::arg("b"),
          py::arg("use_vectorized") = true);
}
