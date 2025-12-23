/**
 * PyTorch C++ extension interface for fused LayerNorm + Residual kernel
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA forward declarations - note we declare them as templated functions
// that will be explicitly instantiated in the .cu file
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
);

void fused_layernorm_residual_cuda_forward_half(
    const void* x,
    const void* residual,
    const void* gamma,
    const void* beta,
    void* output,
    int blocks,
    int threads,
    int shared_mem_size,
    int hidden_size,
    float eps
);
}

// Wrapper function that dispatches to appropriate kernel based on dtype
torch::Tensor fused_layernorm_residual_cuda(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps
) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(residual.is_cuda(), "residual must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(), "gamma must be a CUDA tensor");
    TORCH_CHECK(beta.is_cuda(), "beta must be a CUDA tensor");
    
    TORCH_CHECK(x.dim() == 3, "x must be 3D tensor [batch, seq_len, hidden_size]");
    TORCH_CHECK(x.sizes() == residual.sizes(), "x and residual must have same shape");
    
    int batch = x.size(0);
    int seq_len = x.size(1);
    int hidden_size = x.size(2);
    int total_tokens = batch * seq_len;
    
    TORCH_CHECK(gamma.dim() == 1 && gamma.size(0) == hidden_size, 
                "gamma must be 1D tensor with size hidden_size");
    TORCH_CHECK(beta.dim() == 1 && beta.size(0) == hidden_size,
                "beta must be 1D tensor with size hidden_size");
    
    // Ensure contiguous memory layout
    x = x.contiguous();
    residual = residual.contiguous();
    gamma = gamma.contiguous();
    beta = beta.contiguous();
    
    // Allocate output tensor
    auto output = torch::empty_like(x);
    
    // Launch configuration
    // Each thread block processes one token
    int threads = std::min(hidden_size, 1024);
    int blocks = total_tokens;
    
    // Shared memory size: enough for warp reduction results
    int num_warps = (threads + 31) / 32;
    size_t shared_mem_size = num_warps * sizeof(float) + 2 * sizeof(float);  // +2 for mean and variance
    
    // Dispatch based on dtype
    if (x.scalar_type() == torch::kFloat32) {
        fused_layernorm_residual_cuda_forward_float(
            x.data_ptr<float>(),
            residual.data_ptr<float>(),
            gamma.data_ptr<float>(),
            beta.data_ptr<float>(),
            output.data_ptr<float>(),
            blocks,
            threads,
            shared_mem_size,
            hidden_size,
            eps
        );
    } else if (x.scalar_type() == torch::kFloat16) {
        fused_layernorm_residual_cuda_forward_half(
            x.data_ptr<at::Half>(),
            residual.data_ptr<at::Half>(),
            gamma.data_ptr<at::Half>(),
            beta.data_ptr<at::Half>(),
            output.data_ptr<at::Half>(),
            blocks,
            threads,
            shared_mem_size,
            hidden_size,
            eps
        );
    } else {
        AT_ERROR("Unsupported dtype, only float32 and float16 are supported");
    }
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch error: ", cudaGetErrorString(err));
    
    return output;
}

// Python binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_layernorm_residual", &fused_layernorm_residual_cuda,
          "Fused LayerNorm + Residual Add (CUDA)",
          py::arg("x"),
          py::arg("residual"),
          py::arg("gamma"),
          py::arg("beta"),
          py::arg("eps") = 1e-5);
}
