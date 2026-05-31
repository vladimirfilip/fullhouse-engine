#include "network_cuda.cuh"
#include "config.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

// ── Error helpers ─────────────────────────────────────────────────────────────

static void cuda_throw(cudaError_t e, const char* fn) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(fn) + ": " + cudaGetErrorString(e));
}
static void cublas_throw(cublasStatus_t s, const char* fn) {
    if (s != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(fn) + " cuBLAS error " + std::to_string((int)s));
}
#define CUDA_CHK(x)   cuda_throw((x),   #x)
#define CUBLAS_CHK(x) cublas_throw((x), #x)

// ── Kernels ───────────────────────────────────────────────────────────────────
//
// cuBLAS GEMM result layout: col-major [out_dim × batch].
// Element (row=d, col=b) lives at Y[d + b*out_dim].
// Equivalently, reading Y as row-major [batch × out_dim] gives Y[b,d] at the
// same offset — so no explicit transpose is needed after the GEMM.
//
// Bias b[d] must be added to every element whose row index = d,
// i.e. idx % out_dim == d.

__global__ void k_bias_leaky_relu(float* __restrict__ Y,
                                   const float* __restrict__ bias,
                                   int out_dim, int total, float alpha)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float v = Y[idx] + bias[idx % out_dim];
    Y[idx] = (v < 0.f) ? alpha * v : v;
}

__global__ void k_bias(float* __restrict__ Y,
                        const float* __restrict__ bias,
                        int out_dim, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    Y[idx] += bias[idx % out_dim];
}

// ── CudaMLPImpl ──────────────────────────────────────────────────────────────

struct CudaLayer {
    float* W_dev   = nullptr;
    float* b_dev   = nullptr;
    int    in_dim  = 0;
    int    out_dim = 0;
};

struct CudaMLPImpl {
    cublasHandle_t         handle;
    std::vector<CudaLayer> layers;
    float* act_a   = nullptr;  // ping buffer [max_batch × max_hidden_dim]
    float* act_b   = nullptr;  // pong buffer
    float* in_dev  = nullptr;  // [max_batch × in_dim0]
    float* out_dev = nullptr;  // [max_batch × out_dimN]
    int    max_batch = 0;
};

// ── Factory ──────────────────────────────────────────────────────────────────

CudaMLPImpl* cuda_mlp_create(
    const std::vector<const float*>& weight_ptrs,
    const std::vector<std::pair<int,int>>& dims,
    int max_batch)
{
    int n_layers = (int)dims.size();
    if ((int)weight_ptrs.size() != n_layers * 2)
        throw std::runtime_error("cuda_mlp_create: weight_ptrs.size() != n_layers*2");

    auto* mlp = new CudaMLPImpl{};
    mlp->max_batch = max_batch;
    CUBLAS_CHK(cublasCreate(&mlp->handle));

    mlp->layers.resize(n_layers);
    for (int i = 0; i < n_layers; i++) {
        auto [out_dim, in_dim] = dims[i];
        auto& l   = mlp->layers[i];
        l.in_dim  = in_dim;
        l.out_dim = out_dim;

        CUDA_CHK(cudaMalloc(&l.W_dev, (size_t)out_dim * in_dim * sizeof(float)));
        CUDA_CHK(cudaMalloc(&l.b_dev, (size_t)out_dim           * sizeof(float)));
        CUDA_CHK(cudaMemcpy(l.W_dev, weight_ptrs[i*2],
                            (size_t)out_dim * in_dim * sizeof(float),
                            cudaMemcpyHostToDevice));
        CUDA_CHK(cudaMemcpy(l.b_dev, weight_ptrs[i*2+1],
                            (size_t)out_dim * sizeof(float),
                            cudaMemcpyHostToDevice));
    }

    // Ping-pong buffers sized to the widest hidden dim.
    int max_mid = 0;
    for (int i = 0; i < n_layers - 1; i++)
        max_mid = std::max(max_mid, dims[i].first);

    CUDA_CHK(cudaMalloc(&mlp->in_dev,
                        (size_t)max_batch * dims[0].second * sizeof(float)));
    CUDA_CHK(cudaMalloc(&mlp->out_dev,
                        (size_t)max_batch * dims[n_layers-1].first * sizeof(float)));
    if (max_mid > 0) {
        CUDA_CHK(cudaMalloc(&mlp->act_a, (size_t)max_batch * max_mid * sizeof(float)));
        CUDA_CHK(cudaMalloc(&mlp->act_b, (size_t)max_batch * max_mid * sizeof(float)));
    }

    return mlp;
}

// ── Forward pass ─────────────────────────────────────────────────────────────
//
// We want:  Y_row[batch × out_dim] = X_row[batch × in_dim] * W_row^T[in_dim × out_dim]
//
// cuBLAS is column-major. Treat row-major matrices as transposed column-major:
//   X_row[batch × in_dim]  ==  X^T_col[in_dim × batch]    (op = CUBLAS_OP_N, lda=in_dim)
//   W_row[out_dim × in_dim] ==  W^T_col[in_dim × out_dim]  (op = CUBLAS_OP_T, lda=in_dim)
//
// cuBLAS call: C = op(A) * op(B)
//   m = out_dim, n = batch, k = in_dim
//   op(A) = CUBLAS_OP_T on W_dev  →  W_col[out_dim × in_dim]
//   op(B) = CUBLAS_OP_N on X_dev  →  X^T_col[in_dim × batch]
//   C_col[out_dim × batch]  ==  Y_row[batch × out_dim] in the same memory.

void cuda_mlp_forward(CudaMLPImpl* mlp, const float* host_in,
                      float* host_out, int batch)
{
    int n_layers = (int)mlp->layers.size();
    int in0_dim  = mlp->layers[0].in_dim;

    CUDA_CHK(cudaMemcpy(mlp->in_dev, host_in,
                        (size_t)batch * in0_dim * sizeof(float),
                        cudaMemcpyHostToDevice));

    const float one = 1.f, zero = 0.f;
    constexpr int THREADS = 256;

    float* in_ptr = mlp->in_dev;
    float* cur    = mlp->act_a;
    float* nxt    = mlp->act_b;

    for (int li = 0; li < n_layers; li++) {
        const auto& l    = mlp->layers[li];
        bool        last = (li == n_layers - 1);
        float* out_ptr   = last ? mlp->out_dev : cur;

        CUBLAS_CHK(cublasSgemm(mlp->handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            l.out_dim, batch, l.in_dim,
            &one,
            l.W_dev, l.in_dim,
            in_ptr,  l.in_dim,
            &zero,
            out_ptr, l.out_dim));

        int total  = l.out_dim * batch;
        int blocks = (total + THREADS - 1) / THREADS;
        if (last)
            k_bias<<<blocks, THREADS>>>(out_ptr, l.b_dev, l.out_dim, total);
        else
            k_bias_leaky_relu<<<blocks, THREADS>>>(out_ptr, l.b_dev,
                                                    l.out_dim, total, LEAKY_ALPHA);
        CUDA_CHK(cudaGetLastError());

        in_ptr = out_ptr;
        if (!last) std::swap(cur, nxt);
    }

    CUDA_CHK(cudaDeviceSynchronize());

    int out_dim = mlp->layers.back().out_dim;
    CUDA_CHK(cudaMemcpy(host_out, mlp->out_dev,
                        (size_t)batch * out_dim * sizeof(float),
                        cudaMemcpyDeviceToHost));
}

// ── Cleanup ───────────────────────────────────────────────────────────────────

void cuda_mlp_destroy(CudaMLPImpl* mlp) {
    if (!mlp) return;
    for (auto& l : mlp->layers) {
        cudaFree(l.W_dev);
        cudaFree(l.b_dev);
    }
    cudaFree(mlp->act_a);
    cudaFree(mlp->act_b);
    cudaFree(mlp->in_dev);
    cudaFree(mlp->out_dev);
    cublasDestroy(mlp->handle);
    delete mlp;
}
