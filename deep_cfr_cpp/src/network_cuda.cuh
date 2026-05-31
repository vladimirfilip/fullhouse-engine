#pragma once
#include <vector>
#include <utility>

// Opaque handle — hides all CUDA/cuBLAS types from callers.
// CudaMLPImpl is defined in network_cuda.cu; callers only hold a pointer.
struct CudaMLPImpl;

// Create a batched inference MLP on the GPU.
//
// weight_ptrs: flat list of host float* [W0, b0, W1, b1, ...].
//   W_i is [out_dim_i x in_dim_i] row-major.  b_i is [out_dim_i].
// dims: {out_dim, in_dim} for each layer (same order as weight_ptrs).
// max_batch: number of inputs the ping-pong device buffers accommodate.
//
// Copies all weights to device.  Call cuda_mlp_destroy() when done.
CudaMLPImpl* cuda_mlp_create(
    const std::vector<const float*>& weight_ptrs,
    const std::vector<std::pair<int,int>>& dims,
    int max_batch);

// Batched forward pass.
//   host_in:  [batch x in_dim0] row-major, host pointer.
//   host_out: [batch x out_dimN] row-major, written on return.
//   batch:    must be <= max_batch passed to cuda_mlp_create.
// Activation: LeakyReLU(0.01) on all hidden layers; linear output layer.
// Blocks until device computation is complete (cudaDeviceSynchronize).
void cuda_mlp_forward(CudaMLPImpl* mlp, const float* host_in,
                      float* host_out, int batch);

void cuda_mlp_destroy(CudaMLPImpl* mlp);
