#define _CRT_SECURE_NO_WARNINGS
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "network_cuda.cuh"
#include "config.hpp"

namespace py = pybind11;
using farray = py::array_t<float, py::array::c_style | py::array::forcecast>;

// Extract layer dims and raw host pointers from the flat weights list
// [W0, b0, W1, b1, ...] (same layout as binding.cpp's mlp_from_weights).
static farray forward_batch_impl(const std::vector<farray>& weights,
                                  farray inputs,
                                  int max_batch)
{
    int n_layers = (int)weights.size() / 2;
    if ((int)weights.size() % 2 != 0)
        throw std::runtime_error("weights must be even-length [W0,b0,W1,b1,...]");

    std::vector<std::pair<int,int>> dims;
    std::vector<const float*>       ptrs;
    dims.reserve(n_layers);
    ptrs.reserve(n_layers * 2);

    for (int i = 0; i < n_layers; i++) {
        auto W_info = weights[i*2].request();
        auto b_info = weights[i*2+1].request();
        if (W_info.ndim != 2)
            throw std::runtime_error("Each W must be 2-D");
        int out_dim = (int)W_info.shape[0];
        int in_dim  = (int)W_info.shape[1];
        if ((int)b_info.size != out_dim)
            throw std::runtime_error("bias length must equal W rows");
        dims.push_back({out_dim, in_dim});
        ptrs.push_back(static_cast<const float*>(W_info.ptr));
        ptrs.push_back(static_cast<const float*>(b_info.ptr));
    }

    auto in_info = inputs.request();
    if (in_info.ndim != 2)
        throw std::runtime_error("inputs must be 2-D [batch x in_dim]");
    int batch   = (int)in_info.shape[0];
    int in_check = (int)in_info.shape[1];
    if (in_check != dims[0].second)
        throw std::runtime_error("inputs.shape[1] != first layer in_dim");

    if (max_batch < 0) max_batch = batch;
    if (batch > max_batch)
        throw std::runtime_error("batch exceeds max_batch");

    int out_dim = dims.back().first;
    auto result = py::array_t<float>({batch, out_dim});

    {
        py::gil_scoped_release release;
        CudaMLPImpl* mlp = cuda_mlp_create(ptrs, dims, max_batch);
        cuda_mlp_forward(mlp,
                         static_cast<const float*>(in_info.ptr),
                         static_cast<float*>(result.request().ptr),
                         batch);
        cuda_mlp_destroy(mlp);
    }
    return result;
}

PYBIND11_MODULE(deep_cfr_cuda, m) {
    m.doc() = "CUDA cuBLAS MLP forward pass for Deep CFR (Phase 0 parity / Phase 2 data gen)";

    m.attr("INPUT_DIM")  = INPUT_DIM;
    m.attr("N_ACTIONS")  = N_ACTIONS;
    m.attr("HIDDEN_DIM") = HIDDEN_DIM;
    m.attr("N_LAYERS")   = N_LAYERS;

    m.def("forward_batch",
          [](const std::vector<farray>& weights, farray inputs, int max_batch) {
              return forward_batch_impl(weights, inputs, max_batch);
          },
          py::arg("weights"),
          py::arg("inputs"),
          py::arg("max_batch") = -1,
          "Batched MLP forward pass via cuBLAS (linear output, no softmax).\n"
          "weights: [W0, b0, ..., Wn, bn] float32 numpy arrays.\n"
          "inputs:  [batch x in_dim] float32 numpy array.\n"
          "Returns: [batch x out_dim] float32 numpy array.");
}
