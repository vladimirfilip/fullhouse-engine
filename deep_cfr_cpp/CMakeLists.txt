cmake_minimum_required(VERSION 3.16)
project(deep_cfr_cpp CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Build type defaults ───────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ── Compiler flags ────────────────────────────────────────────────────────────
if(MSVC)
  add_compile_options(/O2 /arch:AVX2 /fp:fast /W3)
else()
  add_compile_options(
    -O3 -march=native -ffast-math
    # Allow inlining/devirtualisation across TUs in the shared lib.
    # Without this GCC/Clang assume any symbol could be LD_PRELOAD'd and refuse
    # to inline even with LTO, killing the forward_single hot path.
    -fno-semantic-interposition
    -funroll-loops        # explicit on older GCC where -O3 doesn't enable it
    -Wall -Wextra -Wno-unused-parameter
  )
endif()

# ── Eigen (header-only, fetched automatically) ─────────────────────────────
include(FetchContent)
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
  GIT_SHALLOW    TRUE
)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC     OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(eigen)

# ── Python (for pybind11) ─────────────────────────────────────────────────────
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# ── pybind11 (for Python extension module) ────────────────────────────────────
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.13.6
  GIT_SHALLOW    TRUE
)
set(PYBIND11_FINDPYTHON ON)
FetchContent_MakeAvailable(pybind11)

# ── Threads ───────────────────────────────────────────────────────────────────
find_package(Threads REQUIRED)

# ── Python extension module ───────────────────────────────────────────────────
# This module is the only build target — the canonical training loop lives in
# Python (bots/vlad/deep_cfr/train.py) and calls into the C++ MCCFR via pybind.
set(BINDING_SOURCES
  src/engine.cpp
  src/hand_eval.cpp
  src/features.cpp
  src/network.cpp
  src/mccfr.cpp
  src/train.cpp
  src/binding.cpp
)

pybind11_add_module(deep_cfr_gen ${BINDING_SOURCES})
target_include_directories(deep_cfr_gen PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(deep_cfr_gen PRIVATE Eigen3::Eigen Threads::Threads)
if(MSVC)
  target_compile_options(deep_cfr_gen PRIVATE /O2 /arch:AVX2 /fp:fast)
endif()

# ── CUDA extension (optional) ─────────────────────────────────────────────────
# Build with:  cmake -DENABLE_CUDA=ON ...  (or: make build-cpp-cuda)
# Produces:    deep_cfr_cuda.pyd / .so   — exposes forward_batch() for parity testing
#              and (Phase 2) the full MCCFR traversal engine.
option(ENABLE_CUDA "Build CUDA-accelerated deep_cfr_cuda extension (requires CUDA toolkit)" OFF)

if(ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  endif()

  pybind11_add_module(deep_cfr_cuda
    src/network_cuda.cu    # compiled by nvcc
    src/binding_cuda.cpp   # compiled by host CXX; calls cuda_mlp_* via opaque ptr
  )
  target_include_directories(deep_cfr_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(deep_cfr_cuda PRIVATE CUDA::cublas)

  # Allow constexpr / extended host-device lambdas when nvcc touches pybind11 headers.
  target_compile_options(deep_cfr_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )
  if(MSVC)
    target_compile_options(deep_cfr_cuda PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /O2,/W3>
    )
  else()
    target_compile_options(deep_cfr_cuda PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -O3,-fno-semantic-interposition>
    )
  endif()
  message(STATUS "CUDA extension enabled — target: deep_cfr_cuda")
endif()

# ── Link-Time Optimisation ────────────────────────────────────────────────────
# Enables cross-TU inlining: forward_single (network.cpp), build_feature_vector
# (features.cpp), and the Eigen GEMV can all be seen in one pass and inlined
# into the mccfr.cpp hot path. MSVC: /GL + /LTCG. GCC/Clang: -flto.
include(CheckIPOSupported)
check_ipo_supported(RESULT _ipo_ok OUTPUT _ipo_msg LANGUAGES CXX)
if(_ipo_ok)
  set_target_properties(deep_cfr_gen PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
  message(STATUS "LTO unavailable: ${_ipo_msg}")
endif()
