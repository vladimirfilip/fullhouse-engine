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
  add_compile_options(-O3 -march=native -ffast-math -Wall -Wextra -Wno-unused-parameter)
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
