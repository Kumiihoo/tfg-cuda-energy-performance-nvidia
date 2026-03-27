cmake_minimum_required(VERSION 3.18)
project(tfg_cuda_bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Default targets for your comparison platform:
# - Quadro RTX 5000 -> sm_75
# - A100            -> sm_80
# Can be overridden with:
# cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=75
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75;80)
endif()
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

add_executable(bench
  src/main.cu
  src/device_info.cu
  src/bw_bench.cu
  src/compute_bench.cu
  src/gemm_bench.cu
  src/fft_bench.cu
)

target_compile_options(bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(bench PRIVATE CUDA::cublas CUDA::cufft)
