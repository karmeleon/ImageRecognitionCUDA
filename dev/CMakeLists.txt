### CMakeLists.txt for CUDA

cmake_minimum_required(VERSION 2.8)
find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-arch=sm_20")
cuda_add_executable(blah main.cu features.cu haarfinder.cu dataloader.cu common.cuh sat.cu)