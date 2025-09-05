cmake_minimum_required(VERSION 3.18)  # 确保版本≥3.18
project(untitled1 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

#find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(untitled1 transpose.cu common.h)
target_link_libraries(untitled1 PRIVATE CUDA::cublas)
set_target_properties(untitled1 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


add_executable(main_gemm main_gemm.cu)
set_target_properties(main_gemm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)