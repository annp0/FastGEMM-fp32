cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_ARCHITECTURES 75)
project(GEMM LANGUAGES CUDA CXX)

include_directories(${CMAKE_SOURCE_DIR}/src)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_executable(run src/run.cu)
set_target_properties(run PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(run PRIVATE CUDA::cublas CUDA::cudart)