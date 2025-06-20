#pragma once

#include <cublas_v2.h>

void run_cublas(cublasHandle_t handle, int M, int N, int K, float alpha, float beta, float *A, float *B, float *C)
{
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F, N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}