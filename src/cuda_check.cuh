#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

inline void cuda_check(const cudaError_t result, const char *file, const int line)
{
  if (result != cudaSuccess)
  {
    std::cerr << "[CUDA Error]: "
              << cudaGetErrorString(result)
              << " (error code " << result << ") at "
              << file << ":" << line << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

#define cudaCheck(cmd) cuda_check((cmd), __FILE__, __LINE__)
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

inline void cublasCheck(cublasStatus_t status)
{
  if (status != CUBLAS_STATUS_SUCCESS)
  {
    std::cerr << "cuBLAS Error: ";
    switch (status)
    {
    case CUBLAS_STATUS_NOT_INITIALIZED:
      std::cerr << "CUBLAS_STATUS_NOT_INITIALIZED";
      break;
    case CUBLAS_STATUS_ALLOC_FAILED:
      std::cerr << "CUBLAS_STATUS_ALLOC_FAILED";
      break;
    case CUBLAS_STATUS_INVALID_VALUE:
      std::cerr << "CUBLAS_STATUS_INVALID_VALUE";
      break;
    case CUBLAS_STATUS_ARCH_MISMATCH:
      std::cerr << "CUBLAS_STATUS_ARCH_MISMATCH";
      break;
    case CUBLAS_STATUS_MAPPING_ERROR:
      std::cerr << "CUBLAS_STATUS_MAPPING_ERROR";
      break;
    case CUBLAS_STATUS_EXECUTION_FAILED:
      std::cerr << "CUBLAS_STATUS_EXECUTION_FAILED";
      break;
    case CUBLAS_STATUS_INTERNAL_ERROR:
      std::cerr << "CUBLAS_STATUS_INTERNAL_ERROR";
      break;
    default:
      std::cerr << "UNKNOWN ERROR";
    }
    std::cerr << std::endl;
    exit(EXIT_FAILURE);
  }
}
