#pragma once

#include <cuda_runtime.h>

#include "cuda_check.cuh"

const int WARPSIZE = 32;

template <const int BM, const int BN, const int BK, const int rowStrideA,
          const int rowStrideB>
__device__ void loadFromGmem(int N, int K, const float *A, const float *B,
                             float *As, float *Bs, int innerRowA, int innerColA,
                             int innerRowB, int innerColB) {
  for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
    const float4 tmp = reinterpret_cast<const float4 *>(
        &A[(innerRowA + offset) * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
  }

  for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
    reinterpret_cast<float4 *>(
        &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
        reinterpret_cast<const float4 *>(
            &B[(innerRowB + offset) * N + innerColB * 4])[0];
  }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void
processFromSmem(float *regM, float *regN, float *threadResults, const float *As,
                const float *Bs, const uint warpRow, const uint warpCol,
                const uint threadRowInWarp, const uint threadColInWarp) {
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint i = 0; i < TM; ++i) {
        regM[wSubRowIdx * TM + i] = 
            As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM + i];
      }
    }
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      for (uint i = 0; i < TN; ++i) {
        regN[wSubColIdx * TN + i] =
            Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + i];
      }
    }

    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
          for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) + (wSubColIdx * TN) + resIdxN] 
                += regM[wSubRowIdx * TM + resIdxM] * regN[wSubColIdx * TN + resIdxN];
          }
        }
      }
    }
  }
}


template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    sgemm_optimized(int M, int N, int K, float alpha, float *A, float *B,
                    float beta, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const uint warpIdx = threadIdx.x / WARPSIZE;
  const uint warpCol = warpIdx % (BN / WN);
  const uint warpRow = warpIdx / (BN / WN);

  constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr uint WSUBM = WM / WMITER; // 64/2=32
  constexpr uint WSUBN = WN / WNITER; // 32/2=16

  const uint threadIdxInWarp = threadIdx.x % WARPSIZE;
  const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);
  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  A += cRow * BM * K;
  B += cCol * BN;
  C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);
  constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

  float threadResults[WMITER * TM * WNITER * TN] = {0.0};
  float regM[WMITER * TM] = {0.0};
  float regN[WNITER * TN] = {0.0};

  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
    __syncthreads();
    processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>
        (regM, regN, threadResults, As, Bs, warpRow, warpCol, threadRowInWarp, threadColInWarp);
    A += BK; 
    B += BK * N;
    __syncthreads();
  }

  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          float4 tmp = reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N + threadColInWarp * TN + resIdxN])[0];
          const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) + wSubColIdx * TN + resIdxN;
          tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
          tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
          tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
          tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
          reinterpret_cast<float4 *>(&C_interim[(threadRowInWarp * TM + resIdxM) * N + threadColInWarp * TN + resIdxN])[0] = tmp;
        }
      }
    }
  }
}

void run_sgemm_optimized(int M, int N, int K, float alpha, float beta, float *A, float *B, float *C) {
  const uint NUM_THREADS = 128;
  const uint BN = 128;
  const uint BM = 128;
  const uint BK = 16;
  const uint WN = 64;
  const uint WM = 64;
  const uint WNITER = 4;
  const uint TN = 4;
  const uint TM = 8;
  dim3 blockDim(NUM_THREADS);
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  sgemm_optimized<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}