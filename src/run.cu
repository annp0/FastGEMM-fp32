#include <cstdlib>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "util.cu"
#include "cuda_check.cuh"
#include "cublas.cu"
#include "kernels/optimized.cu"

int main(int argc, char **argv)
{

	// Detect GPU (Device)
	int deviceIdx = 0;
	if (getenv("DEVICE") != nullptr)
	{
		deviceIdx = atoi(getenv("DEVICE"));
	}
	cudaCheck(cudaSetDevice(deviceIdx));

	std::cout << "[MAIN] RUNNING ON DEV " << deviceIdx << std::endl;

	// Set Parameters
	const float alpha = 0.5;
	const float beta = 0.3;
	const int M = 4096;

	std::cout << "[MAIN] DIM=" << M << ", ALPHA=" << alpha << ", BETA=" << beta << std::endl;

	// Allocate Memory
	float *h_A = nullptr, *h_B = nullptr, *h_C = nullptr, *h_Cr = nullptr,
		  *d_A = nullptr, *d_B = nullptr, *d_C = nullptr, *d_Cr = nullptr;
	h_A = (float *)malloc(M * M * sizeof(float));
	h_B = (float *)malloc(M * M * sizeof(float));
	h_C = (float *)malloc(M * M * sizeof(float));
	h_Cr = (float *)malloc(M * M * sizeof(float));

	random_matrix(h_A, M, M);
	random_matrix(h_B, M, M);
	random_matrix(h_C, M, M);

	std::cout << "[MAIN] MATRICES READY ON HOST" << std::endl;

	cudaCheck(cudaMalloc((void **)&d_A, M * M * sizeof(float)));
	cudaCheck(cudaMalloc((void **)&d_B, M * M * sizeof(float)));
	cudaCheck(cudaMalloc((void **)&d_C, M * M * sizeof(float)));
	cudaCheck(cudaMalloc((void **)&d_Cr, M * M * sizeof(float)));

	cudaCheck(cudaMemcpy(d_A, h_A, M * M * sizeof(float), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_B, h_B, M * M * sizeof(float), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_C, h_C, M * M * sizeof(float), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_Cr, h_C, M * M * sizeof(float), cudaMemcpyHostToDevice));

	std::cout << "[MAIN] MATRICES READY ON DEV" << std::endl;

	std::cout << "[MAIN] RUNNING WARMUP AND VERIFICATION" << std::endl;

	cublasHandle_t handle;
	cublasCheck(cublasCreate(&handle));

	run_cublas(handle, M, M, M, alpha, beta, d_A, d_B, d_Cr);

	run_sgemm_optimized(M, M, M, alpha, beta, d_A, d_B, d_C);

	cudaCheck(cudaGetLastError());
	cudaCheck(cudaDeviceSynchronize());

	std::cout << "[MAIN] WARMUP COMPLETE" << std::endl;

	cudaCheck(cudaMemcpy(h_C, d_C, M * M * sizeof(float), cudaMemcpyDeviceToHost));
	cudaCheck(cudaMemcpy(h_Cr, d_Cr, M * M * sizeof(float), cudaMemcpyDeviceToHost));

	if (!is_matrix_equal(h_C, h_Cr, M, M))
	{
		std::cout << "[MAIN] RESULT FAILED TO MATCH BASELINE" << std::endl;
		exit(1);
	}

	std::cout << "[MAIN] RESULT MATCHES BASELINE" << std::endl;

	const int iterations = 5;
	float cublas_times[iterations];
	float naive_times[iterations];

	cudaEvent_t start, stop;
	cudaCheck(cudaEventCreate(&start));
	cudaCheck(cudaEventCreate(&stop));
	float milliseconds = 0;

	std::cout << "[MAIN] BENCHMARKING ON STREAM 0 WITH ITERATION=" << iterations << std::endl;

	for (int i = 0; i < iterations; i++)
	{
		std::cout << "[MAIN] RUNNING ITERATION " << i << std::endl;
		cudaCheck(cudaMemcpy(d_C, h_C, M * M * sizeof(float), cudaMemcpyHostToDevice));
		cudaCheck(cudaMemcpy(d_Cr, h_C, M * M * sizeof(float), cudaMemcpyHostToDevice));

		// Time cuBLAS
		cudaCheck(cudaEventRecord(start));
		run_cublas(handle, M, M, M, alpha, beta, d_A, d_B, d_Cr);
		cudaCheck(cudaEventRecord(stop));
		cudaCheck(cudaEventSynchronize(stop));
		cudaCheck(cudaEventElapsedTime(&milliseconds, start, stop));
		cublas_times[i] = milliseconds;

		// Time native implementation
		cudaCheck(cudaEventRecord(start));

		run_sgemm_optimized(M, M, M, alpha, beta, d_A, d_B, d_C);

		cudaCheck(cudaEventRecord(stop));
		cudaCheck(cudaEventSynchronize(stop));
		cudaCheck(cudaEventElapsedTime(&milliseconds, start, stop));
		naive_times[i] = milliseconds;
	}

	// Calculate statistics
	float cublas_avg = 0.0f, naive_avg = 0.0f;
	float cublas_min = cublas_times[0], cublas_max = cublas_times[0];
	float naive_min = naive_times[0], naive_max = naive_times[0];

	for (int i = 0; i < iterations; i++)
	{
		cublas_avg += cublas_times[i];
		naive_avg += naive_times[i];

		cublas_min = std::min(cublas_min, cublas_times[i]);
		cublas_max = std::max(cublas_max, cublas_times[i]);
		naive_min = std::min(naive_min, naive_times[i]);
		naive_max = std::max(naive_max, naive_times[i]);
	}
	cublas_avg /= iterations;
	naive_avg /= iterations;

	float cublas_stddev = 0.0f, naive_stddev = 0.0f;
	for (int i = 0; i < iterations; i++)
	{
		cublas_stddev += (cublas_times[i] - cublas_avg) * (cublas_times[i] - cublas_avg);
		naive_stddev += (naive_times[i] - naive_avg) * (naive_times[i] - naive_avg);
	}
	cublas_stddev = std::sqrt(cublas_stddev / iterations);
	naive_stddev = std::sqrt(naive_stddev / iterations);

	const float confidence_factor = 2.0f;
	bool is_confident = true;

	if (cublas_min < cublas_avg - confidence_factor * cublas_stddev ||
		cublas_max > cublas_avg + confidence_factor * cublas_stddev ||
		naive_min < naive_avg - confidence_factor * naive_stddev ||
		naive_max > naive_avg + confidence_factor * naive_stddev)
	{
		is_confident = false;
	}

	if (is_confident)
	{
		std::cout << "[MAIN] RESULT CONFIDENT" << std::endl;
	}
	else
	{
		std::cout << "[MAIN] RESULT NOT CONFIDENT" << std::endl;
		exit(1);
	}

	std::cout << "[MAIN] ------ BENCHMARK RESULT ------" << std::endl;
	std::cout << "[MAIN] BASELINE: MIN=" << cublas_min << "ms, MAX=" << cublas_max
			  << "ms" << std::endl;
	std::cout << "[MAIN] KERNEL: MIN=" << naive_min << " ms, MAX=" << naive_max
			  << " ms" << std::endl;
	std::cout << "[MAIN] BASELINE / KERNEL = "
			  << std::fixed << std::setprecision(2) << cublas_avg << " ms / "
			  << naive_avg << " ms = "
			  << (cublas_avg / naive_avg * 100.0f) << "%" << std::endl;
	std::cout << "[MAIN] ------------------------------" << std::endl;

	// Clean up timing events
	cudaCheck(cudaEventDestroy(start));
	cudaCheck(cudaEventDestroy(stop));

	std::cout << "[MAIN] RUNNING MEMORY CLEAN-UP" << std::endl;

	cublasCheck(cublasDestroy(handle));

	free(h_A);
	free(h_B);
	free(h_C);
	free(h_Cr);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaFree(d_Cr);
}
