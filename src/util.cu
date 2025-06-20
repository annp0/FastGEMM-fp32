#pragma once

#include <cstdlib>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>

void random_matrix(float *mat, int M, int N)
{

	const auto now = std::chrono::high_resolution_clock::now();
	const auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();

	auto rng = std::mt19937(static_cast<uint32_t>(nanos));
	auto dist = std::uniform_real_distribution<float>(0.0f, 1.0f);

	for (int i = 0; i < M * N; i++)
	{
		mat[i] = dist(rng);
	}
}

void print_matrix(const float *mat, const int M, const int N)
{
	std::cout << "[PRINT] ---------" << std::endl;
	for (int i = 0; i < M; i++)
	{
		std::cout << "[PRINT]";
		for (int j = 0; j < N; j++)
		{
			std::cout << std::fixed
					  << std::setprecision(4)
					  << std::setw(8)
					  << mat[i * N + j]
					  << " ";
		}
		std::cout << std::endl;
	}
	std::cout << "[PRINT] ---------" << std::endl;
}

bool is_matrix_equal(const float *mat1, const float *mat2, const int M, const int N)
{
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			if (abs(mat1[i * N + j] - mat2[i * N + j]) > 1e-3)
				return false;
		}
	}
	return true;
}