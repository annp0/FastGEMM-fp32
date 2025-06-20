# FastGEMM

Fast GPU-Accelerated General Matrix Multiplication (CUDA)

This implementation reaches about 70% of cuBLAS's performance on a NVIDIA T4.

```
------ BENCHMARK RESULT ------
CUBLAS: MIN=21.362ms, MAX=42.3843ms
KERNEL: MIN=33.038 ms, MAX=45.0127 ms
CUBLAS / KERNEL = 28.25 ms / 38.66 ms = 73.08%
------------------------------
```

To run the benchmark, you need to have a CUDA-capable GPU and the NVIDIA CUDA toolkit installed. Clone the repository and run the following command:

```bash
make run
```

This will compile the code and execute the benchmark.

## Optimizations

- Tiling on all levels (blocks, warps, threads). Each level of tiling helps to improve memory access patterns, allows for better cache, increases arithmetic intensity (the unoptimized version was mainly memory-bound), and hides memory latency (with scheduling).
- Within each blocktile, values are vector-loaded into shared memory with `float4` (128b loads) in parallel.
- Within each warptile, its threads' memory accesses to shared memory are also coalesced (which leads to 128b loads as well).
- For each thread, we adjust the loop structure to increase value reuse on the register file and reduce memory accesses.

## Other Possible Optimizations

- Resolving bank conflicts within each warp.
- Double buffering to further hide memory latency.

Special Thanks to Simon Boehm, for the blog post on MatMul. 

