---
title: "CUDA scalar product"
slug: cuda-scalar-product-naive
weight: 102
draft: false
---
## CUDA scalar product (naive)

```c++
#include <iostream>
#include <cuda.h>

__global__ void pairwiseProducts(const float* input_x, const float* input_y, const int N, float *output) {
    // Get the global thread ID
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Check if the thread ID is within the range of N
    if (idx < N) {
        // Compute the product of the corresponding elements from x and y
        output[idx] = input_x[idx] * input_y[idx];
    }
}

// CUDA kernel for parallel reduction
__global__ void parallelSum(const float* input, const int N, float* output) {
    // Define an external shared array accessible by all threads in a block
    extern __shared__ float sdata[];

    // Get the global and local thread ID
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int tid = threadIdx.x;

    // Load data from global to shared memory
    sdata[tid] = (idx < N) ? input[idx] : 0;

    // Sync all threads in the block
    __syncthreads();

    // Do reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        // Make sure all additions at the current stage are done
        __syncthreads();
    }

    // Write result of this block to global memory
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

int main() {
    const int RUNS = 500;

    const int N = 1<<28;
    size_t size = N * sizeof(float);
    const int BLOCK_SIZE = 256;

    float *h_x, *h_y;
    h_x = (float*)malloc(size);
    h_y = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_x[i] = 1.0;
        h_y[i] = 2.0;
    }

    float *d_x, *d_y, *d_z;
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);
    cudaMalloc(&d_z, size);

    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);

    float h_out;
    float* d_out;

    for(int i = 0; i < RUNS; ++i) {
        pairwiseProducts<<<N/BLOCK_SIZE, BLOCK_SIZE>>>(d_x, d_y, N, d_z);
        cudaDeviceSynchronize();

        // Compute the product of the two vectors.
        pairwiseProducts<<<N/BLOCK_SIZE, BLOCK_SIZE>>>(d_x, d_y, N, d_z);
        cudaDeviceSynchronize();

        // Compute the sum of the products with parallel reduction.
        int numElements = N;
        float* d_in = d_z;

        while(numElements > 1) {
            int numBlocks = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
            cudaMalloc(&d_out, numBlocks*sizeof(float));

            parallelSum<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE*sizeof(float)>>>(d_in, numElements, d_out);
            cudaDeviceSynchronize();

            if (d_in != d_z) {  // Don't free the original input array.
                cudaFree(d_in);
            }

            d_in = d_out;
            numElements = numBlocks;
        }
    }

    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_out);
    free(h_x);
    free(h_y);

    return 0;
}
```