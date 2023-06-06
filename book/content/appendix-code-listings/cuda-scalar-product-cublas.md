---
title: "CUDA scalar product (cuBLAS)"
slug: cuda-scalar-product-cublas
weight: 103
draft: false
---
## CUDA scalar product (cuBLAS)

```c++
#include <iostream>
#include <cublas_v2.h>
#include <cuda.h>

int main() {
    const int RUNS = 500;

    // Initialize cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    const int N = 1<<28;
    size_t size = N * sizeof(float);

    float *h_x, *h_y;
    h_x = (float*)malloc(size);
    h_y = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_x[i] = 1.0;
        h_y[i] = 2.0;
    }

    float *d_x, *d_y;
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);

    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);

    float result;
    for(int i = 0; i < RUNS; ++i) {
        cublasSdot(handle, N, d_x, 1, d_y, 1, &result);
    }

    // Clean up and return
    cublasDestroy(handle);
    cudaFree(d_x);
    cudaFree(d_y);
    free(h_x);
    free(h_y);

    return 0;
}
```