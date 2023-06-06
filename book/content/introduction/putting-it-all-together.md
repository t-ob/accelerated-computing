---
title: "Putting it all together"
slug: putting-it-all-together
date: 2023-06-07T15:40:25+01:00
weight: 14
draft: false
---
## Putting it all together

At this point, we've written the kernels which will do our heavy lifting. We still need to call these from the host. Let's put this all together in a complete program. It does the following:
1. Declares a block size of 256.
2. Creates two constant vectors of length 1048576 and copies them to the GPU.
3. Allocates additional memory on the GPU to store the output of their pairwise component products.
4. Launches the first kernel `pairwiseProducts` to compute and store these products.
5. Repeatedly calls the second kernel `parallelSum`, at each stage halving the number of blocks to consider. Each loop also allocates memory on the device to store intermediate results.
6. Copies the result from the device back to the host and prints it out.
7. Cleans up any allocated memory on both host and device.

```c++
#include <iostream>
#include <cuda.h>

// ...
// CUDA kernels defined above
// ...

int main() {
    const int N = 1<<20;
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

    // Compute the product of the two vectors.
    pairwiseProducts<<<N/BLOCK_SIZE, BLOCK_SIZE>>>(d_x, d_y, d_z, N);
    cudaDeviceSynchronize();

    // Compute the sum of the products with parallel reduction.
    int numElements = N;
    float* d_in = d_z;
    float* d_out;

    while(numElements > 1) {
        int numBlocks = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        cudaMalloc(&d_out, numBlocks*sizeof(float));

        parallelSum<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE*sizeof(float)>>>(d_in, d_out, numElements);
        cudaDeviceSynchronize();

        if (d_in != d_z) {  // Don't free the original input array.
            cudaFree(d_in);
        }

        d_in = d_out;
        numElements = numBlocks;
    }

    float h_out;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "The dot product is: " << h_out << std::endl;

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_out);
    free(h_x);
    free(h_y);

    return 0;
}
```

I won't cover everything line by line, but let's look a bit at memory management and kernel execution.

### Memory management

The CUDA SDK provides an API for memory management between host and device. A full survey is beyond the scope of this article, but we use some of its basic functionality here. For example, the following code allocates space for three floating point arrays on the device with `cudaMalloc`, and copies the input from the host with `cudaMemcpy` with kind `cudaMemcpyHostToDevice` (copying from device to host is later accomplished with `cudaMemcpyDeviceToHost`).

```c++
float *d_x, *d_y, *d_z;
cudaMalloc(&d_x, size);
cudaMalloc(&d_y, size);
cudaMalloc(&d_z, size);

cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);
```

Similarly, like `free`, memory on device can be released once it's no longer needed:

```c++
cudaFree(d_x);
cudaFree(d_y);
cudaFree(d_z);
cudaFree(d_out);
```

### Kernel execution

We execute our kernels in two places in the above program. First, to compute the pairwise products, with the following invocation:

```c++
pairwiseProducts<<<N/BLOCK_SIZE, BLOCK_SIZE>>>(d_x, d_y, d_z, N);
cudaDeviceSynchronize();
```

The expression `<<<N/BLOCK_SIZE, BLOCK_SIZE>>>` is the execution configuration for the kernel. Here, we are declaring a grid of 4096 blocks, each having 256 threads. The CUDA runtime itself is responsible for orchestrating how and when these execute. Launching a kernel is asynchronous by default and returns immediately, but we want to wait for this computation to finish, so we call `cudaDeviceSynchronize()` to wait.

Next, we have the summation itself:

```c++
// Compute the sum of the products with parallel reduction.
int numElements = N;
float* d_in = d_z;
float* d_out;

while(numElements > 1) {
    int numBlocks = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaMalloc(&d_out, numBlocks*sizeof(float));

    parallelSum<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE*sizeof(float)>>>(d_in, d_out, numElements);
    cudaDeviceSynchronize();

    if (d_in != d_z) {  // Don't free the original input array.
        cudaFree(d_in);
    }

    d_in = d_out;
    numElements = numBlocks;
}
```

Each iteration of this loop launches the `parallelSum` kernel with the configuration `<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE*sizeof(float)>>>`. The first two parameters are what we saw above -- the grid size and block size (respectively). The third parameter tells the CUDA runtime to allocate `BLOCK_SIZE*sizeof(float)` bytes of shared memory in each block. After the first iteration, `d_out` contains 4096 scalar sub-products. After the second iteration, it contains 16, and finally in the third iteration the final sum is computed, ready to be copied back to the host.