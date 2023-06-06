---
title: "Example: scalar product"
slug: example-scalar-product
date: 2023-06-07T15:40:25+01:00
weight: 13
draft: false
math: true
---

## Example: scalar product

We'll use the example of computing the scalar product of two floating point vectors. Recall that, for two vectors $x = (x_i)$ and $y = (y_i)$ of some $n$-dimensional vector space, their scalar (or dot) product $x \cdot y$ is the sum of the pairwise products of each vector's components:
$$
x \cdot y = \sum_{i=0}^{n - 1} x_i y_i
$$

The scalar product is a worthwhile place to start for two reasons:

1. Scalar products are everywhere. Matrix multiplications are just lots of scalar products, and machine learning is just lots of matrix multiplications.
2. It is straightforward to implement, but not so trivial that we can't learn anything from it.

We'll actually write two kernels -- one to compute the pairwise products, and a second to compute their sum.

A quick caveat: what follows is likely not the most efficient way to do this -- if you're serious about computing scalar products on a GPU, you should be using something like cuBLAS[^cublas]!

The first kernel looks like this:

```c++
// CUDA kernel to compute the pairwise product of vector elements
__global__ void pairwiseProducts(const float* input_x, const float* input_y, const int N, float *output) {
    // Get the global thread ID
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Check if the thread ID is within the range of N
    if (idx < N) {
        // Compute the product of the corresponding elements from a and b
        output[idx] = input_x[idx] * input_y[idx];
    }
}
```

The first line computes the global thread ID from the built-in variables described above. In this case, our data is one-dimensional, and (as we will see later), we'll launch the kernel with a one-dimensional configuration, so we only care about the `.x` attributes of each. Let's break it down a bit further:

1. `threadIdx.x`: this is the index of the executing thread within its block. There may be another thread in another block with the same index, so this is not globally unique.
2. `blockIdx.x * blockDim.x`: `blockIdx.x` is the index of the block in which the thread is executing, and `blockDim.x` is the number of threads in each block. In particular, `blockIdx.x` is always greater than `threadIdx.x`, so adding a multiple of `blockDim.x` is what makes the sum result in the globally unique index.

We next check if the thread has any work to do. If so, we set the output index to be the product of the inputs at the same index. As this index is unique, there are no race conditions to worry about.

Our second kernel is more interesting. Once we have computed our pairwise products, we now need to sum them. Recall above that we said a block of threads have access to shared memory -- we'll make use of that feature here. We'll proceed as follows:

1. Within a given block, copy a chunk (addressed by the global thread index) to some shared memory.
2. Wait for all threads to finish.
3. Consider a window over all the shared memory. Repeatedly add the right half of the window to the left half, then halve the window size and repeat until done.
4. The first element of the shared memory then contains the scalar sub-product of the thread block.

We may need to call this second kernel multiple times, to reduce the results of each block to a single number.

```c++
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
```

Similar to the first kernel, we use the expression `threadIdx.x + blockIdx.x * blockDim.x` to compute a thread's global index. There's quite a bit more going on here, though. For starters, the kernel begins with the line

```c++
extern __shared__ float sdata[];
```

This is where the block shared memory is declared. This needs a bit of unpacking -- from right to left:
1. `float sdata[]`: an array of floating point numbers.
2. `__shared__`: this keyword tells the CUDA compiler that the array should be placed in shared memory on the device.
3. `extern`: since we do not know the size of the shared memory array at compile time, we tell the compiler it will be defined elsewhere. Specifically, in a kernel launch configuration, we can optionally provide an optional `size_t` number of bytes of shared memory to be allocated per block (see below).

After thread indices established, the shared memory receives a copy of the input data addressed by the block:

```c++
// Load data from global to shared memory
sdata[tid] = (idx < n) ? input[idx] : 0;

// Sync all threads in the block
__syncthreads();
```

We call `__syncthreads()` here to ensure no thread proceeds past this point until the shared memory has been written.

Once the shared memory is populated, the sum occurs:

```c++
// Do reduction in shared memory
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
        sdata[tid] += sdata[tid + s];
    }
    // Make sure all additions at the current stage are done
    __syncthreads();
}

// Write result of this block to global memory
if (tid == 0) output[blockIdx.x] = sdata[0];
```

At the start of the loop we consider a window over the entire shared memory and mark its middle as `s`. The threads whose index fall in the right hand side of this window have no more work to do at this point, while the threads in the left hand side perform the sum. At each iteration of the loop, the working thread takes its index within the block, and updates the shared memory at that index with the corresponding value in the right half of the window. We repeatedly halve the length of the window until we're left with the sum of the block at index zero, at which point the thread with index zero copies it back out to global memory.

[^cublas]: https://docs.nvidia.com/cuda/cublas/