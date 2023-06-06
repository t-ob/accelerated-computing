---
title: "CUDA concepts"
slug: cuda-concepts
date: 2023-06-07T15:40:25+01:00
weight: 12
draft: false
---
## CUDA concepts

By the end of this chapter we will have written a simple program which does the following:

1. Copies data from the host machine to a GPU.
2. Performs some computation on that data.
3. Copies the result back to the host.

Before we can do that, we need to cover some key concepts.

### Kernels

The CUDA programming model has at its core the concept of a `kernel` as its basic unit of execution. Concretely, a CUDA kernel is simply a function which is executed on the GPU. A kernel is written once, and orchestrated by the CUDA runtime to execute multiple times on the device[^host-device]. In CUDA C++, a kernel is written much like any other function, but they must return void and be defined with a `__global__` declaration. For example, the function

```c++
__global__ void kernel_function(const int *input, const int input_length, int *output) {
    // kernel body
}
```

declares itself a kernel function, and takes in a pointer to some input integers, their length, and a pointer to some memory address to write to. Typically, pointer arguments will be pointers to memory on the GPU itself.

### Threads, blocks, and grids

The CUDA programming model also provides a hierarchy of threads, blocks and grids:

1. Thread: the basic unit of execution. Each thread runs the same kernel function, and has a unique ID within its block.
2. Block: a group of threads that can cooperate with each other through shared memory. All threads in a block are guaranteed to run on the same Streaming Multiprocessor[^sm] (SM). Each block has its own unique ID within its grid.
3. Grid: a group of blocks that execute the same kernel function. Blocks in a grid can be scheduled on any SM and run independently in parallel. Each grid runs on a single device.

The CUDA SDK provides makes available three built-in variables to each thread, which can be used to determine both their unique global index, and index within a given block. These are `threadIdx`, `blockIdx`, and `blockDim`:

1. `threadIdx`: this built-in variable is a three-component vector (`threadIdx.x`, `threadIdx.y`, and `threadIdx.z`) that provides the unique ID for each thread within a block. The thread IDs are zero-based, which means they start from 0. If a block of threads is one-dimensional, you only need to use `threadIdx.x`. If it's two-dimensional, you can use `threadIdx.x` and `threadIdx.y`, and so on.

2. `blockIdx`: similar to threadIdx, blockIdx is also a three-component vector (`blockIdx.x`, `blockIdx.y`, and `blockIdx.z`) providing the unique ID for each block within a grid. The block IDs are also zero-based.

3. `blockDim`: this is a three-component vector (`blockDim.x`, `blockDim.y`, and `blockDim.z`) containing the number of threads in a block along each dimension. For example, if you have a block size of 128 threads, and you've organized it as a 4x32 two-dimensional block, blockDim.x would be 4 and blockDim.y would be 32.

When launching a kernel, at run time, you specify the number of blocks in the grid and the number of threads in each block. We go into more detail below, but for example, the kernel launch

```c++
kernel_function<<<16, 1024>>>( /* ... params ... */ )
```

results in `kernel_function` executing across a grid of 16 blocks, each having 1024 threads. Note that we could also pass in `dim3` types instead of integers; in this case they are implicitly converted to 1-dimensional representations.

Finally, we will also need to manage memory access and synchronisation between host and device. The CUDA SDK provides this functionality, as we'll see next.

[^host-device]: In the context of these posts, and generally in CUDA parlance, "device" refers to the physical hardware that executes a kernel (eg. my 2080 Super), and "host" refers to the machine which calls it.
[^sm]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-implementation
