---
title: "Performance"
slug: performance
date: 2023-06-07T15:40:25+01:00
weight: 15
draft: false
---
## Performance

Now that we have our first CUDA program under our belt, we should ask ourselves: was it all worth it? After all, we have jumped through a fair few hoops to accomplish what can be expressed as a one-liner in Python:

```python
def scalar_product(xs, ys):
    return sum(x * y for x, y in zip(xs, ys))
```

This is a contrived example, but it illustrates the tradeoffs involved in CUDA programming, namely performance (which we are still yet to measure!) that comes at the cost of simplicity.

In the remainder of this chapter we will make a first attempt at measuring GPU performance. Specifically, a first attempt to get a sense of how fast (or not) our program is. We keep things deliberately light, as this introduction has gone on long enough already, and the topic of GPU performance and optimisation deserves (at least!) a chapter of its own.

We will compare our scalar product program to the following:

1. A naive host CPU-bound implementation in C++.
2. A second host CPU-bound implementation in C++ which uses AVX intrinsics.
3. A second CUDA program which leverages NVidia's cuBLAS library.

We are somewhat turning a blind eye toward scientific rigour here -- there are many variables at play, eg. the CPU, GPU, wider host configuration etc. -- but again, if there is one place amongst these articles where some light hand waving is appropriate, it's here in the introduction.

For reference, these experiments were carried out on a system running Ubuntu 20.04, running a Ryzen 9 3900X, and an NVidia RTX 2080 Super.

With that disclaimer out of the way, let's create some benchmark programs. Each will do two things:
1. Create two vectors of 268435456 `float`s.
2. Compute their scalar products 500 times.

Why 500 times? There are a couple of reasons, but mainly I wanted to approximate a "real" workload (even the most naive implementation will execute once in less than a second). Furthermore, as we shall see later, moving data from host to GPU is expensive, so we amortise that cost by running our GPU calculations multiple times.

What follows are the relevant parts of our benchmark programs. Complete listings can be found in the [Appendix]({{< relref "/appendix-code-listings" >}}).

### Benchmark program: CPU (naive)

This is a straightforward implementation which reads pairs of vector elements one at a time and accumulates the result:

```c++
float scalarProductBasic(const float* v1, const float* v2, const size_t N) {
    float result = 0.0;
    for (size_t i = 0; i < N; ++i) {
        result += v1[i] * v2[i];
    }
    return result;
}
```

### Benchmark program: CPU (AVX)

This implementation is similar to the above, but uses AVX-256[^avx-256] instructions to operate on multiple elements at a time:

```c++
#include <immintrin.h> // AVX intrinsics

float scalarProductAVX256(const float* v1, const float* v2, const size_t N) {
    __m256 result = _mm256_setzero_ps(); // Initialize result vector with zeros
    for (size_t i = 0; i < N; i += 8) {
        __m256 a = _mm256_load_ps(v1 + i); // Load 16 consecutive elements from v1
        __m256 b = _mm256_load_ps(v2 + i); // Load 16 consecutive elements from v2
        __m256 prod = _mm256_mul_ps(a, b); // Multiply elements
        result = _mm256_add_ps(result, prod); // Add to result
    }
    // Extract final result from the AVX512 register
    alignas(32) float output[8];
    _mm256_store_ps(output, result);
    return output[0] + output[1] + output[2] + output[3] + output[4] + output[5] + output[6] + output[7];
}
```

### Benchmark program: GPU (cuBLAS)

This program is similar to our hand-rolled scalar product implementation, though uses the optimised cuBLAS library to perform its heavy lifting. We include this program as it is likely to give us an indication of a performance upper bound:

```c++
#include <iostream>
#include <cublas_v2.h>
#include <cuda.h>

int main() {
    const int RUNS = 500;
    // Initialize cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Allocate vectors and copy to device
    // ...

    float result;
    for(int i = 0; i < RUNS; ++i) {
        cublasSdot(handle, N, d_x, 1, d_y, 1, &result);
    }

    // Clean up and return
    // ...

    return 0;
}
```

### Benchmark

We have four programs: `scalar_product_host_naive`, `scalar_product_host_avx`, `scalar_product_cuda_naive` (our implementation), and `scalar_product_cuda_cublas`. Each has been compiled (we will talk about what it even means to compile a CUDA program later - another topic worthy of its own chapter) with the highest optimisation level. Let's see how they compare, using everyone's favourite command line utility `time`:

```bash
$ time ./scalar_product_host_naive 

real    1m49.165s
user    1m48.664s
sys     0m0.492s
```

```bash
$ time ./scalar_product_host_avx 

real    0m41.073s
user    0m40.602s
sys     0m0.468s
```

```bash
$ time ./scalar_product_cuda_naive 

real    0m25.610s
user    0m24.954s
sys     0m0.631s
```

```bash
$ time ./scalar_product_cuda_cublas 

real    0m4.491s
user    0m3.670s
sys     0m0.820s
```

The optimised cuBLAS implementation's runtime is less than the naive implementation on my hardware by more than a factor of 20. Even our naive implementation does well against AVX intrinsics.

At this stage, we shouldn't read too much into these measurements. However, I hope they go some way to address the question at the top of this section: was it all worth it? For me, at least, the answer is an emphatic yes.

[^avx-256]: I would have used AVX-512 instructions but the CPU available to me only supports up AVX instructions up to and including the 256-bit variants.