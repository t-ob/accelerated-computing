---
title: "Host scalar product (AVX)"
slug: host-scalar-product-avx
weight: 105
draft: false
---
## Host scalar product (AVX)

```c++
#include <cstdlib>
#include <immintrin.h> // AVX intrinsics

float scalarProductAVX256(const float* v1, const float* v2, const size_t N) {
    __m256 result = _mm256_setzero_ps(); // Initialize result vector with zeros
    for (size_t i = 0; i < N; i += 8) {
        __m256 a = _mm256_load_ps(v1 + i); // Load 8 consecutive elements from v1
        __m256 b = _mm256_load_ps(v2 + i); // Load 8 consecutive elements from v2
        __m256 prod = _mm256_mul_ps(a, b); // Multiply elements
        result = _mm256_add_ps(result, prod); // Add to result
    }
    // Extract final result from the AVX512 register
    alignas(32) float output[8];
    _mm256_store_ps(output, result);
    return output[0] + output[1] + output[2] + output[3] + output[4] + output[5] + output[6] + output[7];
}

int main() {
    const int RUNS = 500;

    const int N = 1<<28;
    size_t size = N * sizeof(float);

    float *x, *y;

    x = (float*) std::aligned_alloc(32, size);
    y = (float*) std::aligned_alloc(32, size);

    for (int i = 0; i < N; i++) {
        x[i] = 1.0;
        y[i] = 2.0;
    }

    // Mark as volatile so the compiler doesn't optimise out the loop
    volatile float result;
    for(int i = 0; i < RUNS; ++i) {
        result = scalarProductAVX256(x, y, N);
    }

    return 0;
}
```