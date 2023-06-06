---
title: "Host scalar product (naive)"
slug: host-scalar-product-naive
weight: 104
draft: false
---
## Host scalar product (naive)

```c++
#include <cstdlib>

float scalarProductBasic(const float* v1, const float* v2, const size_t N) {
    float result = 0.0;
    for (size_t i = 0; i < N; ++i) {
        result += v1[i] * v2[i];
    }
    return result;
}

int main() {
    const int RUNS = 500;

    const int N = 1<<28;
    size_t size = N * sizeof(float);

    float *x, *y;
    x = (float*)malloc(size);
    y = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        x[i] = 1.0;
        y[i] = 2.0;
    }

    // Mark as volatile so the compiler doesn't optimise out the loop
    volatile float result;
    for(int i = 0; i < RUNS; ++i) {
        result = scalarProductBasic(x, y, N);
    }

    return 0;
}
```