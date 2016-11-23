// exp_float exp_double log_float log_double
// mul_float mul_double add_float add_double sub_float sub_double

#include "array.h"

#define ELEMWISE_UNARY_CAPI(NAME, T, OP) \
__global__ void NAME ## _ ## T(Array<T,1> x, Array<T,1> y) { \
    int idx = threadIdx.x + blockIdx.x * blockDim.x; \
    if (idx >= x.length()) return; \
    y[idx] = OP(x[idx]); \
}

extern "C" {
    ELEMWISE_UNARY_CAPI(exp, float, exp)
    ELEMWISE_UNARY_CAPI(exp, double, exp)
    ELEMWISE_UNARY_CAPI(log, float, log)
    ELEMWISE_UNARY_CAPI(log, double, log)
}

#define ELEMWISE_BINARY_CAPI(NAME, T, OP) \
__global__ void NAME ## _ ## T(Array<T,1> x1, Array<T,1> x2, Array<T,1> y) { \
    int idx = threadIdx.x + blockIdx.x * blockDim.x; \
    if (idx >= y.length()) return; \
    int idx_x1 = idx < x1.length() ? idx : idx % x1.length(); \
    int idx_x2 = idx < x2.length() ? idx : idx % x2.length(); \
    y[idx] = x1[idx_x1] OP x2[idx_x2]; \
}

extern "C" {
    ELEMWISE_BINARY_CAPI(mul, float, *)
    ELEMWISE_BINARY_CAPI(mul, double, *)
    ELEMWISE_BINARY_CAPI(add, float, +)
    ELEMWISE_BINARY_CAPI(add, double, +)
    ELEMWISE_BINARY_CAPI(sub, float, -)
    ELEMWISE_BINARY_CAPI(sub, double, -)
}
