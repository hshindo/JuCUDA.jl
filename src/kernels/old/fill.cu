// fill_float fill_double

#include "array.h"

#define FILL_CAPI(T) \
__global__ void fill ## _ ## T(Array<T,1> x, T value) { \
  int idx = threadIdx.x + blockIdx.x * blockDim.x; \
  if (idx < x.length()) x[idx] = value; \
}

extern "C" {
  FILL_CAPI(float)
  FILL_CAPI(double)
}
