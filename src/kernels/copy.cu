// copy_float1 copy_float2 copy_float3 copy_float4 copy_float5
// copy_double1 copy_double2 copy_double3 copy_double4 copy_double5

#include "array.h"

template<typename T, int N>
__device__ void copy(Array<T,N> &dest, Array<T,N> &src) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= dest.length()) return;
  dest(idx) = src(idx);
}

#define COPY_CAPI(T,N) \
__global__ void copy ## _ ## T ## N(Array<T,N> dest, Array<T,N> src) { \
  int idx = threadIdx.x + blockIdx.x * blockDim.x; \
  if (idx >= dest.length()) return; \
  dest(idx) = src(idx); \
}

extern "C" {
  COPY_CAPI(float,1)
  COPY_CAPI(float,2)
  COPY_CAPI(float,3)
  COPY_CAPI(float,4)
  COPY_CAPI(float,5)
  COPY_CAPI(double,1)
  COPY_CAPI(double,2)
  COPY_CAPI(double,3)
  COPY_CAPI(double,4)
  COPY_CAPI(double,5)
}

/*
@nvrtc """
extern "C" __global__ void $(op)_$T(Array<$T,1> x, Array<$T,1> y) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < y.length()) y[idx] = $(op)(x[idx]);
}

extern "C" __global__ void $(op)_T(Array<$T,1> x1, Array<$T,1> x2, Array<$T,1> y) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= y.length()) return;
  int idx_x1 = idx < x1.length() ? idx : idx % x1.length();
  int idx_x2 = idx < x2.length() ? idx : idx % x2.length();
  y(idx) = x1[idx_x1] $op x2[idx_x2];
}
"""

@nvrtc """
extern "C" __global__ void copy1d(Array<$T,1> dest, Array<$T,1> src) {
    int idx0 = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx0 >= src.dims[0]) return;
    dest(idx0) = src(idx0);
}

extern "C" __global__ void copy2d(Array<$T,2> dest, Array<$T,2> src) {
    int idx0 = threadIdx.x + blockIdx.x * blockDim.x;
    int idx1 = threadIdx.y + blockIdx.y * blockDim.y;
    if (idx0 >= src.dims[0] || idx1 >= src.dims[1]) return;
    dest(idx0,idx1) = src(idx0,idx1);
}

extern "C" __global__ void copy3d(Array<$T,3> dest, Array<$T,3> src) {
    int idx0 = threadIdx.x + blockIdx.x * blockDim.x;
    int idx1 = threadIdx.y + blockIdx.y * blockDim.y;
    int idx2  = threadIdx.z + blockIdx.z * blockDim.z;
    if (idx0 >= src.dims[0] || idx1 >= src.dims[1] || idx2 >= src.dims[2]) return;
    dest(idx0,idx1,idx2) = src(idx0,idx1,idx2);
}

extern "C" __global__ void copynd(Array<$T,$N> dest, Array<$T,$N> src) {

}
*/
