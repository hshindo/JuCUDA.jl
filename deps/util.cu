template <typename T>
__device__ void fill(T *x, size_t n, T value) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = idx; i < n; i += gridDim.x * blockDim.x) x[i] = value;
}

template <typename T>
__device__ void axpy(T a, T *x, T *y, size_t n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = idx; i < n; i += gridDim.x * blockDim.x) y[i] += a * x[i];
}

#define FILL_C(T) \
  __global__ void fill ## _ ## T(T *x, size_t n, T value) { fill(x, n, value); }

#define AXPY_C(T) \
  __global__ void axpy ## _ ## T(T a, T *x, T *y, size_t n) { axpy(a, x, y, n); }

extern "C" {
  FILL_C(float)
  FILL_C(double)
  AXPY_C(float)
  AXPY_C(double)
}
